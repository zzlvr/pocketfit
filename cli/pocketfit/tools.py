"""Locate or download the llama.cpp command line tools used for quantization and quality checks."""

import hashlib
import json
import os
import platform
import shutil
import stat
import subprocess
import tarfile
import urllib.request
import zipfile
from pathlib import Path

# Keep in sync with LLAMA_TAG in .github/workflows/ios-build.yml so the PC tools and the app agree.
LLAMA_TAG = "b12230"
LLAMA_COMMIT = "a4d4f2fd4ec85ec6a0a9ece4870416c1b6cc24f5"  # the commit that tag points to
LLAMA_REPO = "https://github.com/ggml-org/llama.cpp.git"
RELEASE_API = f"https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/{LLAMA_TAG}"
TEXT_URL = "https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip"
TEXT_MEMBER = "wikitext-2-raw/wiki.test.raw"


class ToolError(Exception):
    """A required tool could not be found, downloaded or verified."""


def home():
    return Path(os.environ.get("POCKETFIT_HOME") or Path.home() / ".pocketfit")


def _asset_name():
    system = platform.system()
    machine = platform.machine().lower()
    x86 = machine in ("amd64", "x86_64")
    if system == "Windows" and x86:
        return f"llama-{LLAMA_TAG}-bin-win-cpu-x64.zip"
    if system == "Darwin":
        return f"llama-{LLAMA_TAG}-bin-macos-{'x64' if x86 else 'arm64'}.tar.gz"
    if system == "Linux" and x86:
        return f"llama-{LLAMA_TAG}-bin-ubuntu-x64.tar.gz"
    raise ToolError(f"No prebuilt llama.cpp tools for {system} {machine}; pass --tools with a folder that has them")


def _download(url, destination):
    request = urllib.request.Request(url, headers={"User-Agent": "pocketfit"})
    with urllib.request.urlopen(request, timeout=60) as response, open(destination, "wb") as handle:
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            handle.write(chunk)


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inside(base, target):
    base = os.path.realpath(base)
    target = os.path.realpath(target)
    return target == base or target.startswith(base + os.sep)


def _extract(archive, destination):
    destination.mkdir(parents=True, exist_ok=True)
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as bundle:
            for name in bundle.namelist():
                if not _inside(destination, destination / name):
                    raise ToolError(f"Unsafe path in archive: {name}")
            bundle.extractall(destination)
    else:
        with tarfile.open(archive) as bundle:
            for member in bundle.getmembers():
                if not _inside(destination, destination / member.name) or member.issym() or member.islnk():
                    raise ToolError(f"Unsafe entry in archive: {member.name}")
            bundle.extractall(destination)


def _find(directory, name):
    for candidate in (name, name + ".exe"):
        for path in directory.rglob(candidate):
            if path.is_file():
                return path
    return None


def ensure_tools(folder=None):
    """Return (quantize_path, perplexity_path). Downloads a verified release on first use."""
    if folder:
        directory = Path(folder)
    else:
        directory = home() / "tools" / LLAMA_TAG
        if not (directory.exists() and _find(directory, "llama-quantize")):
            _install(directory)
    quantize = _find(directory, "llama-quantize")
    perplexity = _find(directory, "llama-perplexity")
    if not quantize or not perplexity:
        raise ToolError(f"llama-quantize and llama-perplexity not found in {directory}")
    for path in (quantize, perplexity):
        if os.name != "nt":
            path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return quantize, perplexity


def _install(directory):
    name = _asset_name()
    print(f"Downloading llama.cpp {LLAMA_TAG} tools ({name})...", flush=True)
    request = urllib.request.Request(RELEASE_API, headers={"User-Agent": "pocketfit"})
    with urllib.request.urlopen(request, timeout=30) as response:
        release = json.load(response)
    asset = next((a for a in release.get("assets", []) if a["name"] == name), None)
    if not asset:
        raise ToolError(f"{name} not found in release {LLAMA_TAG}")
    digest = (asset.get("digest") or "").removeprefix("sha256:")
    if not digest:
        raise ToolError("The release lists no checksum for the tools; refusing to run unverified programs")

    directory.parent.mkdir(parents=True, exist_ok=True)
    archive = directory.parent / name
    _download(asset["browser_download_url"], archive)
    if _sha256(archive) != digest:
        archive.unlink()
        raise ToolError("Checksum mismatch for the downloaded tools; refusing to use them")
    _extract(archive, directory)
    archive.unlink()


def ensure_text(path=None):
    """Return a plain text file for the quality check (WikiText-2 test split unless a path is given)."""
    if path:
        if not os.path.isfile(path):
            raise ToolError(f"Text file not found: {path}")
        return Path(path)
    target = home() / "data" / "wiki.test.raw"
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    print("Downloading the WikiText-2 test text for the quality check...", flush=True)
    archive = target.parent / "wikitext-2-raw-v1.zip"
    _download(TEXT_URL, archive)
    with zipfile.ZipFile(archive) as bundle:
        data = bundle.read(TEXT_MEMBER)
    archive.unlink()
    if not 1_000_000 < len(data) < 2_000_000:
        raise ToolError("Unexpected size for the downloaded text; refusing to use it")
    target.write_bytes(data)
    return target


def _git(*args, cwd=None):
    result = subprocess.run(["git", "-c", "advice.detachedHead=false", *args], cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        raise ToolError(f"git {args[0]} failed: {(result.stderr or result.stdout).strip()[-300:]}")
    return result.stdout.strip()


def ensure_converter(folder=None):
    """Return a llama.cpp source folder that has convert_hf_to_gguf.py, at the pinned commit.

    The commit is checked, so the script that runs is exactly the one that was reviewed for this version.
    """
    if shutil.which("git") is None:
        raise ToolError("git is required to fetch and verify the model converter")
    path = Path(folder) if folder else home() / "src" / f"llama.cpp-{LLAMA_TAG}"
    if not (path / "convert_hf_to_gguf.py").is_file():
        if folder:
            raise ToolError(f"convert_hf_to_gguf.py not found in {path}")
        print(f"Fetching the model converter (llama.cpp {LLAMA_TAG})...", flush=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        _git("clone", "--depth", "1", "--branch", LLAMA_TAG, "--quiet", LLAMA_REPO, str(path))
    head = _git("rev-parse", "HEAD", cwd=path)
    if head != LLAMA_COMMIT:
        raise ToolError(f"The converter in {path} is at commit {head[:10]}, expected {LLAMA_COMMIT[:10]}; refusing to run it")
    return path
