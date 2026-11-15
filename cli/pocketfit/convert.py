"""Turn a Hugging Face model folder into a full-precision GGUF file."""

import os
import subprocess
import sys
from pathlib import Path

from .tools import ToolError, ensure_converter

REQUIRED_MODULES = ["torch", "transformers", "sentencepiece", "numpy", "safetensors"]


def _check_python(python):
    probe = subprocess.run(
        [python, "-c", "import " + ", ".join(REQUIRED_MODULES)],
        capture_output=True, text=True, timeout=180,
    )
    if probe.returncode != 0:
        raise ToolError(
            f"The Python at {python} cannot import {', '.join(REQUIRED_MODULES)}. "
            "Install them (pip install torch transformers sentencepiece numpy safetensors) "
            "or pass --python with an interpreter that has them"
        )


def _check_model_dir(directory):
    if not (directory / "config.json").is_file():
        raise ToolError(f"{directory} has no config.json; it is not a Hugging Face model folder")
    if not list(directory.glob("*.safetensors")):
        # Pickle based checkpoints (.bin) can run arbitrary code when they are loaded.
        raise ToolError(
            f"{directory} has no .safetensors weights. Checkpoints in the older pickle format are refused "
            "because loading them can execute code; convert them to safetensors first"
        )


def _is_current(output, directory):
    if not output.is_file():
        return False
    newest = max((p.stat().st_mtime for p in directory.iterdir() if p.is_file()), default=0)
    return output.stat().st_mtime >= newest


def prepare_source(source, args):
    """Return a GGUF path for `source`: the file itself, or a conversion of a Hugging Face folder."""
    path = Path(source)
    if path.is_file():
        return path.resolve()
    if not path.is_dir():
        raise ToolError(f"Model not found: {path}")

    directory = path.resolve()
    _check_model_dir(directory)
    output = Path(args.out).resolve() / "source" / f"{directory.name}-f16.gguf"
    if _is_current(output, directory):
        print(f"Using the existing conversion {output}", file=sys.stderr, flush=True)
        return output

    python = args.python or sys.executable
    _check_python(python)
    llama = ensure_converter(args.llama_source)
    output.parent.mkdir(parents=True, exist_ok=True)
    print(f"Converting {directory.name} to GGUF (F16)...", file=sys.stderr, flush=True)

    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(llama / "gguf-py") + os.pathsep + environment.get("PYTHONPATH", "")
    environment["TRANSFORMERS_OFFLINE"] = "1"  # use only the files in the folder
    environment["HF_HUB_OFFLINE"] = "1"
    result = subprocess.run(
        [python, str(llama / "convert_hf_to_gguf.py"), str(directory), "--outtype", "f16", "--outfile", str(output)],
        capture_output=True, text=True, encoding="utf-8", errors="replace", env=environment,
    )
    if result.returncode != 0:
        output.unlink(missing_ok=True)
        raise ToolError(f"Conversion failed: {(result.stderr or result.stdout).strip()[-600:]}")
    return output
