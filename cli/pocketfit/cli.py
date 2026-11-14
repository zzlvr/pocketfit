"""Command line client for the PocketFit iPhone app.

The app runs a small HTTP server (port 8765) and advertises it over Bonjour.
Every request carries the pairing token shown in the app.
"""

import argparse
import http.client
import json
import os
import sys
import time
from urllib.parse import quote, urlencode

DEFAULT_PORT = 8765
SERVICE_TYPE = "_pocketfit._tcp.local."
DEFAULT_QUANTS = ["Q8_0", "Q6_K", "Q5_K_M", "Q5_0", "Q4_K_M", "Q4_0", "Q3_K_M", "Q2_K"]


class DeviceError(Exception):
    """The device could not be reached or rejected a request."""


class Device:
    def __init__(self, host, token, port=DEFAULT_PORT, timeout=30):
        self.host = host
        self.port = port
        self.token = token
        self.timeout = timeout

    def request(self, method, path, body=None, headers=None, timeout=None):
        connection = http.client.HTTPConnection(
            self.host, self.port, timeout=timeout or self.timeout, blocksize=1 << 20
        )
        merged = {"X-PocketFit-Token": self.token}
        merged.update(headers or {})
        try:
            connection.request(method, path, body=body, headers=merged)
            response = connection.getresponse()
            data = response.read()
        except (OSError, http.client.HTTPException) as error:
            raise DeviceError(f"Cannot reach {self.host}:{self.port}: {error}") from error
        finally:
            connection.close()

        try:
            payload = json.loads(data) if data else {}
        except ValueError:
            payload = {"raw": data.decode("utf-8", errors="replace")}
        if response.status >= 400:
            message = payload.get("error", payload) if isinstance(payload, dict) else payload
            raise DeviceError(f"HTTP {response.status}: {message}")
        return payload

    def info(self):
        return self.request("GET", "/info")

    def models(self):
        return self.request("GET", "/models")["models"]

    def upload(self, path):
        name = os.path.basename(path)
        size = os.path.getsize(path)
        with open(path, "rb") as handle:
            body = _ProgressReader(handle, size, name)
            result = self.request(
                "PUT",
                "/models/" + quote(name),
                body=body,
                headers={"Content-Length": str(size), "Content-Type": "application/octet-stream"},
                timeout=300,
            )
        print(file=sys.stderr)
        return result

    def delete(self, name):
        return self.request("DELETE", "/models/" + quote(name))

    def bench(self, name, gpu, runs, pause, tokens):
        query = urlencode({
            "model": name,
            "gpu": 1 if gpu else 0,
            "runs": runs,
            "pause": pause,
            "tokens": tokens,
        })
        timeout = max(600, runs * (pause + 180))
        return self.request("POST", "/bench?" + query, timeout=timeout)


class _ProgressReader:
    """File wrapper that prints upload progress to stderr."""

    def __init__(self, handle, total, name):
        self.handle = handle
        self.total = total
        self.name = name
        self.sent = 0
        self.started = time.time()
        self.last_print = 0.0

    def read(self, size=-1):
        chunk = self.handle.read(size)
        self.sent += len(chunk)
        now = time.time()
        if chunk and now - self.last_print < 0.5 and self.sent < self.total:
            return chunk
        self.last_print = now
        elapsed = max(now - self.started, 0.001)
        percent = 100.0 * self.sent / self.total if self.total else 100.0
        speed = self.sent / elapsed / 1_048_576
        print(
            f"\rUploading {self.name}: {percent:5.1f}%  {self.sent / 1_048_576:.0f}/{self.total / 1_048_576:.0f} MB  {speed:.1f} MB/s",
            end="",
            file=sys.stderr,
        )
        return chunk


def discover(timeout=4.0):
    """Find PocketFit apps on the local network via Bonjour. Returns a list of (name, host, port)."""
    try:
        from zeroconf import ServiceBrowser, ServiceListener, Zeroconf
    except ImportError as error:
        raise DeviceError(
            "Bonjour discovery needs the optional package: pip install zeroconf (or pass --host)"
        ) from error

    found = {}

    class Listener(ServiceListener):
        def add_service(self, zc, type_, name):
            info = zc.get_service_info(type_, name, timeout=2000)
            if info and info.parsed_addresses():
                found[name] = (name.split(".")[0], info.parsed_addresses()[0], info.port)

        def update_service(self, zc, type_, name):
            pass

        def remove_service(self, zc, type_, name):
            found.pop(name, None)

    zeroconf = Zeroconf()
    try:
        ServiceBrowser(zeroconf, SERVICE_TYPE, Listener())
        time.sleep(timeout)
    finally:
        zeroconf.close()
    return list(found.values())


def _device_from_args(args):
    token = args.token or os.environ.get("POCKETFIT_TOKEN")
    if not token:
        raise DeviceError("A pairing token is required: pass --token or set POCKETFIT_TOKEN")
    host = args.host or os.environ.get("POCKETFIT_HOST")
    port = args.port
    if host and host.count(":") == 1:
        # Accept "address:port" as well as a plain address.
        host, _, port_text = host.partition(":")
        if not port_text.isdigit():
            raise DeviceError(f"Invalid port in --host: {port_text!r}")
        port = int(port_text)
    if not host:
        devices = discover()
        if not devices:
            raise DeviceError("No device found. Turn on 'Accept connections' in the app or pass --host")
        if len(devices) > 1:
            listing = ", ".join(f"{host}:{port}" for _, host, port in devices)
            raise DeviceError(f"Several devices answered ({listing}). Pass --host to choose one")
        name, host, port = devices[0]
        print(f"Using {name} at {host}:{port}", file=sys.stderr)
    return Device(host, token, port)


def _print_json(value):
    print(json.dumps(value, indent=2, sort_keys=True))


def cmd_discover(args):
    devices = discover()
    if not devices:
        print("No devices found", file=sys.stderr)
        return 1
    for name, host, port in devices:
        print(f"{name}\t{host}:{port}")
    return 0


def cmd_info(args):
    _print_json(_device_from_args(args).info())
    return 0


def cmd_models(args):
    _print_json(_device_from_args(args).models())
    return 0


def cmd_delete(args):
    _print_json(_device_from_args(args).delete(args.name))
    return 0


def cmd_bench(args):
    device = _device_from_args(args)
    path = os.path.abspath(args.model)
    if not os.path.isfile(path):
        raise DeviceError(f"Model file not found: {path}")
    name = os.path.basename(path)
    size = os.path.getsize(path)

    on_device = next((m for m in device.models() if m["name"] == name), None)
    if on_device and on_device["size_bytes"] == size and not args.force_upload:
        print(f"{name} is already on the device, skipping upload", file=sys.stderr)
    else:
        device.upload(path)

    print(f"Benchmarking {name} ({args.runs} runs, {'GPU' if args.gpu else 'CPU'})...", file=sys.stderr)
    result = device.bench(name, args.gpu, args.runs, args.pause, args.tokens)
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
        summary = result.get("summary", {})
        print(
            f"Mean {summary.get('mean_tokens_per_second', 0):.1f} tokens/s "
            f"(min {summary.get('min_tokens_per_second', 0):.1f}, max {summary.get('max_tokens_per_second', 0):.1f}). "
            f"Saved to {args.out}",
            file=sys.stderr,
        )
    else:
        print(text)
    return 0


def cmd_tune(args):
    from .tools import ToolError
    from .tune import tune

    device = _device_from_args(args)
    try:
        report, best, out_dir = tune(device, args)
    except ToolError as error:
        raise DeviceError(str(error)) from error
    print((out_dir / "report.md").read_text(encoding="utf-8"))
    if best is None:
        print("error: no candidate met the constraints; see the table above", file=sys.stderr)
        return 2
    print(f"Best: {best['quantization']} -> {out_dir / 'models' / best['file']}")
    return 0


def _bounded(low, high):
    """argparse type that accepts integers from low to high."""
    def parse(text):
        try:
            value = int(text)
        except ValueError:
            raise argparse.ArgumentTypeError(f"{text!r} is not a whole number")
        if not low <= value <= high:
            raise argparse.ArgumentTypeError(f"must be between {low} and {high}")
        return value
    return parse


def build_parser():
    parser = argparse.ArgumentParser(prog="pocketfit", description="Benchmark GGUF models on a real iPhone.")
    parser.add_argument("--host", help="device address (default: discover via Bonjour)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--token", help="pairing token shown in the app (or POCKETFIT_TOKEN)")
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("discover", help="list devices found over Bonjour").set_defaults(func=cmd_discover)
    commands.add_parser("info", help="show device information").set_defaults(func=cmd_info)
    commands.add_parser("models", help="list models stored on the device").set_defaults(func=cmd_models)

    delete = commands.add_parser("delete", help="delete a model from the device")
    delete.add_argument("name")
    delete.set_defaults(func=cmd_delete)

    bench = commands.add_parser("bench", help="upload a model if needed and benchmark it")
    bench.add_argument("model", help="path to a .gguf file")
    bench.add_argument("--gpu", action="store_true", help="use the GPU (Metal); default is CPU only")
    bench.add_argument("--runs", type=_bounded(1, 20), default=5, help="runs, 1 to 20")
    bench.add_argument("--pause", type=_bounded(0, 120), default=10, help="seconds between runs, 0 to 120")
    bench.add_argument("--tokens", type=_bounded(1, 512), default=64, help="tokens to generate per run, 1 to 512")
    bench.add_argument("--force-upload", action="store_true", help="upload even if a file of the same size exists")
    bench.add_argument("--out", help="write the JSON result to this file")
    bench.set_defaults(func=cmd_bench)

    tune = commands.add_parser("tune", help="try several quantizations of a model on the device and pick the best")
    tune.add_argument("source", help="path to the full-precision .gguf (F16 or F32)")
    tune.add_argument("--min-tps", type=float, default=10.0, help="minimum tokens/s the choice must reach (default 10)")
    tune.add_argument("--max-loss", type=float, default=None, help="maximum quality loss in percent (perplexity increase)")
    tune.add_argument("--quants", default=",".join(DEFAULT_QUANTS), help="comma separated list, e.g. Q8_0,Q4_K_M")
    tune.add_argument("--out", default="pocketfit-out", help="folder for the quantized files and the report")
    tune.add_argument("--runs", type=_bounded(1, 19), default=5, help="timed runs per candidate (one more warm-up run is added)")
    tune.add_argument("--pause", type=_bounded(0, 120), default=10, help="seconds between runs")
    tune.add_argument("--tokens", type=_bounded(1, 512), default=64, help="tokens generated per run")
    tune.add_argument("--settle", type=_bounded(0, 600), default=30, help="seconds to wait after an upload before timing")
    tune.add_argument("--ppl-chunks", type=_bounded(1, 580), default=20, help="text chunks used for the quality check")
    tune.add_argument("--gpu", action="store_true", help="measure on the GPU (Metal); default is CPU only")
    tune.add_argument("--tools", help="folder with llama-quantize and llama-perplexity (default: download a verified copy)")
    tune.add_argument("--text", help="plain text file for the quality check (default: WikiText-2 test split)")
    tune.set_defaults(func=cmd_tune)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except DeviceError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
