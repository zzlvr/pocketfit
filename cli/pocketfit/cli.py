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

    def read(self, size=-1):
        chunk = self.handle.read(size)
        self.sent += len(chunk)
        elapsed = max(time.time() - self.started, 0.001)
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
    bench.add_argument("--runs", type=int, default=5)
    bench.add_argument("--pause", type=int, default=10, help="seconds between runs")
    bench.add_argument("--tokens", type=int, default=64, help="tokens to generate per run")
    bench.add_argument("--force-upload", action="store_true", help="upload even if a file of the same size exists")
    bench.add_argument("--out", help="write the JSON result to this file")
    bench.set_defaults(func=cmd_bench)
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
