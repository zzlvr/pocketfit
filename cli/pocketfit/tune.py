"""Pick the best quantization of a model by measuring each candidate on the real device."""

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from .tools import ToolError, ensure_text, ensure_tools

DEFAULT_QUANTS = ["Q8_0", "Q6_K", "Q5_K_M", "Q5_0", "Q4_K_M", "Q4_0", "Q3_K_M", "Q2_K"]
PPL_PATTERN = re.compile(r"Final estimate: PPL = ([0-9.]+) \+/- ([0-9.]+)")
# A model should use at most this share of the phone's RAM; a rule of thumb, not a measured limit.
RAM_SHARE = 0.45
DISK_MARGIN_MB = 200
# A run set whose fastest and slowest runs differ by more than this share of the median is flagged.
UNSTABLE_SPREAD = 0.25


def log(message):
    print(message, file=sys.stderr, flush=True)


def _run(command):
    result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
    output = (result.stdout or "") + (result.stderr or "")
    if result.returncode != 0:
        raise ToolError(f"{Path(command[0]).name} failed ({result.returncode}): {output[-400:]}")
    return output


def quantize(quantize_tool, source, target, quant):
    _run([str(quantize_tool), str(source), str(target), quant])


def perplexity(perplexity_tool, model, text, chunks):
    output = _run([str(perplexity_tool), "-m", str(model), "-f", str(text), "-c", "512", "--chunks", str(chunks)])
    match = PPL_PATTERN.search(output)
    if not match:
        raise ToolError("Could not read the perplexity result")
    return float(match.group(1)), float(match.group(2))


def choose(results, min_tps, max_loss):
    """Highest quality (lowest loss) among candidates that meet the speed and quality limits."""
    fit = [
        r for r in results
        if r["status"] == "ok"
        and (min_tps is None or r["tokens_per_second_median"] >= min_tps)
        and (max_loss is None or r["loss_percent"] <= max_loss)
    ]
    if not fit:
        return None
    return min(fit, key=lambda r: (r["loss_percent"], -r["tokens_per_second_median"]))


def tune(device, args):
    source = Path(args.source).resolve()
    if not source.is_file():
        raise ToolError(f"Source model not found: {source}")
    out_dir = Path(args.out).resolve()
    models_dir = out_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    quantize_tool, perplexity_tool = ensure_tools(args.tools)
    text = ensure_text(args.text)
    quants = [q.strip() for q in args.quants.split(",") if q.strip()]
    stem = re.sub(r"[-_.](f16|fp16|f32|bf16)$", "", source.stem, flags=re.IGNORECASE)

    info = device.info()
    log(f"Device: {info['identifier']}, iOS {info['ios_version']}, {info['physical_memory_mb']} MB RAM, {info['free_disk_mb']} MB free")
    if info.get("charging"):
        log("Warning: the phone is charging; charging heats it and can lower measured speed")
    ram_limit_mb = RAM_SHARE * info["physical_memory_mb"]

    log("Measuring the source model for the quality baseline...")
    base_ppl, base_err = perplexity(perplexity_tool, source, text, args.ppl_chunks)
    log(f"  source perplexity {base_ppl:.3f} +/- {base_err:.3f}")

    results = []
    for quant in quants:
        name = f"{stem}-{quant}.gguf"
        path = models_dir / name
        record = {"quantization": quant, "file": name, "status": "ok"}
        log(f"\n[{quant}]")
        try:
            if not path.exists():
                quantize(quantize_tool, source, path, quant)
            record["size_mb"] = round(path.stat().st_size / 1_048_576, 1)

            free_mb = device.info()["free_disk_mb"]
            if record["size_mb"] > ram_limit_mb:
                record.update(status="skipped", reason=f"larger than {RAM_SHARE:.0%} of the phone's RAM")
                log(f"  skipped: {record['reason']}")
                results.append(record)
                continue
            if record["size_mb"] + DISK_MARGIN_MB > free_mb:
                record.update(status="skipped", reason="not enough free storage on the phone")
                log(f"  skipped: {record['reason']}")
                results.append(record)
                continue

            ppl, err = perplexity(perplexity_tool, path, text, args.ppl_chunks)
            record.update(perplexity=ppl, perplexity_error=err, loss_percent=round((ppl / base_ppl - 1) * 100, 2))
            log(f"  size {record['size_mb']} MB, perplexity {ppl:.3f}, loss {record['loss_percent']:+.2f}%")

            device.upload(str(path))
            try:
                log(f"  letting the phone settle for {args.settle} s after the upload...")
                time.sleep(args.settle)
                # One extra run is a warm-up and is not counted.
                bench = device.bench(name, args.gpu, args.runs + 1, args.pause, args.tokens)
            finally:
                try:
                    device.delete(name)
                except Exception as error:  # the measurement matters more than cleanup
                    log(f"  could not delete {name} from the phone: {error}")
            runs = bench["runs"][1:]
            speeds = [r["tokens_per_second"] for r in runs]
            median = sorted(speeds)[len(speeds) // 2] if len(speeds) % 2 else sum(sorted(speeds)[len(speeds) // 2 - 1:len(speeds) // 2 + 1]) / 2
            record.update(
                tokens_per_second_median=round(median, 2),
                tokens_per_second_runs=[round(s, 2) for s in speeds],
                unstable=(max(speeds) - min(speeds)) > UNSTABLE_SPREAD * median,
                tokens_per_second_mean=round(sum(speeds) / len(speeds), 2),
                tokens_per_second_min=round(min(speeds), 2),
                tokens_per_second_max=round(max(speeds), 2),
                thermal_state_end=runs[-1]["thermal_state_end"],
                charging=any(r["charging"] for r in runs),
            )
            log(f"  speed {record['tokens_per_second_median']} tokens/s median (min {record['tokens_per_second_min']}, max {record['tokens_per_second_max']})" + ("  [unstable]" if record["unstable"] else ""))
        except Exception as error:
            record.update(status="failed", reason=str(error))
            log(f"  failed: {error}")
        results.append(record)

    best = choose(results, args.min_tps, args.max_loss)
    report = {
        "source": source.name,
        "device": info,
        "backend": "GPU" if args.gpu else "CPU",
        "quality_metric": {
            "name": "perplexity on WikiText-2 test text",
            "context": 512,
            "chunks": args.ppl_chunks,
            "source_perplexity": base_ppl,
            "source_perplexity_error": base_err,
        },
        "constraints": {"min_tokens_per_second": args.min_tps, "max_loss_percent": args.max_loss},
        "runs_per_candidate": args.runs,
        "candidates": results,
        "selected": best["quantization"] if best else None,
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out_dir / "report.md").write_text(render_markdown(report), encoding="utf-8")
    log(f"\nReport written to {out_dir}")
    return report, best, out_dir


def render_markdown(report):
    lines = [
        f"# Quantization report: {report['source']}",
        "",
        f"Device: {report['device']['identifier']}, iOS {report['device']['ios_version']}, "
        f"{report['device']['physical_memory_mb']} MB RAM. Backend: {report['backend']}.",
        "",
        "| Quantization | Size (MB) | Perplexity | Loss | Tokens/s (median, min-max) | Status |",
        "|---|---|---|---|---|---|",
    ]
    for r in report["candidates"]:
        if r["status"] != "ok":
            lines.append(f"| {r['quantization']} | {r.get('size_mb', '')} | | | | {r['status']}: {r.get('reason', '')} |")
            continue
        mark = " (selected)" if r["quantization"] == report["selected"] else ""
        lines.append(
            f"| {r['quantization']}{mark} | {r['size_mb']} | {r['perplexity']:.2f} | {r['loss_percent']:+.2f}% | "
            f"{r['tokens_per_second_median']} ({r['tokens_per_second_min']}-{r['tokens_per_second_max']}) | "
            f"{'ok, unstable timing' if r.get('unstable') else 'ok'} |"
        )
    c = report["constraints"]
    lines += [
        "",
        f"Constraints: min speed {c['min_tokens_per_second']} tokens/s, max loss {c['max_loss_percent']}%.",
        f"Selected: {report['selected'] or 'none (no candidate met the constraints)'}.",
        "",
        "Loss is the relative increase in perplexity over the source model on the WikiText-2 test text "
        f"({report['quality_metric']['chunks']} chunks of {report['quality_metric']['context']} tokens). "
        "It measures how closely the quantized model follows the source on general English text, "
        "not accuracy on a specific task.",
        f"Each candidate ran {report['runs_per_candidate']} timed runs after one warm-up run. "
        "Speed is the median of the timed runs; a candidate is marked unstable when its fastest and slowest runs "
        f"differ by more than {int(UNSTABLE_SPREAD * 100)}% of the median.",
        f"The standard error of the source perplexity is {report['quality_metric']['source_perplexity_error']:.2f} "
        f"({report['quality_metric']['source_perplexity_error'] / report['quality_metric']['source_perplexity'] * 100:.1f}%); "
        "loss differences smaller than that are not meaningful.",
        "",
    ]
    return "\n".join(lines)
