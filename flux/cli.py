from __future__ import annotations

import argparse
import json
import mimetypes
import runpy
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple
from urllib.parse import unquote, urlparse

from .aggregator import aggregate_records
from .analyzer import analyze_records
from .gpu_analyzer import analyze_gpu_records
from .profiler import FluxProfiler
from .trace_exporter import export_trace


def _read_trace_payload(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_records_from_payload(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    events = payload.get("traceEvents", [])
    records: List[Dict[str, Any]] = []
    for event in events:
        if event.get("ph") != "X":
            continue
        args = event.get("args") or {}
        records.append(
            {
                "op_name": event.get("name", args.get("op_name", "unknown")),
                "start_us": int(event.get("ts", 0)),
                "duration_us": int(event.get("dur", 0)),
                "end_us": int(event.get("ts", 0)) + int(event.get("dur", 0)),
                "thread_id": int(event.get("tid", 0)),
                "classification": args.get("classification"),
                "is_cuda": bool(args.get("is_cuda", False)),
                "device_id": int(args.get("device_id", -1)),
                "stream_id": int(args.get("stream_id", -1)),
                "cuda_elapsed_us": float(args.get("cuda_elapsed_us", -1.0)),
            }
        )
    records.sort(key=lambda x: (x["start_us"], x["end_us"]))
    return records


def _extract_records_from_trace(path: Path) -> List[Dict[str, Any]]:
    return _extract_records_from_payload(_read_trace_payload(path))


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _op_means(
    records: Iterable[Dict[str, Any]],
    metric_key: str = "duration_us",
    use_cuda_fallback: bool = False,
    only_cuda: bool = False,
) -> Dict[str, float]:
    buckets: Dict[str, List[float]] = {}
    for item in records:
        if only_cuda and not bool(item.get("is_cuda", False)):
            continue
        op_name = str(item.get("op_name", "unknown"))
        metric = _as_float(item.get(metric_key), -1.0)
        if metric <= 0 and use_cuda_fallback:
            metric = _as_float(item.get("duration_us"), 0.0)
        if metric <= 0:
            continue
        buckets.setdefault(op_name, []).append(metric)
    return {name: (sum(values) / len(values)) for name, values in buckets.items() if values}


def _format_us(value: float) -> str:
    return f"{value:.2f} us"


def _print_aggregate(aggregate: Dict[str, Any]) -> None:
    print(f"Total ops: {aggregate['total_ops']}")
    print(f"Total op duration: {_format_us(float(aggregate['total_duration_us']))}")
    print(f"Wall time: {_format_us(float(aggregate['wall_time_us']))}")
    print(f"Idle time: {_format_us(float(aggregate['idle_time_us']))}")
    print(f"Utilization: {aggregate['utilization_pct']:.2f}%")
    print("")
    print("Top ops by total duration:")
    for row in aggregate["ops"][:10]:
        print(
            f"  {row['op_name']}: count={row['count']} "
            f"mean={_format_us(float(row['mean_us']))} "
            f"total={_format_us(float(row['total_us']))} "
            f"share={row['pct_total_duration']:.2f}%"
        )


def _print_gpu_diagnostics(gpu_summary: Dict[str, Any]) -> None:
    print("")
    print("GPU diagnostics:")
    if not gpu_summary.get("available", False):
        print("  GPU data: not available in this trace.")
        return

    print(f"  CUDA ops: {int(gpu_summary.get('cuda_ops', 0))}")
    print(f"  CUDA time: {_format_us(float(gpu_summary.get('cuda_time_us', 0.0)))}")
    print(f"  GPU activity estimate: {float(gpu_summary.get('gpu_activity_pct', 0.0)):.2f}%")
    print(
        "  SM utilization estimate: "
        f"{float(gpu_summary.get('sm_utilization_estimate_pct', 0.0)):.2f}%"
    )
    print(
        "  Memory bandwidth pressure estimate: "
        f"{float(gpu_summary.get('memory_bandwidth_pressure_estimate_pct', 0.0)):.2f}%"
    )
    print(
        "  Host-to-device transfer pressure estimate: "
        f"{float(gpu_summary.get('h2d_transfer_pressure_estimate_pct', 0.0)):.2f}%"
    )

    memory = gpu_summary.get("memory") or {}
    if memory.get("cuda_available", False):
        totals = memory.get("totals") or {}
        print(
            f"  Memory allocated delta: {int(totals.get('allocated_delta_bytes', 0))} bytes"
        )
        print(
            f"  Memory reserved delta: {int(totals.get('reserved_delta_bytes', 0))} bytes"
        )
        print(f"  Peak allocated: {int(totals.get('peak_allocated_bytes', 0))} bytes")
        print(f"  Peak reserved: {int(totals.get('peak_reserved_bytes', 0))} bytes")


def _run_script_under_profiler(
    script_path: Path, script_args: List[str], timing_mode: str
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")

    # argparse.REMAINDER may keep a leading "--" separator.
    if script_args and script_args[0] == "--":
        script_args = script_args[1:]

    old_argv = list(sys.argv)
    sys.argv = [str(script_path)] + script_args
    try:
        with FluxProfiler(timing_mode=timing_mode) as profiler:
            runpy.run_path(str(script_path), run_name="__main__")
        return profiler.records, profiler.gpu_memory_telemetry
    finally:
        sys.argv = old_argv


def _cmd_profile(args: argparse.Namespace) -> int:
    records, gpu_memory_telemetry = _run_script_under_profiler(
        Path(args.script), args.script_args, args.timing_mode
    )
    aggregate = aggregate_records(records)
    analyzed = analyze_records(
        records,
        peak_flops_tflops=args.peak_flops_tflops,
        memory_bandwidth_gbps=args.memory_bandwidth_gbps,
    )
    gpu_summary = analyze_gpu_records(
        analyzed["records"],
        wall_time_us=float(aggregate.get("wall_time_us", 0.0)),
        memory_telemetry=gpu_memory_telemetry,
    )
    payload = export_trace(
        analyzed["records"],
        args.output,
        summary={**aggregate, "classification": analyzed["summary"], "gpu": gpu_summary},
        metadata={"timing_mode": args.timing_mode},
    )

    print(
        f"Wrote trace with {len(payload['traceEvents'])} events: {args.output} "
        f"(timing_mode={args.timing_mode})"
    )
    if args.timing_mode == "cuda":
        cuda_records = [item for item in records if bool(item.get("is_cuda", False))]
        positive_cuda_elapsed = sum(
            1 for item in cuda_records if float(item.get("cuda_elapsed_us", -1.0)) > 0.0
        )
        if cuda_records and positive_cuda_elapsed == 0:
            print(
                "Warning: CUDA timing mode is enabled, but no positive cuda_elapsed_us values were captured."
            )
    _print_aggregate(aggregate)
    _print_gpu_diagnostics(gpu_summary)
    return 0


def _regression_report(
    current_means: Dict[str, float],
    baseline_means: Dict[str, float],
    threshold_pct: float,
    min_baseline_us: float = 0.0,
    min_regression_delta_us: float = 0.0,
) -> Tuple[List[Tuple[str, float, float, float, float]], List[str], List[str]]:
    regressions: List[Tuple[str, float, float, float, float]] = []
    missing_in_baseline: List[str] = []
    skipped_small_baseline: List[str] = []

    for op_name, current in current_means.items():
        baseline = baseline_means.get(op_name)
        if baseline is None:
            missing_in_baseline.append(op_name)
            continue
        if baseline <= 0:
            continue

        if baseline < min_baseline_us:
            skipped_small_baseline.append(op_name)
            continue

        delta_us = current - baseline
        delta_pct = ((current - baseline) / baseline) * 100.0
        if delta_pct > threshold_pct and delta_us > min_regression_delta_us:
            regressions.append((op_name, baseline, current, delta_pct, delta_us))

    regressions.sort(key=lambda x: x[3], reverse=True)
    return regressions, sorted(missing_in_baseline), sorted(skipped_small_baseline)


def _print_regression_block(
    *,
    label: str,
    baseline_path: str,
    threshold: float,
    min_baseline_us: float,
    min_regression_delta_us: float,
    regressions: List[Tuple[str, float, float, float, float]],
    missing: List[str],
    skipped: List[str],
    value_unit: str = "us",
) -> None:
    print("")
    print(
        f"{label} regression check vs baseline ({baseline_path}) with threshold {threshold:.2f}%:"
    )
    print(
        f"  Filters: min_baseline_us={min_baseline_us:.2f}, "
        f"min_regression_delta_us={min_regression_delta_us:.2f}"
    )
    if missing:
        print(f"  Ops missing in baseline: {', '.join(missing[:10])}")
    if skipped:
        print(f"  Ops skipped by min_baseline_us: {', '.join(skipped[:10])}")
    if not regressions:
        print("  No regressions detected.")
        return

    print("  Regressions:")
    for op_name, baseline, current, delta_pct, delta_us in regressions:
        print(
            f"    {op_name}: baseline={_format_us(baseline)} "
            f"current={_format_us(current)} delta={delta_pct:.2f}% "
            f"({delta_us:.2f} {value_unit})"
        )


def _cmd_analyze(args: argparse.Namespace) -> int:
    trace_path = Path(args.trace)
    payload = _read_trace_payload(trace_path)
    records = _extract_records_from_payload(payload)
    aggregate = aggregate_records(records)
    _print_aggregate(aggregate)
    gpu_summary = analyze_gpu_records(
        records,
        wall_time_us=float(aggregate.get("wall_time_us", 0.0)),
        memory_telemetry=(payload.get("summary", {}) or {}).get("gpu", {}).get("memory"),
    )
    _print_gpu_diagnostics(gpu_summary)

    if not args.baseline:
        return 0

    baseline_payload = _read_trace_payload(Path(args.baseline))
    baseline_records = _extract_records_from_payload(baseline_payload)
    current_means = _op_means(records)
    baseline_means = _op_means(baseline_records)
    regressions, missing, skipped = _regression_report(
        current_means,
        baseline_means,
        args.threshold,
        min_baseline_us=args.min_baseline_us,
        min_regression_delta_us=args.min_regression_delta_us,
    )
    _print_regression_block(
        label="CPU",
        baseline_path=args.baseline,
        threshold=args.threshold,
        min_baseline_us=args.min_baseline_us,
        min_regression_delta_us=args.min_regression_delta_us,
        regressions=regressions,
        missing=missing,
        skipped=skipped,
    )
    has_cpu_regression = bool(regressions)

    has_gpu_regression = False
    if args.gpu_ci_mode != "off":
        current_gpu_means = _op_means(
            records,
            metric_key="cuda_elapsed_us",
            only_cuda=True,
        )
        baseline_gpu_means = _op_means(
            baseline_records,
            metric_key="cuda_elapsed_us",
            only_cuda=True,
        )

        if not current_gpu_means or not baseline_gpu_means:
            print("")
            if not current_gpu_means and not baseline_gpu_means:
                print(
                    "GPU regression check skipped: no CUDA elapsed timing data in current or baseline trace."
                )
            elif not current_gpu_means:
                print(
                    "GPU regression check skipped: no CUDA elapsed timing data in current trace."
                )
            else:
                print(
                    "GPU regression check skipped: no CUDA elapsed timing data in baseline trace."
                )

            if args.gpu_ci_mode == "require":
                print("GPU regression mode is require, so this is a failure.")
                has_gpu_regression = True
        else:
            gpu_regressions, gpu_missing, gpu_skipped = _regression_report(
                current_gpu_means,
                baseline_gpu_means,
                args.gpu_threshold,
                min_baseline_us=args.gpu_min_baseline_us,
                min_regression_delta_us=args.gpu_min_regression_delta_us,
            )
            _print_regression_block(
                label="GPU",
                baseline_path=args.baseline,
                threshold=args.gpu_threshold,
                min_baseline_us=args.gpu_min_baseline_us,
                min_regression_delta_us=args.gpu_min_regression_delta_us,
                regressions=gpu_regressions,
                missing=gpu_missing,
                skipped=gpu_skipped,
            )
            has_gpu_regression = bool(gpu_regressions)

    return 1 if (has_cpu_regression or has_gpu_regression) else 0


def _dashboard_root_dir() -> Path:
    # Prefer the current repo checkout when running from project root.
    cwd_dashboard = Path.cwd() / "dashboard"
    if (cwd_dashboard / "src").exists() and (cwd_dashboard / "package.json").exists():
        return cwd_dashboard

    # Fallback to the dashboard shipped next to the installed Flux package.
    project_root = Path(__file__).resolve().parent.parent
    return project_root / "dashboard"


def _dashboard_dist_dir() -> Path:
    return _dashboard_root_dir() / "dist"


def _latest_mtime_in_dir(path: Path) -> float:
    if not path.exists():
        return 0.0
    latest = 0.0
    for file_path in path.rglob("*"):
        if file_path.is_file():
            latest = max(latest, file_path.stat().st_mtime)
    return latest


def _dashboard_needs_build(dashboard_root: Path) -> bool:
    dist_index = dashboard_root / "dist" / "index.html"
    if not dist_index.exists():
        return True

    latest_source = 0.0
    for file_path in (
        dashboard_root / "index.html",
        dashboard_root / "package.json",
        dashboard_root / "package-lock.json",
        dashboard_root / "vite.config.js",
    ):
        if file_path.exists():
            latest_source = max(latest_source, file_path.stat().st_mtime)

    latest_source = max(latest_source, _latest_mtime_in_dir(dashboard_root / "src"))
    latest_source = max(latest_source, _latest_mtime_in_dir(dashboard_root / "public"))
    return latest_source > dist_index.stat().st_mtime


def _build_dashboard(dashboard_root: Path) -> bool:
    if not dashboard_root.exists():
        return False
    result = subprocess.run(
        ["npm", "run", "build"],
        cwd=dashboard_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        return True
    print("Dashboard build failed. Serving fallback/previous build.")
    if result.stdout.strip():
        print(result.stdout.strip())
    if result.stderr.strip():
        print(result.stderr.strip())
    return False


def _fallback_index_html(trace_name: str, dashboard_dist: Path) -> bytes:
    body = f"""<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <title>Flux Trace Server</title>
    <style>
      body {{
        font-family: sans-serif;
        margin: 2rem;
        line-height: 1.4;
      }}
      code {{
        background: #f5f5f5;
        padding: 0.15rem 0.35rem;
        border-radius: 0.3rem;
      }}
    </style>
  </head>
  <body>
    <h1>Flux Trace Server</h1>
    <p>Trace file: <code>{trace_name}</code></p>
    <p>Direct JSON URL: <a href="/trace.json">/trace.json</a></p>
    <p>Dashboard build not found at <code>{dashboard_dist}</code>.</p>
    <p>Build it with:</p>
    <pre><code>cd dashboard
npm install
npm run build</code></pre>
    <p>Then run <code>flux serve --trace ...</code> again.</p>
  </body>
</html>
"""
    return body.encode("utf-8")


def _safe_static_path(dist_dir: Path, request_path: str) -> Path | None:
    normalized = urlparse(request_path).path
    relative = unquote(normalized.lstrip("/"))
    if not relative or relative == ".":
        relative = "index.html"
    candidate = (dist_dir / relative).resolve()
    try:
        candidate.relative_to(dist_dir.resolve())
    except ValueError:
        return None
    return candidate


def _build_trace_server(trace_path: Path, port: int) -> HTTPServer:
    trace_bytes = trace_path.read_bytes()
    trace_name = trace_path.name
    dist_dir = _dashboard_dist_dir()
    dist_exists = dist_dir.exists()

    class TraceHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/trace.json":
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
                self.send_header("Pragma", "no-cache")
                self.send_header("Expires", "0")
                self.send_header("Content-Length", str(len(trace_bytes)))
                self.end_headers()
                self.wfile.write(trace_bytes)
                return

            if dist_exists:
                requested = _safe_static_path(dist_dir, self.path)
                if requested is None:
                    self.send_response(400)
                    self.end_headers()
                    return

                if requested.exists() and requested.is_file():
                    content_type, _ = mimetypes.guess_type(str(requested))
                    data = requested.read_bytes()
                    self.send_response(200)
                    self.send_header(
                        "Content-Type", (content_type or "application/octet-stream")
                    )
                    self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
                    self.send_header("Pragma", "no-cache")
                    self.send_header("Expires", "0")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                    return

                # SPA fallback for client-side routes.
                index_file = dist_dir / "index.html"
                if index_file.exists():
                    data = index_file.read_bytes()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
                    self.send_header("Pragma", "no-cache")
                    self.send_header("Expires", "0")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                    return

            if self.path in ("/", "/index.html"):
                body = _fallback_index_html(trace_name, dist_dir)
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
                self.send_header("Pragma", "no-cache")
                self.send_header("Expires", "0")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            self.send_response(404)
            self.end_headers()

        def log_message(self, fmt: str, *args: Any) -> None:
            return

    return HTTPServer(("127.0.0.1", port), TraceHandler)


def _cmd_serve(args: argparse.Namespace) -> int:
    trace_path = Path(args.trace)
    if not trace_path.exists():
        raise FileNotFoundError(f"Trace file not found: {trace_path}")

    dashboard_root = _dashboard_root_dir()
    if args.dashboard_build == "always":
        print(f"Building dashboard (mode=always) from {dashboard_root} ...")
        _build_dashboard(dashboard_root)
    elif args.dashboard_build == "auto":
        if _dashboard_needs_build(dashboard_root):
            print(f"Building dashboard (mode=auto) from {dashboard_root} ...")
            _build_dashboard(dashboard_root)

    server = _build_trace_server(trace_path, args.port)
    dashboard_dist = _dashboard_dist_dir()
    if dashboard_dist.exists():
        print(f"Serving dashboard from {dashboard_dist} with trace {trace_path}")
    else:
        print(f"Serving trace only (dashboard build missing at {dashboard_dist})")
    print(f"URL: http://127.0.0.1:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="flux", description="Flux ML profiler CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    profile = subparsers.add_parser("profile", help="Run a Python script under the profiler")
    profile.add_argument("--script", required=True, help="Path to Python script to execute")
    profile.add_argument("--output", required=True, help="Trace JSON output path")
    profile.add_argument(
        "--peak-flops-tflops",
        type=float,
        default=10.0,
        help="Peak hardware throughput used by analyzer heuristics",
    )
    profile.add_argument(
        "--memory-bandwidth-gbps",
        type=float,
        default=300.0,
        help="Peak memory bandwidth used by analyzer heuristics",
    )
    profile.add_argument(
        "--timing-mode",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Timing mode for profiling (auto, cpu, cuda)",
    )
    profile.add_argument("script_args", nargs=argparse.REMAINDER)
    profile.set_defaults(func=_cmd_profile)

    analyze = subparsers.add_parser("analyze", help="Analyze a trace JSON file")
    analyze.add_argument("--trace", required=True, help="Trace JSON path")
    analyze.add_argument("--baseline", help="Baseline trace JSON path")
    analyze.add_argument(
        "--threshold",
        type=float,
        default=5.0,
        help="Regression threshold percentage for baseline comparison",
    )
    analyze.add_argument(
        "--min-baseline-us",
        type=float,
        default=0.0,
        help="Ignore ops with baseline mean below this value (microseconds)",
    )
    analyze.add_argument(
        "--min-regression-delta-us",
        type=float,
        default=0.0,
        help="Require this minimum absolute delta (microseconds) to flag a regression",
    )
    analyze.add_argument(
        "--gpu-ci-mode",
        choices=["off", "auto", "require"],
        default="off",
        help="GPU regression behavior: off, auto (check if GPU timings exist), require",
    )
    analyze.add_argument(
        "--gpu-threshold",
        type=float,
        default=5.0,
        help="GPU regression threshold percentage for baseline comparison",
    )
    analyze.add_argument(
        "--gpu-min-baseline-us",
        type=float,
        default=0.0,
        help="Ignore GPU ops with baseline mean below this value (microseconds)",
    )
    analyze.add_argument(
        "--gpu-min-regression-delta-us",
        type=float,
        default=0.0,
        help="Require this minimum absolute GPU delta (microseconds) to flag a regression",
    )
    analyze.set_defaults(func=_cmd_analyze)

    serve = subparsers.add_parser("serve", help="Serve a trace JSON file over HTTP")
    serve.add_argument("--trace", required=True, help="Trace JSON path")
    serve.add_argument("--port", type=int, default=8080, help="Port for local server")
    serve.add_argument(
        "--dashboard-build",
        choices=["auto", "always", "never"],
        default="auto",
        help="Dashboard build policy before serving (default: auto)",
    )
    serve.set_defaults(func=_cmd_serve)

    return parser


def main(argv: List[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
