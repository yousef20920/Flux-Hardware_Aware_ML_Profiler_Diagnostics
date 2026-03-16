from __future__ import annotations

import argparse
import json
import mimetypes
import runpy
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple
from urllib.parse import unquote, urlparse

from .aggregator import aggregate_records
from .analyzer import analyze_records
from .profiler import FluxProfiler
from .trace_exporter import export_trace


def _extract_records_from_trace(path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
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
            }
        )
    records.sort(key=lambda x: (x["start_us"], x["end_us"]))
    return records


def _op_means(records: Iterable[Dict[str, Any]]) -> Dict[str, float]:
    buckets: Dict[str, List[int]] = {}
    for item in records:
        op_name = str(item.get("op_name", "unknown"))
        buckets.setdefault(op_name, []).append(int(item.get("duration_us", 0)))
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


def _run_script_under_profiler(script_path: Path, script_args: List[str]) -> List[Dict[str, Any]]:
    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")

    old_argv = list(sys.argv)
    sys.argv = [str(script_path)] + script_args
    try:
        with FluxProfiler() as profiler:
            runpy.run_path(str(script_path), run_name="__main__")
        return profiler.records
    finally:
        sys.argv = old_argv


def _cmd_profile(args: argparse.Namespace) -> int:
    records = _run_script_under_profiler(Path(args.script), args.script_args)
    aggregate = aggregate_records(records)
    analyzed = analyze_records(
        records,
        peak_flops_tflops=args.peak_flops_tflops,
        memory_bandwidth_gbps=args.memory_bandwidth_gbps,
    )
    payload = export_trace(
        analyzed["records"],
        args.output,
        summary={**aggregate, "classification": analyzed["summary"]},
    )

    print(f"Wrote trace with {len(payload['traceEvents'])} events: {args.output}")
    _print_aggregate(aggregate)
    return 0


def _regression_report(
    current_means: Dict[str, float], baseline_means: Dict[str, float], threshold_pct: float
) -> Tuple[List[Tuple[str, float, float, float]], List[str]]:
    regressions: List[Tuple[str, float, float, float]] = []
    missing_in_baseline: List[str] = []

    for op_name, current in current_means.items():
        baseline = baseline_means.get(op_name)
        if baseline is None:
            missing_in_baseline.append(op_name)
            continue
        if baseline <= 0:
            continue
        delta_pct = ((current - baseline) / baseline) * 100.0
        if delta_pct > threshold_pct:
            regressions.append((op_name, baseline, current, delta_pct))

    regressions.sort(key=lambda x: x[3], reverse=True)
    return regressions, sorted(missing_in_baseline)


def _cmd_analyze(args: argparse.Namespace) -> int:
    trace_path = Path(args.trace)
    records = _extract_records_from_trace(trace_path)
    aggregate = aggregate_records(records)
    _print_aggregate(aggregate)

    if not args.baseline:
        return 0

    baseline_records = _extract_records_from_trace(Path(args.baseline))
    current_means = _op_means(records)
    baseline_means = _op_means(baseline_records)
    regressions, missing = _regression_report(current_means, baseline_means, args.threshold)

    print("")
    print(
        f"Regression check vs baseline ({args.baseline}) with threshold {args.threshold:.2f}%:"
    )
    if missing:
        print(f"  Ops missing in baseline: {', '.join(missing[:10])}")
    if not regressions:
        print("  No regressions detected.")
        return 0

    print("  Regressions:")
    for op_name, baseline, current, delta_pct in regressions:
        print(
            f"    {op_name}: baseline={_format_us(baseline)} "
            f"current={_format_us(current)} delta={delta_pct:.2f}%"
        )
    return 1


def _dashboard_dist_dir() -> Path:
    project_root = Path(__file__).resolve().parent.parent
    return project_root / "dashboard" / "dist"


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
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                    return

            if self.path in ("/", "/index.html"):
                body = _fallback_index_html(trace_name, dist_dir)
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
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
    analyze.set_defaults(func=_cmd_analyze)

    serve = subparsers.add_parser("serve", help="Serve a trace JSON file over HTTP")
    serve.add_argument("--trace", required=True, help="Trace JSON path")
    serve.add_argument("--port", type=int, default=8080, help="Port for local server")
    serve.set_defaults(func=_cmd_serve)

    return parser


def main(argv: List[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
