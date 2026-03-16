# Flux: Full MVP Implementation Plan

## Context

Flux is a hardware-aware ML profiler and diagnostics tool. The goal is to build and extend the MVP across 6 phases: C++ profiling hooks, Python aggregation layer, React dashboard, CI/CD automation, CUDA timing integration, and GPU hardware metrics.

## Directory Structure

```
flux/                          # Python package
  __init__.py
  profiler.py                  # FluxProfiler context manager
  aggregator.py                # Group/summarize timing records
  analyzer.py                  # Memory-bound vs compute-bound classification
  trace_exporter.py            # Chrome Trace Event JSON export
  gpu_analyzer.py              # GPU-specific utilization and memory analysis
  cli.py                       # CLI entry point (profile, analyze, serve)

csrc/                          # C++ extension
  flux_hooks.h
  flux_hooks.cpp               # ATen dispatcher hooks + pybind11 bindings
  flux_cuda_hooks.cpp          # CUDA event timing + stream metadata

dashboard/                     # React timeline viewer
  package.json
  vite.config.js
  public/index.html
  src/
    App.jsx
    index.js
    components/
      Timeline.jsx             # Canvas-based Gantt chart
      TraceLoader.jsx           # File picker / drag-drop
      SummaryPanel.jsx          # Stats overview
      EventDetail.jsx           # Click-to-inspect panel
    utils/parseTrace.js         # Chrome Trace Event parser

ci/
  Dockerfile
  Jenkinsfile
  .github/workflows/perf-regression.yml

examples/
  profile_simple_model.py

tests/
  test_profiler.py
  test_aggregator.py
  test_trace_exporter.py
  test_cpp_hooks.py

setup.py                       # C++ extension build via torch.utils.cpp_extension
pyproject.toml
requirements.txt
.gitignore
```

## Phase 1: C++ Profiling Hooks

**Files:** `csrc/flux_hooks.cpp`, `csrc/flux_hooks.h`, `setup.py`, `pyproject.toml`, `.gitignore`

1. Use PyTorch's `at::RecordFunctionCallback` to hook into the ATen dispatcher
2. On pre-callback: capture `std::chrono::high_resolution_clock::now()`
3. On post-callback: compute elapsed microseconds, store in thread-local `vector<TimingRecord>`
4. Filter to relevant ops (`aten::linear`, `aten::relu`, `aten::addmm`, etc.)
5. Expose via pybind11: `flux_start()`, `flux_stop()`, `flux_get_records()` returning list of dicts
6. Build system: `setup.py` using `torch.utils.cpp_extension.CppExtension`

**Key decision:** Use `std::chrono` (not CUDA events) for MVP — works on CPU-only/macOS. CUDA event timing is a post-MVP enhancement.

## Phase 2: Python Data Aggregation & Export

**Files:** `flux/__init__.py`, `flux/profiler.py`, `flux/aggregator.py`, `flux/analyzer.py`, `flux/trace_exporter.py`, `flux/cli.py`, `examples/profile_simple_model.py`

1. `FluxProfiler` — context manager wrapping C++ hooks
2. `aggregator.py` — group records by op name, compute total/mean/min/max/count, find idle gaps
3. `analyzer.py` — classify ops as memory-bound vs compute-bound using heuristics (compare actual duration against theoretical compute time based on matrix dimensions and GPU peak FLOPS)
4. `trace_exporter.py` — export to Chrome Trace Event JSON (`{"traceEvents": [...]}` with `ph: "X"` complete events)
5. `cli.py` — argparse CLI with subcommands:
   - `flux profile --script <path> --output trace.json`
   - `flux analyze --trace trace.json [--baseline baseline.json --threshold 5]`
   - `flux serve --trace trace.json --port 8080`

## Phase 3: React Timeline Dashboard

**Files:** `dashboard/` directory (see structure above)

1. Vite + React SPA, served by `flux serve` via Python `http.server`
2. `parseTrace.js` — parse Chrome Trace Event JSON, group by thread ID into swim lanes
3. `Timeline.jsx` — Canvas-based Gantt chart with zoom/scroll, color-coded by classification (green=compute-bound, red=memory-bound, grey=unknown)
4. `TraceLoader.jsx` — drag-and-drop file loader
5. `SummaryPanel.jsx` — total time, op breakdown, bound classification percentages
6. `EventDetail.jsx` — detail panel on event click

## Phase 4: CI/CD Automation

**Files:** `ci/Dockerfile`, `ci/Jenkinsfile`, `ci/.github/workflows/perf-regression.yml`

1. Dockerfile based on `pytorch/pytorch` image, installs Flux, pre-builds dashboard
2. GitHub Actions workflow: profile on PR, compare against baseline, fail if >5% regression
3. Jenkinsfile with Build → Profile → Regression Check → Archive stages
4. Regression detection logic in `flux analyze --baseline --threshold`: compare per-op mean durations, exit non-zero if threshold exceeded

## Phase 5: CUDA Timing Integration

**Files:** `csrc/flux_cuda_hooks.cpp`, `csrc/flux_hooks.cpp`, `csrc/flux_hooks.h`, `flux/profiler.py`, `flux/trace_exporter.py`, `flux/cli.py`

1. Add optional CUDA timing path using CUDA events (`cudaEventRecord`) for GPU ops
2. Capture both host duration (`std::chrono`) and device duration (`cuda_elapsed_us`) when CUDA is available
3. Capture per-event GPU metadata: `device_id`, `stream_id`, and `is_cuda`
4. Extend pybind payload and Python records to include the new GPU timing fields
5. Add CLI timing selection:
   - `--timing-mode auto|cpu|cuda` (default `auto`)
6. Keep CPU fallback behavior unchanged on non-CUDA environments

## Phase 6: GPU Diagnostics and CI Gate

**Files:** `flux/gpu_analyzer.py`, `flux/analyzer.py`, `flux/trace_exporter.py`, `flux/cli.py`, `dashboard/src/utils/parseTrace.js`, `dashboard/src/components/SummaryPanel.jsx`, `dashboard/src/components/Timeline.jsx`, `.github/workflows/perf-regression.yml`, `ci/Jenkinsfile`

1. Add GPU memory telemetry (allocated/reserved/peak bytes) to profiling outputs
2. Add GPU diagnostics summary:
   - SM utilization estimate
   - memory bandwidth pressure estimate
   - host-to-device transfer pressure
3. Render GPU metrics and lanes in the dashboard (summary cards + timeline metadata)
4. Extend `flux analyze` output with GPU-specific regression deltas
5. Add optional GPU CI mode that compares `cuda_elapsed_us` against baseline and fails when threshold is exceeded

## Implementation Order

| Step | What | Depends On |
|------|------|------------|
| 1 | Build configs: `setup.py`, `pyproject.toml`, `.gitignore`, `requirements.txt` | — |
| 2 | C++ extension: `csrc/flux_hooks.cpp`, `csrc/flux_hooks.h` | Step 1 |
| 3 | Python package: `flux/profiler.py`, `flux/aggregator.py`, `flux/analyzer.py`, `flux/trace_exporter.py` | Step 2 |
| 4 | CLI: `flux/cli.py` + example script | Step 3 |
| 5 | Dashboard: `dashboard/` (can partially overlap with Steps 3-4) | — |
| 6 | Wire `flux serve` to dashboard | Steps 4, 5 |
| 7 | CI/CD: Dockerfile, Jenkinsfile, GitHub Actions | Steps 4, 5 |
| 8 | Tests for CPU pipeline | Steps 2-4 |
| 9 | CUDA timing integration | Steps 2-4 |
| 10 | GPU diagnostics + dashboard GPU panels + CI GPU gate | Steps 5, 7, 9 |

## Verification

1. **Phase 1:** `pip install -e .` succeeds, `python -c "import flux._C; flux._C.flux_start(); flux._C.flux_stop(); print(flux._C.flux_get_records())"` works
2. **Phase 2:** `python examples/profile_simple_model.py` produces a valid `trace.json`, `flux analyze --trace trace.json` prints summary
3. **Phase 3:** `flux serve --trace trace.json` opens browser with interactive timeline
4. **Phase 4:** `docker build -f ci/Dockerfile .` succeeds, GitHub Actions workflow syntax is valid (`actionlint`)
5. **Phase 5:** On a CUDA machine, `flux profile --timing-mode cuda --script examples/profile_simple_model.py --output trace-gpu.json` includes `cuda_elapsed_us`, `device_id`, and `stream_id`
6. **Phase 6:** `flux analyze --trace trace-gpu.json --baseline ci/baseline/trace-baseline.json --threshold 5` reports GPU diagnostics and enforces GPU regression checks
7. **All tests pass:** `pytest tests/`
