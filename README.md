# Flux

Flux is a local-first ML profiler that captures low-level PyTorch operation timing, exports Chrome Trace JSON, and provides a timeline dashboard for debugging performance bottlenecks.

## Project Status

- Active development
- CPU profiling path is implemented and usable today
- CUDA timing mode is implemented with CPU fallback on non-CUDA environments
- GPU diagnostics summary and memory telemetry are implemented
- Dashboard + CI regression checks are implemented, including optional GPU regression gate

## Why Flux

Most ML performance issues are not visible from high-level model code alone. Flux helps you:

- Capture op-level timing from PyTorch internals
- Detect idle gaps and poor hardware utilization
- Compare traces in CI and fail builds on regressions
- Visualize execution with an interactive local dashboard

## Features

- C++/PyTorch instrumentation via `RecordFunction`
- Python profiler context manager (`FluxProfiler`)
- Trace export in Chrome Trace Event format
- GPU diagnostics summary (activity, SM estimate, memory pressure, transfer pressure)
- GPU memory telemetry (allocated/reserved/peak bytes)
- CLI workflow for profile, analyze, and serve
- React dashboard with zoomable timeline, CPU/GPU lanes, and event inspector
- CI templates for GitHub Actions and Jenkins

## Requirements

- Python 3.9+
- Node.js 20+ and npm (for dashboard build)
- PyTorch (installed automatically via `pip install -e .`)

## Quickstart

### 1) Install

```bash
python3 -m pip install -e .
```

If you hit a local extension ABI mismatch, reinstall with:

```bash
python3 -m pip install --no-build-isolation -e .
```

### 2) Create a trace

```bash
flux profile --script examples/profile_simple_model.py --output trace.json
```

### 3) Analyze the trace

```bash
flux analyze --trace trace.json
```

On a CUDA machine, you can force GPU timing and GPU gate checks:

```bash
flux profile --timing-mode cuda --script examples/profile_simple_model.py --output trace-gpu.json
flux analyze --trace trace-gpu.json --gpu-ci-mode require
```

### 4) Build and open dashboard

```bash
cd dashboard
npm install
npm run build
cd ..
flux serve --trace trace.json --port 8080
```

Open `http://127.0.0.1:8080`.

## CLI Reference

### `flux profile`

Run a Python script under Flux profiling and write a trace JSON.

```bash
flux profile --script <script.py> --output <trace.json>
```

Options:

- `--peak-flops-tflops` (default `10.0`)
- `--memory-bandwidth-gbps` (default `300.0`)
- `--timing-mode auto|cpu|cuda` (default `auto`)

Timing mode notes:

- `auto`: capture CUDA timing when CUDA support is available, otherwise CPU timing
- `cpu`: force host-side timing only
- `cuda`: require CUDA timing support, otherwise command exits with an error

### `flux analyze`

Analyze a trace and optionally compare against a baseline.

```bash
flux analyze --trace <trace.json>
```

```bash
flux analyze --trace <current.json> --baseline <baseline.json> --threshold 5
```

Returns non-zero when regression exceeds threshold.

Useful noise guards for CI:

- `--min-baseline-us` ignores very small baseline ops
- `--min-regression-delta-us` requires a minimum absolute delta
- `--gpu-ci-mode off|auto|require` controls GPU regression gating
- `--gpu-threshold` sets percentage threshold for GPU regressions
- `--gpu-min-baseline-us` ignores very small GPU baseline ops
- `--gpu-min-regression-delta-us` requires a minimum absolute GPU delta

GPU mode behavior:

- `off`: CPU regression gate only
- `auto`: run GPU regression check when CUDA timing data exists
- `require`: fail if CUDA timing data is missing, and fail on GPU regressions

### `flux serve`

Serve the dashboard and expose `/trace.json`.

```bash
flux serve --trace <trace.json> --port 8080
```

If `dashboard/dist` is missing, Flux serves a fallback page with build instructions.

## CI/CD Regression Flow

This repository includes:

- GitHub Actions workflow: `.github/workflows/perf-regression.yml`
- Jenkins pipeline: `ci/Jenkinsfile`
- Dockerfile: `ci/Dockerfile`

Default CI behavior uses an **ephemeral baseline** generated in the same run for stability across machines.

Baseline mode:

- `ephemeral` (default): generate baseline in current run
- `checked-in`: use `ci/baseline/trace-baseline.json`

Example local regression run:

```bash
# Warmup run helps reduce one-time startup skew.
flux profile --timing-mode auto --script examples/profile_simple_model.py --output trace-warmup.json
flux profile --timing-mode auto --script examples/profile_simple_model.py --output trace-current.json
flux profile --timing-mode auto --script examples/profile_simple_model.py --output trace-baseline.json
flux analyze \
  --trace trace-current.json \
  --baseline trace-baseline.json \
  --threshold 5 \
  --min-baseline-us 25 \
  --min-regression-delta-us 3 \
  --gpu-ci-mode auto \
  --gpu-threshold 5 \
  --gpu-min-baseline-us 25 \
  --gpu-min-regression-delta-us 3
```

To enforce GPU traces in CI:

```bash
flux analyze --trace trace-current.json --baseline trace-baseline.json --gpu-ci-mode require
```

## Repository Layout

```text
flux/                       Python package (profiler, analyzer, CLI)
csrc/                       C++ extension hooks
dashboard/                  React/Vite UI
ci/                         Dockerfile + Jenkins pipeline + baseline assets
.github/workflows/          GitHub Actions workflows
examples/                   Example scripts
PLAN.md                     Implementation roadmap
```

## Development

Install backend + frontend dependencies:

```bash
python3 -m pip install -e .
cd dashboard && npm ci
```

Build dashboard assets:

```bash
cd dashboard && npm run build
```

Useful checks:

```bash
flux --help
python3 -m flux --help
```

## Known Limitations

- CUDA event timing requires a CUDA-capable runtime
- `cuda` mode depends on CUDA tensor inputs for per-op GPU timing fields
- GPU diagnostics are estimate-based and intended for quick triage, not kernel-level precision
- Regression checks can still vary slightly between environments

## Roadmap

See [PLAN.md](./PLAN.md) for current phase-by-phase implementation plan.

## Contributing

Issues and pull requests are welcome.

For performance-related contributions, include:

- Reproduction steps
- Sample trace (`trace.json`) when possible
- Before/after `flux analyze` output

## License

A license file has not been added yet. Add `LICENSE` before public release.
