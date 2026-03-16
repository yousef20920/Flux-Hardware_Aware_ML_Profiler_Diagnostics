# Build dashboard assets in a Node stage.
FROM node:20-bookworm-slim AS dashboard-builder

WORKDIR /build/dashboard
COPY dashboard/package.json dashboard/package-lock.json ./
RUN npm ci
COPY dashboard/ ./
RUN npm run build

# Runtime image with PyTorch and Flux installed in editable mode.
FROM pytorch/pytorch:2.1.2-cpu

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt pyproject.toml setup.py README.md ./
COPY flux ./flux
COPY csrc ./csrc
COPY examples ./examples
COPY dashboard/package.json dashboard/package-lock.json ./dashboard/
COPY --from=dashboard-builder /build/dashboard/dist ./dashboard/dist

RUN python -m pip install --upgrade pip \
    && python -m pip install -e .

EXPOSE 8080

CMD ["bash", "-lc", "flux profile --script examples/profile_simple_model.py --output trace.json && flux serve --trace trace.json --port 8080"]
