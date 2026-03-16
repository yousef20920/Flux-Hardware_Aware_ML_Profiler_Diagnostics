#include "flux_cuda_hooks.h"

#include <cstdio>
#include <cstdlib>

#if defined(__has_include)
#if __has_include(<cuda_runtime_api.h>) && __has_include(<c10/cuda/CUDAFunctions.h>) && \
    __has_include(<c10/cuda/CUDAStream.h>) && __has_include(<c10/cuda/CUDAGuard.h>)
#define FLUX_HAS_CUDA_TIMING 1
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime_api.h>
#else
#define FLUX_HAS_CUDA_TIMING 0
#endif
#else
#define FLUX_HAS_CUDA_TIMING 0
#endif

namespace flux::cuda_hooks {

#if FLUX_HAS_CUDA_TIMING

namespace {

bool is_success(cudaError_t err) {
  return err == cudaSuccess;
}

bool debug_enabled() {
  const char* env = std::getenv("FLUX_CUDA_DEBUG");
  return env != nullptr && env[0] != '\0' && env[0] != '0';
}

void log_cuda_error(const char* stage, cudaError_t err, int device_id) {
  if (!debug_enabled()) {
    return;
  }
  std::fprintf(
      stderr,
      "[flux-cuda] %s failed device=%d code=%d name=%s msg=%s\n",
      stage,
      device_id,
      static_cast<int>(err),
      cudaGetErrorName(err),
      cudaGetErrorString(err));
}

cudaEvent_t to_event(std::uintptr_t ptr) {
  return reinterpret_cast<cudaEvent_t>(ptr);
}

cudaStream_t to_stream(std::uintptr_t ptr) {
  return reinterpret_cast<cudaStream_t>(ptr);
}

}  // namespace

bool is_available() {
  return c10::cuda::device_count() > 0;
}

int64_t current_stream_id(int device_id) {
  if (!is_available()) {
    return -1;
  }
  int resolved_device = device_id >= 0 ? device_id : c10::cuda::current_device();
  auto stream = c10::cuda::getCurrentCUDAStream(resolved_device).stream();
  return static_cast<int64_t>(reinterpret_cast<std::uintptr_t>(stream));
}

void reset_pair(CudaEventPair* pair) {
  if (pair == nullptr) {
    return;
  }
  if (pair->start_event != 0) {
    cudaEventDestroy(to_event(pair->start_event));
  }
  if (pair->end_event != 0) {
    cudaEventDestroy(to_event(pair->end_event));
  }
  pair->start_event = 0;
  pair->end_event = 0;
  pair->stream_ptr = 0;
  pair->stream_id = -1;
  pair->device_id = -1;
  pair->active = false;
}

bool record_start(CudaEventPair* pair, int device_id) {
  if (pair == nullptr || !is_available()) {
    return false;
  }

  reset_pair(pair);

  int resolved_device = device_id >= 0 ? device_id : c10::cuda::current_device();
  // Ensure event lifecycle and stream operations execute on the intended device.
  c10::cuda::CUDAGuard device_guard(resolved_device);
  auto stream = c10::cuda::getCurrentCUDAStream(resolved_device).stream();
  cudaEvent_t start_event = nullptr;
  cudaEvent_t end_event = nullptr;

  auto err = cudaEventCreateWithFlags(&start_event, cudaEventDefault);
  if (!is_success(err)) {
    log_cuda_error("cudaEventCreateWithFlags(start)", err, resolved_device);
    reset_pair(pair);
    return false;
  }
  err = cudaEventCreateWithFlags(&end_event, cudaEventDefault);
  if (!is_success(err)) {
    log_cuda_error("cudaEventCreateWithFlags(end)", err, resolved_device);
    cudaEventDestroy(start_event);
    reset_pair(pair);
    return false;
  }
  err = cudaEventRecord(start_event, stream);
  if (!is_success(err)) {
    log_cuda_error("cudaEventRecord(start)", err, resolved_device);
    cudaEventDestroy(start_event);
    cudaEventDestroy(end_event);
    reset_pair(pair);
    return false;
  }

  pair->start_event = reinterpret_cast<std::uintptr_t>(start_event);
  pair->end_event = reinterpret_cast<std::uintptr_t>(end_event);
  pair->stream_ptr = reinterpret_cast<std::uintptr_t>(stream);
  pair->stream_id = static_cast<int64_t>(pair->stream_ptr);
  pair->device_id = resolved_device;
  pair->active = true;
  return true;
}

double record_end_elapsed_us(CudaEventPair* pair) {
  if (pair == nullptr || !pair->active || pair->start_event == 0 ||
      pair->end_event == 0 || pair->stream_ptr == 0) {
    return -1.0;
  }

  int resolved_device = pair->device_id >= 0 ? pair->device_id : c10::cuda::current_device();
  // Re-enter the same device context before recording/syncing the end event.
  c10::cuda::CUDAGuard device_guard(resolved_device);
  auto start_event = to_event(pair->start_event);
  auto end_event = to_event(pair->end_event);
  auto stream = to_stream(pair->stream_ptr);

  auto err = cudaEventRecord(end_event, stream);
  if (!is_success(err)) {
    log_cuda_error("cudaEventRecord(end)", err, resolved_device);
    reset_pair(pair);
    return -1.0;
  }
  err = cudaEventSynchronize(end_event);
  if (!is_success(err)) {
    log_cuda_error("cudaEventSynchronize(end)", err, resolved_device);
    reset_pair(pair);
    return -1.0;
  }

  float elapsed_ms = 0.0f;
  err = cudaEventElapsedTime(&elapsed_ms, start_event, end_event);
  if (!is_success(err)) {
    log_cuda_error("cudaEventElapsedTime", err, resolved_device);
    reset_pair(pair);
    return -1.0;
  }

  reset_pair(pair);
  return static_cast<double>(elapsed_ms) * 1000.0;
}

#else

bool is_available() {
  return false;
}

int64_t current_stream_id(int) {
  return -1;
}

void reset_pair(CudaEventPair* pair) {
  if (pair == nullptr) {
    return;
  }
  pair->start_event = 0;
  pair->end_event = 0;
  pair->stream_ptr = 0;
  pair->stream_id = -1;
  pair->device_id = -1;
  pair->active = false;
}

bool record_start(CudaEventPair* pair, int) {
  reset_pair(pair);
  return false;
}

double record_end_elapsed_us(CudaEventPair* pair) {
  reset_pair(pair);
  return -1.0;
}

#endif

}  // namespace flux::cuda_hooks
