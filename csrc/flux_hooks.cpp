#include "flux_hooks.h"
#include "flux_cuda_hooks.h"

#include <ATen/record_function.h>
#include <torch/extension.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

namespace py = pybind11;

namespace {

// High-resolution wall-clock for host-side timing in the MVP.
using Clock = std::chrono::high_resolution_clock;
using TimePoint = Clock::time_point;

struct ThreadLocalRecordBuffer;

void register_record_buffer(ThreadLocalRecordBuffer* buffer);
void unregister_record_buffer(ThreadLocalRecordBuffer* buffer);

// Context passed from start callback to end callback for one op.
struct FluxObserverContext final : public at::ObserverContext {
  FluxObserverContext(
      TimePoint start_time,
      std::string op_name,
      uint64_t logical_thread_id,
      bool is_cuda,
      int64_t device_id,
      int64_t stream_id)
      : start_time(start_time),
        op_name(std::move(op_name)),
        logical_thread_id(logical_thread_id),
        is_cuda(is_cuda),
        device_id(device_id),
        stream_id(stream_id) {}

  TimePoint start_time;
  std::string op_name;
  uint64_t logical_thread_id;
  bool is_cuda{false};
  int64_t device_id{-1};
  int64_t stream_id{-1};
  flux::cuda_hooks::CudaEventPair cuda_events{};
};

struct ThreadLocalRecordBuffer {
  // Register this thread's buffer so records can be merged at read time.
  ThreadLocalRecordBuffer() {
    register_record_buffer(this);
  }

  // Unregister on thread teardown to avoid dangling pointers.
  ~ThreadLocalRecordBuffer() {
    unregister_record_buffer(this);
  }

  std::vector<flux::TimingRecord> records;
};

// Global profiler session state.
std::atomic<bool> g_is_running{false};
std::atomic<flux::TimingMode> g_timing_mode{flux::TimingMode::kAuto};
std::mutex g_registry_mutex;
std::vector<ThreadLocalRecordBuffer*> g_record_buffers;
std::optional<at::CallbackHandle> g_callback_handle;
TimePoint g_session_start_time = Clock::now();

ThreadLocalRecordBuffer& tls_record_buffer() {
  // One buffer per thread minimizes lock contention during callback writes.
  thread_local ThreadLocalRecordBuffer buffer;
  return buffer;
}

void register_record_buffer(ThreadLocalRecordBuffer* buffer) {
  // Protect shared registry used by all threads.
  std::lock_guard<std::mutex> lock(g_registry_mutex);
  g_record_buffers.push_back(buffer);
}

void unregister_record_buffer(ThreadLocalRecordBuffer* buffer) {
  // Safe removal when thread-local buffers are destroyed.
  std::lock_guard<std::mutex> lock(g_registry_mutex);
  g_record_buffers.erase(
      std::remove(g_record_buffers.begin(), g_record_buffers.end(), buffer),
      g_record_buffers.end());
}

bool is_tracked_op(std::string_view op_name) {
  // MVP: profile a focused set of common ops instead of every ATen event.
  static const std::unordered_set<std::string> kTrackedOps = {
      "aten::linear",
      "aten::relu",
      "aten::addmm",
      "aten::mm",
      "aten::matmul",
      "aten::conv1d",
      "aten::conv2d",
      "aten::conv3d",
      "aten::convolution",
      "aten::batch_norm",
      "aten::layer_norm",
      "aten::gelu",
  };
  return kTrackedOps.find(std::string(op_name)) != kTrackedOps.end();
}

flux::TimingMode parse_timing_mode(const std::string& mode) {
  if (mode == "auto") {
    return flux::TimingMode::kAuto;
  }
  if (mode == "cpu") {
    return flux::TimingMode::kCpu;
  }
  if (mode == "cuda") {
    return flux::TimingMode::kCuda;
  }
  throw std::runtime_error(
      "Invalid timing mode. Expected one of: auto, cpu, cuda.");
}

std::string timing_mode_to_string(flux::TimingMode mode) {
  switch (mode) {
    case flux::TimingMode::kAuto:
      return "auto";
    case flux::TimingMode::kCpu:
      return "cpu";
    case flux::TimingMode::kCuda:
      return "cuda";
  }
  return "auto";
}

struct CudaInputInfo {
  bool is_cuda{false};
  int64_t device_id{-1};
};

CudaInputInfo infer_cuda_input_info(const at::RecordFunction& fn) {
  CudaInputInfo info{};
  for (const auto& input : fn.inputs()) {
    if (input.isTensor()) {
      const auto& t = input.toTensor();
      if (t.defined() && t.is_cuda()) {
        info.is_cuda = true;
        info.device_id = t.get_device();
        return info;
      }
      continue;
    }

    if (input.isTensorList()) {
      const auto list = input.toTensorVector();
      for (const auto& t : list) {
        if (t.defined() && t.is_cuda()) {
          info.is_cuda = true;
          info.device_id = t.get_device();
          return info;
        }
      }
    }
  }
  return info;
}

bool should_capture_cuda_timing(bool is_cuda_op) {
  if (!is_cuda_op) {
    return false;
  }

  const auto mode = g_timing_mode.load(std::memory_order_relaxed);
  if (mode == flux::TimingMode::kCpu) {
    return false;
  }
  if (mode == flux::TimingMode::kCuda) {
    return flux::cuda_hooks::is_available();
  }
  return flux::cuda_hooks::is_available();
}

int64_t to_relative_us(TimePoint t) {
  // Store relative timestamps to keep trace payloads compact and readable.
  return std::chrono::duration_cast<std::chrono::microseconds>(
             t - g_session_start_time)
      .count();
}

std::unique_ptr<at::ObserverContext> on_record_start(
    const at::RecordFunction& fn) {
  // Fast exit when session is not active.
  if (!g_is_running.load(std::memory_order_relaxed)) {
    return nullptr;
  }

  const char* op_name = fn.name();
  // Skip events outside our tracked operator set.
  if (op_name == nullptr || !is_tracked_op(op_name)) {
    return nullptr;
  }

  const auto cuda_info = infer_cuda_input_info(fn);
  int64_t stream_id = -1;
  if (cuda_info.is_cuda) {
    stream_id = flux::cuda_hooks::current_stream_id(
        static_cast<int>(cuda_info.device_id));
  }

  // Save per-op metadata so end callback can compute final duration.
  auto context = std::make_unique<FluxObserverContext>(
      Clock::now(),
      std::string(op_name),
      fn.threadId(),
      cuda_info.is_cuda,
      cuda_info.device_id,
      stream_id);

  if (should_capture_cuda_timing(context->is_cuda)) {
    flux::cuda_hooks::record_start(
        &context->cuda_events, static_cast<int>(context->device_id));
    if (context->cuda_events.active) {
      context->stream_id = context->cuda_events.stream_id;
    }
  }

  return context;
}

void on_record_end(const at::RecordFunction&, at::ObserverContext* context) {
  // Null context means start callback opted out for this event.
  if (context == nullptr || !g_is_running.load(std::memory_order_relaxed)) {
    return;
  }

  auto* flux_context = static_cast<FluxObserverContext*>(context);
  const auto end_time = Clock::now();
  const double cuda_elapsed_us =
      flux::cuda_hooks::record_end_elapsed_us(&flux_context->cuda_events);

  flux::TimingRecord record{
      flux_context->op_name,
      to_relative_us(flux_context->start_time),
      to_relative_us(end_time),
      std::chrono::duration_cast<std::chrono::microseconds>(
          end_time - flux_context->start_time)
          .count(),
      static_cast<int64_t>(flux_context->logical_thread_id),
      flux_context->is_cuda,
      flux_context->device_id,
      flux_context->stream_id,
      cuda_elapsed_us};

  // Append to this thread's local buffer.
  auto& buffer = tls_record_buffer();
  buffer.records.push_back(std::move(record));
}

std::vector<flux::TimingRecord> collect_records() {
  // Merge thread-local buffers into one snapshot for Python callers.
  std::vector<flux::TimingRecord> records;
  std::lock_guard<std::mutex> lock(g_registry_mutex);
  for (auto* buffer : g_record_buffers) {
    records.insert(
        records.end(), buffer->records.begin(), buffer->records.end());
  }
  return records;
}

}  // namespace

namespace flux {

void flux_start() {
  // Idempotent start: no-op if already running.
  if (g_is_running.exchange(true)) {
    return;
  }

  // Reset timeline origin and previous samples for a fresh session.
  g_session_start_time = Clock::now();
  flux_clear_records();

  const auto mode = g_timing_mode.load(std::memory_order_relaxed);
  if (mode == TimingMode::kCuda && !flux::cuda_hooks::is_available()) {
    g_is_running.store(false);
    throw std::runtime_error(
        "CUDA timing mode requested, but CUDA timing support is unavailable.");
  }

  // Register thread-local ATen callback pair.
  auto callback = at::RecordFunctionCallback(&on_record_start, &on_record_end);
  callback.needsInputs(true);
  callback.needsIds(true);
  g_callback_handle = at::addThreadLocalCallback(std::move(callback));
}

void flux_stop() {
  // Idempotent stop: no-op if already stopped.
  if (!g_is_running.exchange(false)) {
    return;
  }

  // Remove callback handle to stop receiving ATen events.
  if (g_callback_handle.has_value()) {
    at::removeCallback(*g_callback_handle);
    g_callback_handle.reset();
  }
}

void flux_clear_records() {
  // Clear all known thread-local buffers.
  std::lock_guard<std::mutex> lock(g_registry_mutex);
  for (auto* buffer : g_record_buffers) {
    buffer->records.clear();
  }
}

std::vector<TimingRecord> flux_get_records() {
  // Keep lifecycle explicit: read only after stop().
  if (g_is_running.load()) {
    throw std::runtime_error(
        "flux_get_records() called while profiler is running. Call flux_stop() first.");
  }

  auto records = collect_records();
  // Ensure deterministic ordering for downstream analysis/export.
  std::sort(
      records.begin(),
      records.end(),
      [](const TimingRecord& a, const TimingRecord& b) {
        if (a.start_us == b.start_us) {
          return a.end_us < b.end_us;
        }
        return a.start_us < b.start_us;
      });
  return records;
}

void flux_set_timing_mode(const std::string& timing_mode) {
  if (g_is_running.load()) {
    throw std::runtime_error(
        "Cannot change timing mode while profiler is running.");
  }
  g_timing_mode.store(parse_timing_mode(timing_mode), std::memory_order_relaxed);
}

std::string flux_get_timing_mode() {
  return timing_mode_to_string(g_timing_mode.load(std::memory_order_relaxed));
}

bool flux_cuda_available() {
  return flux::cuda_hooks::is_available();
}

}  // namespace flux

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // Python entrypoints for the profiler lifecycle.
  m.doc() = "Flux ATen record function hooks";

  m.def(
      "flux_start",
      &flux::flux_start,
      "Start collecting ATen operation timing records.");
  m.def(
      "flux_stop",
      &flux::flux_stop,
      "Stop collecting ATen operation timing records.");
  m.def(
      "flux_clear_records",
      &flux::flux_clear_records,
      "Clear all collected timing records.");
  m.def(
      "flux_set_timing_mode",
      &flux::flux_set_timing_mode,
      "Set timing mode (auto, cpu, cuda).");
  m.def(
      "flux_get_timing_mode",
      &flux::flux_get_timing_mode,
      "Get active timing mode.");
  m.def(
      "flux_cuda_available",
      &flux::flux_cuda_available,
      "Return true if CUDA timing support is available.");
  m.def("flux_get_records", []() {
    // Convert C++ records into a Python-native list[dict].
    py::list output;
    for (const auto& record : flux::flux_get_records()) {
      py::dict item;
      item["op_name"] = record.op_name;
      item["start_us"] = record.start_us;
      item["end_us"] = record.end_us;
      item["duration_us"] = record.duration_us;
      item["thread_id"] = record.thread_id;
      item["is_cuda"] = record.is_cuda;
      item["device_id"] = record.device_id;
      item["stream_id"] = record.stream_id;
      item["cuda_elapsed_us"] = record.cuda_elapsed_us;
      output.append(std::move(item));
    }
    return output;
  });
}
