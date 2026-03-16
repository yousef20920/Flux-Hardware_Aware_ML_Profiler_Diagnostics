#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace flux {

enum class TimingMode {
  kAuto = 0,
  kCpu = 1,
  kCuda = 2,
};

// One timing sample for a single ATen op invocation.
struct TimingRecord {
  // Operator name (e.g. aten::mm, aten::relu).
  std::string op_name;
  // Start timestamp relative to flux_start().
  int64_t start_us;
  // End timestamp relative to flux_start().
  int64_t end_us;
  // Duration in microseconds.
  int64_t duration_us;
  // Logical thread id reported by RecordFunction.
  int64_t thread_id;
  // True when the op consumed CUDA tensor inputs.
  bool is_cuda;
  // CUDA device id for CUDA ops, else -1.
  int64_t device_id;
  // CUDA stream id for CUDA ops, else -1.
  int64_t stream_id;
  // Device-side elapsed time from CUDA events when available, else -1.
  double cuda_elapsed_us;
};

// Enable ATen callbacks and begin collecting records.
void flux_start();
// Disable callbacks and stop collecting.
void flux_stop();
// Remove all currently collected records.
void flux_clear_records();
// Return a merged, time-sorted snapshot of collected records.
std::vector<TimingRecord> flux_get_records();
// Configure timing mode: auto, cpu, or cuda.
void flux_set_timing_mode(const std::string& timing_mode);
// Return active timing mode string.
std::string flux_get_timing_mode();
// Return true if CUDA timing support is available.
bool flux_cuda_available();

}  // namespace flux
