#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace flux {

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
};

// Enable ATen callbacks and begin collecting records.
void flux_start();
// Disable callbacks and stop collecting.
void flux_stop();
// Remove all currently collected records.
void flux_clear_records();
// Return a merged, time-sorted snapshot of collected records.
std::vector<TimingRecord> flux_get_records();

}  // namespace flux
