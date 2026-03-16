#pragma once

#include <cstdint>

namespace flux::cuda_hooks {

// Opaque CUDA event pair tracked per profiled op.
struct CudaEventPair {
  std::uintptr_t start_event{0};
  std::uintptr_t end_event{0};
  std::uintptr_t stream_ptr{0};
  int64_t stream_id{-1};
  int device_id{-1};
  bool active{false};
};

// Returns true when CUDA timing support is compiled and CUDA devices are present.
bool is_available();

// Returns current stream id for a device, or -1 when unavailable.
int64_t current_stream_id(int device_id);

// Records a CUDA start event on the current stream for the device.
bool record_start(CudaEventPair* pair, int device_id);

// Records the CUDA end event and returns elapsed microseconds, or -1 on failure.
double record_end_elapsed_us(CudaEventPair* pair);

// Safely destroys events and clears the pair.
void reset_pair(CudaEventPair* pair);

}  // namespace flux::cuda_hooks
