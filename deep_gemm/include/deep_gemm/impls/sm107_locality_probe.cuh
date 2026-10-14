#pragma once

#include <cstdint>
#include <cutlass/cutlass.h>

#include <deep_gemm/ptx/utils.cuh>

namespace deep_gemm {

// Locality-domain probe (port of upstream DeepGEMM 26/09/30 `locality_domain_probe_chase`): one block per SM chases
// pointer chains through `kNumChunksPerSM` chunks of `buf` (one hop per 128-byte line, L2 flushed by the host), and
// records the cycles per hop. Chunks homed in the SM's own locality domain are ~1.5x faster on Rubin (VR200).
template <uint32_t kNumHops, uint32_t kNumChunksPerSM, uint32_t kNumChunkBytes>
CUTLASS_GLOBAL __launch_bounds__(32, 1)
void sm107_locality_probe_chase(const uint32_t* buf, uint16_t* out) {
    if (threadIdx.x != 0)
        return;
    const auto ld_cg = [](const uint32_t* ptr) {
        uint32_t value;
        asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(value) : "l"(ptr));
        return value;
    };
    const auto sm_idx = ptx::get_sm_idx();
    for (uint32_t chunk_idx = 0; chunk_idx < kNumChunksPerSM; ++ chunk_idx) {
        const auto chunk = buf + static_cast<uint64_t>(sm_idx * kNumChunksPerSM + chunk_idx) * (kNumChunkBytes / 4);
        uint32_t idx = ld_cg(chunk);
        const auto start = clock64();
        #pragma unroll
        for (uint32_t i = 0; i < kNumHops; ++ i)
            idx = ld_cg(chunk + idx);
        const auto cycles = clock64() - start;
        out[sm_idx * kNumChunksPerSM + chunk_idx] = idx == 0xffffffffu ? 0u : static_cast<uint16_t>(cycles / kNumHops);
    }
}

}  // namespace deep_gemm
