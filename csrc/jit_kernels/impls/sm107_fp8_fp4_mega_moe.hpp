#pragma once

#include <array>
#include <filesystem>
#include <format>
#include <torch/python.h>

#include <deep_gemm/layout/mega_moe.cuh>
#include <deep_gemm/layout/sym_buffer.cuh>

#include "../../runtime/runtime.hpp"
#include "../../utils/exception.hpp"
#include "../heuristics/mega_moe.hpp"
#include "runtime_utils.hpp"
#include "sm107_locality.hpp"

namespace deep_gemm {

// Launch with Rubin oversized shared memory: identical to `deep_jit::cuda::Kernel::launch`, except that the
// function is switched to `CU_SHARED_MEMORY_MODE_ALLOW_OVERSIZED_SHARED_MEMORY` (which is what grants more than the
// portable opt-in size; `CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES` must not be set then)
template <typename Kernel, typename... Args>
static void sm107_launch_oversized(const Kernel& kernel, const deep_jit::cuda::LaunchOptions& options, const Args&... args) {
    using namespace deep_jit::cuda;
    DJ_CUDA_DRIVER_CHECK(driver::lazy_cuFuncSetAttribute(
        kernel.kernel_handle, CU_FUNC_ATTRIBUTE_SHARED_MEMORY_MODE, CU_SHARED_MEMORY_MODE_ALLOW_OVERSIZED_SHARED_MEMORY));
    std::array<CUlaunchAttribute, 2> attributes{};
    unsigned int num_attributes = 0;
    if (options.cluster_dim->x > 1) {
        auto& attribute = attributes[num_attributes ++];
        attribute.id = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
        attribute.value.clusterDim.x = options.cluster_dim->x;
        attribute.value.clusterDim.y = 1;
        attribute.value.clusterDim.z = 1;
    }
    if (*options.enable_pdl) {
        auto& attribute = attributes[num_attributes ++];
        attribute.id = CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION;
        attribute.value.programmaticStreamSerializationAllowed = 1;
    }
    void* kernel_arg_ptrs[sizeof...(Args) + 1] = {kernel_arg_pointer(args)..., nullptr};
    CUlaunchConfig config{};
    config.gridDimX = options.grid_dim->x, config.gridDimY = options.grid_dim->y, config.gridDimZ = options.grid_dim->z;
    config.blockDimX = options.block_dim->x, config.blockDimY = options.block_dim->y, config.blockDimZ = options.block_dim->z;
    config.sharedMemBytes = *options.num_smem_bytes;
    config.hStream = options.stream ? *options.stream : at::cuda::getCurrentCUDAStream().stream();
    config.attrs = num_attributes == 0 ? nullptr : attributes.data();
    config.numAttrs = num_attributes;
    DJ_CUDA_DRIVER_CHECK(driver::lazy_cuLaunchKernelEx(&config, kernel.kernel_handle, kernel_arg_ptrs, nullptr));
}

// Rubin (SM107) variant of `sm100_fp8_fp4_mega_moe`: same algorithm and math (up to FP32 accumulation order).
// Opt-in via `DG_MEGA_MOE_SM107=1` (see `apis/mega_moe.hpp`). Differences, each with an env knob (defaults = best):
//   DG_MEGA_MOE_SM107_UMMA_K:          64 (Rubin K=64 block-scaled UMMA, FP4 weights packed in smem, 8 KB weight
//                                      stages, shared-expert FP8 tiles in half-K stages) or 32 (SM100 shape)
//   DG_MEGA_MOE_SM107_DISPATCH_PIPE:   3 (dispatch pull through registers, no smem bounce), 1 (TMA), 2 (TMA, 2 buffers)
//   DG_MEGA_MOE_SM107_BLOCK_K:         128 (default) or 256 K per pipeline stage
//   DG_MEGA_MOE_SM107_SMEM / _STAGES:  stage sizing capacity (334848 = oversized smem; slower, see README) / stage cap
//   DG_MEGA_MOE_SM107_EPI_PF:          0 (default) / 1: prefetch the next epilogue store block from TMEM
//   DG_MEGA_MOE_SM107_PREFETCH:        0 (default) / n: L2-prefetch routed weights n k-blocks ahead
//   DG_MEGA_MOE_SM107_ARCH:            JIT target override, default the image's "100f"; "107a" uses CUTLASS 4.8 headers
//   DG_MEGA_MOE_SM107_TRACE_PTR / _DEBUG: debug-only tracing and component skipping (results invalid with _DEBUG)
static void sm107_fp8_fp4_mega_moe(
    const torch::Tensor& y,
    const torch::Tensor& l1_acts, const torch::Tensor& l1_acts_sf,
    const torch::Tensor& l2_acts, const torch::Tensor& l2_acts_sf,
    const torch::Tensor& shared_l1_acts, const torch::Tensor& shared_l1_acts_sf,
    const torch::Tensor& shared_l2_acts, const torch::Tensor& shared_l2_acts_sf,
    const torch::Tensor& l1_weights, const torch::Tensor& l2_weights,
    const torch::Tensor& l1_weights_sf, const torch::Tensor& l2_weights_sf,
    const torch::Tensor& shared_l1_weights, const torch::Tensor& shared_l2_weights,
    const torch::Tensor& shared_l1_weights_sf, const torch::Tensor& shared_l2_weights_sf,
    const std::optional<torch::Tensor> cumulative_local_expert_recv_stats,
    const std::vector<int64_t>& sym_buffer_ptrs,
    const int& rank_idx, const int& num_max_tokens_per_rank,
    const int& num_experts_per_rank,
    const int& num_shared_experts,
    const int& num_tokens, const int& num_topk,
    const int& hidden, const int& intermediate_hidden,
    const float& activation_clamp,
    const float& activation_alpha,
    const float& activation_beta,
    const bool& fast_math
) {
    const auto num_ranks = static_cast<int>(sym_buffer_ptrs.size());
    const auto num_experts = num_experts_per_rank * num_ranks;
    const auto num_ring_tokens = static_cast<int>(l1_acts.size(0));
    const auto num_sf_ring_tokens = static_cast<int>(l1_acts_sf.size(0));
    const auto shared_intermediate_hidden = intermediate_hidden * num_shared_experts;

    // Heuristics
    auto config = get_mega_moe_config(
        num_ranks, num_experts, num_experts_per_rank,
        num_max_tokens_per_rank, num_tokens, num_topk, hidden, intermediate_hidden,
        num_ring_tokens, num_sf_ring_tokens,
        MmaKind::MXFP8FP4);

    // K per pipeline stage (`DG_MEGA_MOE_SM107_BLOCK_K`, 128 or 256): 256 halves the per-k-block handshakes
    config.block_k = deep_jit::get_env<int>("DG_MEGA_MOE_SM107_BLOCK_K", config.block_k);
    DG_HOST_ASSERT(config.block_k == 128 or config.block_k == 256);

    // SM107 pipeline depth: Rubin can opt into "oversized" shared memory (up to 327 KB per block, at the cost of an
    // 8 KB L1), which buys more TMA stages. `DG_MEGA_MOE_SM107_SMEM` sets the capacity used to size the stages
    // (default: the portable 232448 B), `DG_MEGA_MOE_SM107_STAGES` caps the stage count.
    {
        const int smem_capacity = deep_jit::get_env<int>("DG_MEGA_MOE_SM107_SMEM", SM100ArchSpec::smem_capacity);
        const int max_stages = deep_jit::get_env<int>("DG_MEGA_MOE_SM107_STAGES", 0);
        const auto get_pipeline = [&](const int& capacity) {
            return get_pipeline_config_for_mega_moe(
                capacity, num_experts, hidden,
                config.block_m, config.block_n, config.block_k, config.num_bytes_per_pull, config.store_block_m,
                config.sf_block_m, config.sf_block_n, 32,
                config.num_dispatch_threads / 32, config.num_epilogue_threads / 32, MmaKind::MXFP8FP4);
        };
        // Derive the fixed and per-stage sizes from two (host-only) evaluations of the SM100 sizing
        const auto [num_stages_1, smem_size_1] = get_pipeline(SM100ArchSpec::smem_capacity);
        const auto [num_stages_2, smem_size_2] = get_pipeline(SM100ArchSpec::smem_capacity + (1 << 18));
        // Packed FP4 weights (K=64 UMMA) halve the weight stage (8 KB instead of 16 KB); shared FP8 weights then run in
        // half-K stages (see the kernel)
        const bool packed_weights = deep_jit::get_env<int>("DG_MEGA_MOE_SM107_UMMA_K", 64) == 64 and l1_weights.scalar_type() == kPackedFP4;
        const int smem_per_stage = (smem_size_2 - smem_size_1) / (num_stages_2 - num_stages_1) -
            (packed_weights ? config.load_block_n * config.block_k / 2 : 0);
        // Double-buffered dispatch pull (`DG_MEGA_MOE_SM107_DISPATCH_PIPE=2`) adds one pull buffer + barrier per dispatch warp
        const int dispatch_pipe = deep_jit::get_env<int>("DG_MEGA_MOE_SM107_DISPATCH_PIPE", 3);
        DG_HOST_ASSERT(dispatch_pipe >= 1 and dispatch_pipe <= 3);
        const int pull_buffer_bytes = (config.num_dispatch_threads / 32) * config.num_bytes_per_pull;
        const int smem_fixed = smem_size_1 - num_stages_1 * smem_per_stage +
            (dispatch_pipe == 2 ? pull_buffer_bytes + (config.num_dispatch_threads / 32) * 8 : 0) -
            (dispatch_pipe == 3 ? pull_buffer_bytes - 1024 : 0);
        int num_stages = std::min((smem_capacity - smem_fixed) / smem_per_stage, 32);
        if (max_stages > 0)
            num_stages = std::min(num_stages, max_stages);
        DG_HOST_ASSERT(num_stages >= 2);
        config.num_stages = num_stages;
        config.smem_size = smem_fixed + num_stages * smem_per_stage;
        if (deep_jit::get_env<int>("DG_PRINT_CONFIGS"))
            std::cout << "SM107 MegaMoE: stages=" << config.num_stages << ", smem=" << config.smem_size << std::endl;
    }

    // Make tensormap
    constexpr int kGranK = 32;
    const int sf_smem_outer_dim = config.block_k / (kGranK * 4);
    const auto tensor_map_l1_acts = make_tma_2d_desc(l1_acts,
                                                     hidden, config.num_ring_tokens,
                                                     config.block_k, config.load_block_m,
                                                     static_cast<int>(l1_acts.stride(-2)),
                                                     config.swizzle_acts_mode);
    const auto tensor_map_l1_acts_sf = make_tma_sf_desc(cute::UMMA::Major::MN, l1_acts_sf,
                                                        config.num_sf_ring_tokens, hidden,
                                                        config.sf_block_m, kGranK,
                                                        1, 0, 0, false,
                                                        sf_smem_outer_dim);
    // NOTES: the SM107 K=64 UMMA reads FP4 operands densely packed; keep routed FP4 weights packed in smem (64B swizzle)
    const int umma_k_ = deep_jit::get_env<int>("DG_MEGA_MOE_SM107_UMMA_K", 64);
    const bool fp4_packed_smem = umma_k_ == 64 and l1_weights.scalar_type() == kPackedFP4;
    const int routed_swizzle_weights_mode = fp4_packed_smem ? 64 : config.swizzle_weights_mode;
    // Localized weights (per-die path): `(2, [E,] N / 2, K)` -> 3D TMA `(K, [E x] N / 2, 2)`
    const bool localized_weights = l1_weights.dim() == 4;
    const bool localized_shared_weights = num_shared_experts > 0 and shared_l1_weights.dim() == 3;
    const auto make_weights_desc = [&](const torch::Tensor& w, const bool& localized, const int& shape_k, const int& num_rows,
                                       const int& block_k, const int& swizzle, const bool& packed) {
        if (not localized)
            return make_tma_2d_desc(w, shape_k, num_rows, block_k, config.load_block_n,
                                    static_cast<int>(w.stride(-2)), swizzle, 0, false, not packed);
        DG_HOST_ASSERT(w.size(0) == 2 and num_rows % 2 == 0 and w.stride(0) <= std::numeric_limits<int>::max());
        return make_tma_3d_desc(w, shape_k, num_rows / 2, 2, block_k, config.load_block_n, 1,
                                static_cast<int>(w.stride(-2)), static_cast<int>(w.stride(0)), swizzle, 0, false, not packed);
    };
    const auto tensor_map_l1_weights = make_weights_desc(l1_weights, localized_weights, hidden, num_experts_per_rank * intermediate_hidden * 2,
                                                         config.block_k, routed_swizzle_weights_mode, fp4_packed_smem);
    const auto tensor_map_l1_weights_sf = make_tma_sf_desc(cute::UMMA::Major::MN, l1_weights_sf,
                                                           intermediate_hidden * 2, hidden,
                                                           config.block_n, kGranK,
                                                           num_experts_per_rank, 0, 0, false,
                                                           sf_smem_outer_dim);
    // NOTES: L1 output and L2 activations are essentially the same tensor.
    // Post-SwiGLU output has half the N width (`BLOCK_N / 2` per input tile),
    // so the swizzle mode is also halved (128 -> 64).
    const auto tensor_map_l1_output = make_tma_2d_desc(l2_acts,
                                                       intermediate_hidden, config.num_ring_tokens,
                                                       config.block_n / 2, config.store_block_m,
                                                       static_cast<int>(l2_acts.stride(-2)),
                                                       config.swizzle_acts_mode / 2);
    const auto tensor_map_l2_acts = make_tma_2d_desc(l2_acts,
                                                     intermediate_hidden, config.num_ring_tokens,
                                                     config.block_k, config.load_block_m,
                                                     static_cast<int>(l2_acts.stride(-2)),
                                                     config.swizzle_acts_mode);
    const auto tensor_map_l2_acts_sf = make_tma_sf_desc(cute::UMMA::Major::MN, l2_acts_sf,
                                                        config.num_sf_ring_tokens, intermediate_hidden,
                                                        config.sf_block_m, kGranK,
                                                        1, 0, 0, false,
                                                        sf_smem_outer_dim);
    const auto tensor_map_l2_weights = make_weights_desc(l2_weights, localized_weights, intermediate_hidden, num_experts_per_rank * hidden,
                                                         config.block_k, routed_swizzle_weights_mode, fp4_packed_smem);
    const auto tensor_map_l2_weights_sf = make_tma_sf_desc(cute::UMMA::Major::MN, l2_weights_sf,
                                                           hidden, intermediate_hidden,
                                                           config.block_n, kGranK,
                                                           num_experts_per_rank, 0, 0, false,
                                                        sf_smem_outer_dim);

    // Half-K shared stages (packed FP4 mode): 64-K boxes with 64B swizzle for the shared tokens and FP8 weights
    const int shared_block_k = fp4_packed_smem ? config.block_k / 2 : config.block_k;
    const int shared_swizzle = fp4_packed_smem ? std::min(shared_block_k, 128) : config.swizzle_acts_mode;
    const auto tensor_map_shared_l1_acts = num_shared_experts > 0 ? make_tma_2d_desc(
        shared_l1_acts,
        hidden, num_max_tokens_per_rank,
        shared_block_k, config.load_block_m,
        static_cast<int>(shared_l1_acts.stride(-2)),
        shared_swizzle) : tensor_map_l1_acts;
    const auto tensor_map_shared_l1_acts_sf = num_shared_experts > 0 ? make_tma_sf_desc(
        cute::UMMA::Major::MN, shared_l1_acts_sf,
        static_cast<int>(shared_l1_acts_sf.size(0)), hidden,
        config.sf_block_m, kGranK,
        1, 0, 0, false,
        sf_smem_outer_dim) : tensor_map_l1_acts_sf;
    const auto tensor_map_shared_l1_weights = num_shared_experts > 0 ?
        make_weights_desc(shared_l1_weights, localized_shared_weights, hidden, shared_intermediate_hidden * 2, shared_block_k, shared_swizzle, false) :
        tensor_map_l1_weights;
    const auto tensor_map_shared_l1_weights_sf = num_shared_experts > 0 ? make_tma_sf_desc(
        cute::UMMA::Major::MN, shared_l1_weights_sf,
        shared_intermediate_hidden * 2, hidden,
        config.block_n, kGranK,
        1, 0, 0, false,
        sf_smem_outer_dim) : tensor_map_l1_weights_sf;
    const auto tensor_map_shared_l1_output = num_shared_experts > 0 ? make_tma_2d_desc(
        shared_l2_acts,
        shared_intermediate_hidden, num_max_tokens_per_rank,
        config.block_n / 2, config.store_block_m,
        static_cast<int>(shared_l2_acts.stride(-2)),
        config.swizzle_acts_mode / 2) : tensor_map_l1_output;
    const auto tensor_map_shared_l2_acts = num_shared_experts > 0 ? make_tma_2d_desc(
        shared_l2_acts,
        shared_intermediate_hidden, num_max_tokens_per_rank,
        shared_block_k, config.load_block_m,
        static_cast<int>(shared_l2_acts.stride(-2)),
        shared_swizzle) : tensor_map_l2_acts;
    const auto tensor_map_shared_l2_acts_sf = num_shared_experts > 0 ? make_tma_sf_desc(
        cute::UMMA::Major::MN, shared_l2_acts_sf,
        static_cast<int>(shared_l2_acts_sf.size(0)), shared_intermediate_hidden,
        config.sf_block_m, kGranK,
        1, 0, 0, false,
        sf_smem_outer_dim) : tensor_map_l2_acts_sf;
    const auto tensor_map_shared_l2_weights = num_shared_experts > 0 ?
        make_weights_desc(shared_l2_weights, localized_shared_weights, shared_intermediate_hidden, hidden, shared_block_k, shared_swizzle, false) :
        tensor_map_l2_weights;
    const auto tensor_map_shared_l2_weights_sf = num_shared_experts > 0 ? make_tma_sf_desc(
        cute::UMMA::Major::MN, shared_l2_weights_sf,
        hidden, shared_intermediate_hidden,
        config.block_n, kGranK,
        1, 0, 0, false,
        sf_smem_outer_dim) : tensor_map_l2_weights_sf;

    // Stats can be optional
    int* cumulative_local_expert_recv_stats_ptr = nullptr;
    if (cumulative_local_expert_recv_stats.has_value())
        cumulative_local_expert_recv_stats_ptr = cumulative_local_expert_recv_stats->data_ptr<int>();

    const auto num_sms = runtime->get_num_sms();

    // SM107 options
    const int umma_k = deep_jit::get_env<int>("DG_MEGA_MOE_SM107_UMMA_K", 64);
    DG_HOST_ASSERT(umma_k == 32 or umma_k == 64);
    // Debug trace: device pointer to `num_sms * 64` uint64 slots (zeroed by the caller), see the kernel for slots
    const auto trace_ptr = deep_jit::get_env<unsigned long long>("DG_MEGA_MOE_SM107_TRACE_PTR", 0ull);
    // Per-die execution (`DG_MEGA_MOE_SM107_PERDIE=1`, Rubin 2 locality domains): one task queue per domain, each owning
    // the N half of every weight matrix (homed there if the weights are localized), SMs mapped by a latency probe
    const bool clusters_split = (intermediate_hidden * 2) % (config.block_n * 2 * 2) == 0 and hidden % (config.block_n * 2 * 2) == 0;
    const int num_locality_domains = deep_jit::get_env<int>("DG_MEGA_MOE_SM107_PERDIE", 0) != 0 and clusters_split and
        sm107_locality::get_num_locality_domains() == 2 ? 2 : 1;
    DG_HOST_ASSERT(not (localized_weights or localized_shared_weights) or clusters_split);
    const auto sm_locality_domains = num_locality_domains > 1 ? sm107_locality::get_balanced_sm_locality_domains() : torch::Tensor();
    DG_HOST_ASSERT(num_locality_domains == 1 or sm_locality_domains.numel() == num_sms);
    DG_HOST_ASSERT(not localized_weights or deep_jit::get_env<int>("DG_MEGA_MOE_SM107_PREFETCH", 0) == 0);
    deep_jit::cuda::CompilerOptions compiler_options;
    const auto arch = deep_jit::get_env<std::string>("DG_MEGA_MOE_SM107_ARCH", "");
    if (not arch.empty())
        compiler_options.arch = arch;
    // NOTES: the vendored CUTLASS (4.2.1) predates SM107; arch-specific SM107 targets need CUTLASS >= 4.8 headers,
    // shipped in `include/sm107_cutlass` and searched before the default include directory (only for this kernel)
    if (arch.starts_with("107")) {
        const auto sm107_cutlass_dir = jit->config.include_dirs.at(0) / "sm107_cutlass";
        DG_HOST_ASSERT(std::filesystem::exists(sm107_cutlass_dir / "cutlass" / "version.h"));
        compiler_options.extra_nvcc_flags.push_back("--include-path=" + sm107_cutlass_dir.string());
    }

    // Compile
    const auto kernel = jit->compile("sm107_fp8_fp4_mega_moe", std::format(R"(
#include <deep_gemm/impls/sm107_fp8_fp4_mega_moe.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm107_fp8_fp4_mega_moe_impl<
        {},
        {}, {},
        {}, {},
        {}, {}, {},
        {},
        {}, {},
        {},
        {},
        {},
        {},
        {},
        {}, {}, {},
        {}, {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {}, {}, {}
    >);
}};
)", num_max_tokens_per_rank,
        hidden, intermediate_hidden,
        num_experts, num_shared_experts,
        num_topk,
        config.block_m, config.block_n, config.block_k,
        config.store_block_m,
        config.sf_block_m, config.sf_block_n,
        config.num_ring_tokens,
        config.num_sf_ring_tokens,
        config.num_stages,
        config.num_bytes_per_pull,
        config.num_dispatch_threads, config.num_non_epilogue_threads, config.num_epilogue_threads,
        num_sms, num_ranks,
        to_string(activation_clamp),
        to_string(activation_alpha),
        to_string(activation_beta),
        fast_math ? "true" : "false",
        to_string(l1_weights.scalar_type(), not fp4_packed_smem),
        umma_k, trace_ptr != 0 ? "true" : "false",
        deep_jit::get_env<int>("DG_MEGA_MOE_SM107_DEBUG", 0),
        deep_jit::get_env<int>("DG_MEGA_MOE_SM107_EPI_PF", 0),
        deep_jit::get_env<int>("DG_MEGA_MOE_SM107_DISPATCH_PIPE", 3),
        deep_jit::get_env<int>("DG_MEGA_MOE_SM107_PREFETCH", 0),
        num_locality_domains, localized_weights ? "true" : "false", localized_shared_weights ? "true" : "false"),
        compiler_options);

    // Launch
    // NOTES: shared memory above the portable opt-in limit ("oversized", Rubin) is launched without DeepJIT's
    // `CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES` (which rejects such sizes), see `sm107_launch_oversized`
    const deep_jit::cuda::LaunchOptions launch_options = {
        .num_smem_bytes = config.smem_size,
        .grid_dim = dim3(num_sms, 1, 1),
        .block_dim = dim3(config.num_dispatch_threads + config.num_non_epilogue_threads + config.num_epilogue_threads, 1, 1),
        .cluster_dim = dim3(2, 1, 1),
    };
    const bool oversized_smem = config.smem_size > SM100ArchSpec::smem_capacity;
    const auto launch_fn = [&](const auto&... args) {
        if (oversized_smem)
            sm107_launch_oversized(*kernel, jit->default_launch_options.override_with(launch_options), args...);
        else
            jit->launch(kernel, launch_options, args...);
    };
    launch_fn(
        y.data_ptr(),
        cumulative_local_expert_recv_stats_ptr,
        num_tokens,
        layout::SymBuffer<>(sym_buffer_ptrs, rank_idx),
        tensor_map_l1_acts,
        tensor_map_l1_acts_sf,
        tensor_map_l1_weights,
        tensor_map_l1_weights_sf,
        tensor_map_l1_output,
        tensor_map_l2_acts,
        tensor_map_l2_acts_sf,
        tensor_map_l2_weights,
        tensor_map_l2_weights_sf,
        tensor_map_shared_l1_acts,
        tensor_map_shared_l1_acts_sf,
        tensor_map_shared_l1_weights,
        tensor_map_shared_l1_weights_sf,
        tensor_map_shared_l1_output,
        tensor_map_shared_l2_acts,
        tensor_map_shared_l2_acts_sf,
        tensor_map_shared_l2_weights,
        tensor_map_shared_l2_weights_sf,
        reinterpret_cast<uint64_t*>(trace_ptr),
        num_locality_domains > 1 ? sm_locality_domains.data_ptr<uint8_t>() : nullptr);
}

} // namespace deep_gemm
