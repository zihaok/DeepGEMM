#pragma once

// Rubin (SM107) MegaMoE scheduler with locality-domain (per-die) task queues.
// Port of upstream DeepGEMM 26/09/30 (`deep_gemm/scheduler/mega_moe.cuh`, commit c7849af) as a separate struct, so that
// the production `MegaMoEScheduler` stays untouched:
//   - each locality domain has its own L1 / L2 / shared-L1 / shared-L2 task counter; domain `d`'s counter lives at
//     `<domain 0 counter> + d * kDomainCounterStride` (`layout::MegaMoESignals::task_count_domain1`)
//   - domain `d` owns the N clusters `[d * kNumClusters / D, (d + 1) * kNumClusters / D)`, i.e. the N half whose weights are
//     homed in its memory, and steals from the other queue only with >= 2 waves of tasks
//   - with `kNumLocalityDomains == 1` the behaviour is identical to `MegaMoEScheduler`
#include <deep_gemm/scheduler/mega_moe.cuh>

namespace deep_gemm::sched {

static constexpr uint32_t kNumSM107LocalityDomains = 2;
static constexpr uint32_t kDomainCounterStride = 4;

#if defined(__CUDACC__) or defined(__CLION_IDE__)

template <uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
          uint32_t L1_SHAPE_N, uint32_t L1_SHAPE_K,
          uint32_t L2_SHAPE_N, uint32_t L2_SHAPE_K,
          uint32_t kNumExpertsPerRank,
          uint32_t kNumSMs, uint32_t kNumRanks,
          uint32_t kNumRingBlocks, uint32_t kNumLocalityDomains,
          uint32_t kNumSharedExperts = 0,
          uint32_t kNumExpertsPerLane = math::constexpr_ceil_div(kNumExpertsPerRank, 32u),
          uint32_t kNumL1BlockNs = L1_SHAPE_N / BLOCK_N,
          uint32_t kNumL2BlockNs = L2_SHAPE_N / BLOCK_N,
          uint32_t kNumL1Clusters = kNumL1BlockNs / 2,
          uint32_t kNumL2Clusters = kNumL2BlockNs / 2>
struct MegaMoESchedulerLD {
    static constexpr bool kHasShared = kNumSharedExperts > 0;
    static constexpr uint32_t SHARED_L1_SHAPE_N = L1_SHAPE_N * kNumSharedExperts;
    static constexpr uint32_t SHARED_L1_SHAPE_K = L1_SHAPE_K;
    static constexpr uint32_t SHARED_L2_SHAPE_N = L2_SHAPE_N;
    static constexpr uint32_t SHARED_L2_SHAPE_K = L2_SHAPE_K * kNumSharedExperts;
    using task_info_t = TaskInfo<kHasShared>;

    DG_STATIC_ASSERT(L1_SHAPE_N % (BLOCK_N * 2) == 0, "Invalid shape");
    DG_STATIC_ASSERT(L2_SHAPE_N % (BLOCK_N * 2) == 0, "Invalid shape");
    DG_STATIC_ASSERT(L1_SHAPE_K % BLOCK_K == 0, "Invalid shape");
    DG_STATIC_ASSERT(L2_SHAPE_K % BLOCK_K == 0, "Invalid shape");
    DG_STATIC_ASSERT(SHARED_L1_SHAPE_N % (BLOCK_N * 2) == 0, "Invalid shared shape");
    DG_STATIC_ASSERT(SHARED_L2_SHAPE_N % (BLOCK_N * 2) == 0, "Invalid shared shape");
    DG_STATIC_ASSERT(SHARED_L1_SHAPE_K % BLOCK_K == 0, "Invalid shared shape");
    DG_STATIC_ASSERT(SHARED_L2_SHAPE_K % BLOCK_K == 0, "Invalid shared shape");

    // NOTES: N block counts must be even so that 2 adjacent CTAs in a cluster
    // always land on the same m_block_idx with n_block_idx differing by 1
    DG_STATIC_ASSERT(kNumSMs % 2 == 0, "Number of SMs must be even for 2-CTA cluster");
    DG_STATIC_ASSERT(kNumRingBlocks > 0, "Invalid ring buffer config");

    // Workspace
    const layout::Workspace& workspace;

    // Scheduler configs
    static constexpr uint32_t kNumScheduleStages = 2;
    using Barrier = cutlass::arch::ClusterTransactionBarrier;
    uint32_t sched_stage_idx = 0;
    uint32_t sched_phase = 0;
    Barrier* task_info_full_barriers = nullptr;
    Barrier* task_info_empty_barriers = nullptr;
    task_info_t* task_infos = nullptr;

    // Pre-cached per-expert token counts.
    // Layout: `stored_num_tokens_per_expert[i]` holds expert (i * 32 + lane_idx)'s count.
    uint32_t stored_num_tokens_per_expert[kNumExpertsPerLane] = {};
    uint32_t num_total_m_blocks = 0;

    // Per-scheduler warmup waves; all CTA-pair schedulers together form one global wave.
    static constexpr uint32_t kNumSchedL1WavesDone = 0xffffffffu;
    uint32_t num_sched_l1_waves = 0;

    // Locality domain affinity. When a queue is drained, we try work stealing to ensure load balancing.
    // But when there are few tasks, we disable work stealing to avoid cross-die latency.
    static constexpr uint32_t kNumMinStealWaves = 2;
    DG_STATIC_ASSERT(kNumLocalityDomains == 1 or kNumLocalityDomains == kNumSM107LocalityDomains, "Invalid locality domain count");
    DG_STATIC_ASSERT(kNumL1Clusters % kNumLocalityDomains == 0 and kNumL2Clusters % kNumLocalityDomains == 0, "Each domain must take whole clusters");
    uint32_t sm_locality_domain_idx = 0;

    CUTLASS_DEVICE explicit MegaMoESchedulerLD(const layout::Workspace& workspace):
        workspace(workspace) {}

    CUTLASS_DEVICE MegaMoESchedulerLD(const layout::Workspace& workspace,
                                    Barrier* task_info_full_barriers,
                                    Barrier* task_info_empty_barriers,
                                    task_info_t* task_infos):
        workspace(workspace),
        task_info_full_barriers(task_info_full_barriers),
        task_info_empty_barriers(task_info_empty_barriers),
        task_infos(task_infos) {}

    CUTLASS_DEVICE void advance_sched_pipeline() {
        DG_STATIC_ASSERT(kNumScheduleStages == 2, "Invalid stages");
        sched_stage_idx ^= 1;
        sched_phase ^= sched_stage_idx == 0;
    }

    CUTLASS_DEVICE bool get_next_task(task_info_t& task_info) {
        task_info_full_barriers[sched_stage_idx].wait(sched_phase);
        task_info = task_infos[sched_stage_idx];
        advance_sched_pipeline();
        return task_info.is_valid();
    }

    CUTLASS_DEVICE void release_task_info() const {
        // Complete metadata reads before the scheduler can overwrite this slot.
        ptx::fence_acq_rel_cta();
        task_info_empty_barriers[sched_stage_idx ^ 1].arrive(0u);
    }

    CUTLASS_DEVICE uint32_t get_num_tokens(const uint32_t& expert_idx) const {
        uint32_t valid_value = 0;
        #pragma unroll
        for (uint32_t i = 0; i < kNumExpertsPerLane; ++ i) {
            valid_value = (expert_idx == i * 32 + ptx::get_lane_idx()) ?
                stored_num_tokens_per_expert[i] : valid_value;
        }
        return ptx::exchange(valid_value, expert_idx % 32);
    }

    // Get pool block offset for a given expert index from a per-lane token count array.
    CUTLASS_DEVICE uint32_t get_pool_block_offset(const uint32_t& expert_idx) const {
        uint32_t num_blocks = 0;
        #pragma unroll
        for (uint32_t i = 0; i < kNumExpertsPerLane; ++ i) {
            if (i * 32 + ptx::get_lane_idx() < expert_idx)
                num_blocks += math::ceil_div(stored_num_tokens_per_expert[i], BLOCK_M);
        }
        return __reduce_add_sync(0xffffffff, num_blocks);
    }

    CUTLASS_DEVICE uint32_t get_num_total_pool_blocks() const {
        return get_pool_block_offset(kNumExpertsPerRank);
    }

    CUTLASS_DEVICE void fetch_expert_recv_count() {
        // NOTES: each lane caches experts at indices (i * 32 + lane_idx)
        #pragma unroll
        for (uint32_t i = 0; i < kNumExpertsPerLane; ++ i) {
            const auto expert_idx = i * 32 + ptx::get_lane_idx();
            uint64_t value = 0;
            if (expert_idx < kNumExpertsPerRank) {
                do {
                    value = ptx::ld_volatile(workspace.get_expert_recv_count_sum_ptr(expert_idx));
                } while (static_cast<uint32_t>(value >> 32) != kNumSMs * kNumRanks);
            }
            stored_num_tokens_per_expert[i] = static_cast<uint32_t>(value);
        }
        __syncwarp();

        num_total_m_blocks = get_num_total_pool_blocks();
        const uint32_t num_total_l1_tasks = num_total_m_blocks * kNumL1Clusters;
        const uint32_t num_total_l1_waves = math::ceil_div(num_total_l1_tasks, kNumSMs / 2);
        const uint32_t min_l1_warmup_waves = get_num_l1_warmup_waves(
            num_total_m_blocks, kNumSMs / 2, kNumL1Clusters, kNumL2Clusters);
        num_sched_l1_waves = cute::min(min_l1_warmup_waves, num_total_l1_waves);
    }

    CUTLASS_DEVICE task_info_t create_task(const BlockPhase& block_phase,
                                        const uint32_t& m_block_idx,
                                        const uint32_t& n_cluster_idx,
                                        const uint32_t& shape_n,
                                        const uint32_t& shape_k) const {
        const uint32_t lane_idx = ptx::get_lane_idx();

        task_info_t result(block_phase, 0, 0, n_cluster_idx, m_block_idx, 0, shape_n, shape_k);
        uint32_t block_offset = 0;
        #pragma unroll
        for (uint32_t i = 0; i < kNumExpertsPerLane; ++ i) {
            // Reduce whether task fall in the expert
            const uint32_t expert_idx = i * 32 + lane_idx;
            const uint32_t num_tokens = stored_num_tokens_per_expert[i];
            const uint32_t num_m_blocks = math::ceil_div(num_tokens, BLOCK_M);
            const uint32_t inclusive_num_m_blocks = math::warp_inclusive_sum(num_m_blocks, lane_idx);
            const uint32_t lane_pool_block_offset = block_offset + inclusive_num_m_blocks - num_m_blocks;
            const bool is_owner = expert_idx < kNumExpertsPerRank and
                m_block_idx >= lane_pool_block_offset and m_block_idx < lane_pool_block_offset + num_m_blocks;
            const uint32_t owner_mask = __ballot_sync(0xffffffff, is_owner);

            // Exchange the expert info
            if (owner_mask) {
                const uint32_t owner_lane_idx = static_cast<uint32_t>(__ffs(owner_mask) - 1);
                const uint32_t owner_m_block_idx = m_block_idx - lane_pool_block_offset;
                const uint32_t owner_valid_m = cute::min(num_tokens - owner_m_block_idx * BLOCK_M, BLOCK_M);
                result.local_expert_idx = ptx::exchange(expert_idx, owner_lane_idx);
                result.m_block_idx = ptx::exchange(owner_m_block_idx, owner_lane_idx);
                result.valid_m = ptx::exchange(owner_valid_m, owner_lane_idx);
            }
            block_offset += ptx::exchange(inclusive_num_m_blocks, 31);
        }
        return result;
    }

    static CUTLASS_DEVICE uint32_t get_next_local_task_idx(const uint32_t* task_count_ptr, const uint32_t& locality_domain_idx) {
        uint32_t result = 0;
        if (cute::elect_one_sync())
            result = ptx::atomic_add(task_count_ptr + locality_domain_idx * kDomainCounterStride, 1u);
        return ptx::exchange(result, 0);
    }

    template <uint32_t kNumClusters>
    static CUTLASS_DEVICE bool get_next_task_idx(const uint32_t* task_count_ptr, const uint32_t& locality_domain_idx, const uint32_t& num_m_blocks,
                                                 uint32_t& m_block_idx, uint32_t& n_cluster_idx) {
        constexpr uint32_t kNumLocalClusters = kNumClusters / kNumLocalityDomains;
        const uint32_t task_idx = get_next_local_task_idx(task_count_ptr, locality_domain_idx);
        if (task_idx >= num_m_blocks * kNumLocalClusters)
            return false;
        m_block_idx = task_idx / kNumLocalClusters;
        n_cluster_idx = locality_domain_idx * kNumLocalClusters + task_idx % kNumLocalClusters;
        return true;
    }

    template <uint32_t kNumClusters>
    CUTLASS_DEVICE bool get_next_task_idx(const uint32_t* task_count_ptr, const uint32_t& num_m_blocks,
                                          uint32_t& m_block_idx, uint32_t& n_cluster_idx) const {
        if (get_next_task_idx<kNumClusters>(task_count_ptr, sm_locality_domain_idx, num_m_blocks, m_block_idx, n_cluster_idx))
            return true;
        return kNumLocalityDomains > 1 and num_m_blocks * kNumClusters >= kNumMinStealWaves * (kNumSMs / 2) and
               get_next_task_idx<kNumClusters>(task_count_ptr, sm_locality_domain_idx ^ 1, num_m_blocks, m_block_idx, n_cluster_idx);
    }

    CUTLASS_DEVICE task_info_t get_next_task() {
        uint32_t m_block_idx, n_cluster_idx;
        while (true) {
            if (num_sched_l1_waves != kNumSchedL1WavesDone and num_sched_l1_waves) {
                // One local L1 task per scheduler; globally this is one CTA-pair wave.
                -- num_sched_l1_waves;

                // No more L1 tasks
                if (not get_next_task_idx<kNumL1Clusters>(workspace.get_l1_task_count_ptr(), num_total_m_blocks, m_block_idx, n_cluster_idx)) {
                    num_sched_l1_waves = kNumSchedL1WavesDone;
                    continue;
                }

                // Create task
                return create_task(BlockPhase::Linear1, m_block_idx, n_cluster_idx, L1_SHAPE_N, L1_SHAPE_K);
            } else {
                if (not get_next_task_idx<kNumL2Clusters>(workspace.get_l2_task_count_ptr(), num_total_m_blocks, m_block_idx, n_cluster_idx))
                    break;

                // The next task should be L1
                if (num_sched_l1_waves != kNumSchedL1WavesDone)
                    num_sched_l1_waves = 1;

                // Create task
                auto task_info = create_task(BlockPhase::Linear2, m_block_idx, n_cluster_idx, L2_SHAPE_N, L2_SHAPE_K);

                // Wait until all required L1 tasks are fetched from all scheduling queues
                const auto num_required_l1_tasks = (task_info.pool_block_idx + 1) * (kNumL1Clusters / kNumLocalityDomains);
                #pragma unroll
                for (uint32_t i = 0; i < kNumLocalityDomains; ++ i)
                    while (ptx::ld_volatile(workspace.get_l1_task_count_ptr() + i * kDomainCounterStride) < num_required_l1_tasks) {}
                return task_info;
            }
        }
        return task_info_t(BlockPhase::None, 0, 0, 0, 0, 0, 0, 0);
    }

    CUTLASS_DEVICE void publish_task(const task_info_t& task_info, const uint32_t& lane_idx) {
        if (lane_idx < 2) {
            task_info_full_barriers[sched_stage_idx].arrive_and_expect_tx(sizeof(task_info_t), lane_idx);
            ptx::st_async_cluster(
                task_infos + sched_stage_idx, task_info,
                lane_idx, task_info_full_barriers[sched_stage_idx]
            );
        }
        __syncwarp();
        advance_sched_pipeline();
    }

    template <BlockPhase kBlockPhase, uint32_t kShapeN, uint32_t kShapeK>
    CUTLASS_DEVICE void shared_mainloop(const uint32_t& num_tokens, const uint32_t& lane_idx, const uint32_t* task_count_ptr) {
        constexpr uint32_t kNumNClusters = kShapeN / BLOCK_N / 2;
        DG_STATIC_ASSERT(kNumNClusters % kNumLocalityDomains == 0, "Each domain must take whole clusters");
        const uint32_t num_m_blocks = math::ceil_div(num_tokens, BLOCK_M);
        uint32_t m_block_idx, n_cluster_idx;
        while (true) {
            task_info_empty_barriers[sched_stage_idx].wait(sched_phase ^ 1);

            // Use dynamic scheduling to reduce tailing across shared L1/L2 tile shapes.
            if (not get_next_task_idx<kNumNClusters>(task_count_ptr, num_m_blocks, m_block_idx, n_cluster_idx))
                break;
            const uint32_t valid_m = cute::min(num_tokens - m_block_idx * BLOCK_M, BLOCK_M);
            publish_task(task_info_t(kBlockPhase, 0, m_block_idx, n_cluster_idx, m_block_idx, valid_m, kShapeN, kShapeK), lane_idx);
        }
    }

    CUTLASS_DEVICE void mainloop(const uint32_t& num_tokens, const uint8_t* sm_locality_domains) {
        const auto lane_idx = ptx::get_lane_idx();
        sm_locality_domain_idx = kNumLocalityDomains == 1 ? 0u : sm_locality_domains[ptx::get_sm_idx()];

        // Shared L2 tasks go before the routed tasks if one wave fits the dispatch gap, else at the tail
        constexpr uint32_t kNumSharedL2Clusters = SHARED_L2_SHAPE_N / BLOCK_N / 2;
        const bool is_shared_l2_early = math::ceil_div(num_tokens, BLOCK_M) * kNumSharedL2Clusters <= kNumSMs / 2;
        if constexpr (kHasShared) {
            // Shared expert L1 tasks do not depend on dispatch.
            shared_mainloop<BlockPhase::SharedLinear1, SHARED_L1_SHAPE_N, SHARED_L1_SHAPE_K>(
                num_tokens, lane_idx, workspace.get_shared_l1_task_count_ptr());
            if (is_shared_l2_early)
                shared_mainloop<BlockPhase::SharedLinear2, SHARED_L2_SHAPE_N, SHARED_L2_SHAPE_K>(
                    num_tokens, lane_idx, workspace.get_shared_l2_task_count_ptr());
        }

        // Wait dispatch's results
        fetch_expert_recv_count();

        // Generate routed tasks. Keep the original wait -> claim -> publish ordering:
        // `get_next_task()` advances global task counters and must not run before the
        // schedule slot is released by consumers.
        task_info_t task_info;
        do {
            task_info_empty_barriers[sched_stage_idx].wait(sched_phase ^ 1);
            task_info = get_next_task();
            if (task_info.is_valid()) publish_task(task_info, lane_idx);
        } while (task_info.is_valid());

        if constexpr (kHasShared) {
            if (not is_shared_l2_early)
                shared_mainloop<BlockPhase::SharedLinear2, SHARED_L2_SHAPE_N, SHARED_L2_SHAPE_K>(
                    num_tokens, lane_idx, workspace.get_shared_l2_task_count_ptr());
        }

        // Sentinel.
        task_info_empty_barriers[sched_stage_idx].wait(sched_phase ^ 1);
        publish_task(task_info_t(BlockPhase::None, 0, 0, 0, 0, 0, 0, 0), lane_idx);
    }
};


#endif

} // namespace deep_gemm::sched
