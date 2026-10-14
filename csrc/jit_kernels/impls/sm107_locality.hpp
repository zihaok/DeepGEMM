#pragma once

// Rubin (VR200) locality domains for the opt-in SM107 MegaMoE per-die path (`DG_MEGA_MOE_SM107_PERDIE=1`).
// Port of upstream DeepGEMM 26/09/30 (`csrc/runtime/locality_domain.hpp`, `csrc/apis/locality_domain.hpp`) with one
// difference: upstream allocates die-local memory through an MPS server started with `-mlopart` (IPC of exported
// handles), "prior to CUDA 13.4". CUDA 13.4 (driver >= 620) allocates it natively:
// `cuMemCreate` on `{CU_MEM_LOCATION_TYPE_DEVICE_LOCALITY_DOMAIN, {device, domain}}`, so no MPS / helper process is used.
//   - `get_num_locality_domains()`: `CU_DEVICE_ATTRIBUTE_LOCALITY_DOMAIN_COUNT` (1 if unsupported)
//   - `empty_per_domain(sizes, dtype)`: `(num_domains, *sizes)` tensor, slice `d` homed in domain `d`
//   - `get_balanced_sm_locality_domains()`: per-SM domain (uint8, device), probed by pointer chasing into each domain's
//     memory (upstream's method), then balanced to the same number of TPCs per domain

#include <cuda.h>
#include <format>
#include <map>
#include <mutex>
#include <vector>

#include <c10/cuda/CUDAFunctions.h>
#include <c10/util/accumulate.h>
#include <c10/util/strides.h>
#include <torch/python.h>
#include <deep_jit/backend/cuda/driver.hpp>

#include "../../runtime/runtime.hpp"
#include "../../utils/exception.hpp"
#include "../../utils/math.hpp"
#include "../heuristics/sm100.hpp"

namespace deep_gemm::sm107_locality {

// Driver entry points, resolved lazily from libcuda like DeepJIT's own (the `_C` module does not link libcuda)
DJ_DECL_LAZY_DL_FUNCTION(deep_jit::cuda::driver::get_cuda_handle, cuDeviceGet);
DJ_DECL_LAZY_DL_FUNCTION(deep_jit::cuda::driver::get_cuda_handle, cuDeviceGetAttribute);
DJ_DECL_LAZY_DL_FUNCTION(deep_jit::cuda::driver::get_cuda_handle, cuMemGetAllocationGranularity);
DJ_DECL_LAZY_DL_FUNCTION(deep_jit::cuda::driver::get_cuda_handle, cuMemCreate);
DJ_DECL_LAZY_DL_FUNCTION(deep_jit::cuda::driver::get_cuda_handle, cuMemAddressReserve);
DJ_DECL_LAZY_DL_FUNCTION(deep_jit::cuda::driver::get_cuda_handle, cuMemAddressFree);
DJ_DECL_LAZY_DL_FUNCTION(deep_jit::cuda::driver::get_cuda_handle, cuMemMap);
DJ_DECL_LAZY_DL_FUNCTION(deep_jit::cuda::driver::get_cuda_handle, cuMemUnmap);
DJ_DECL_LAZY_DL_FUNCTION(deep_jit::cuda::driver::get_cuda_handle, cuMemSetAccess);
DJ_DECL_LAZY_DL_FUNCTION(deep_jit::cuda::driver::get_cuda_handle, cuMemRelease);
DJ_DECL_LAZY_DL_FUNCTION(deep_jit::cuda::driver::get_cuda_handle, cuPointerGetAttribute);

static CUdevice get_device() {
    CUdevice device;
    DJ_CUDA_DRIVER_CHECK(lazy_cuDeviceGet(&device, static_cast<int>(c10::cuda::current_device())));
    return device;
}

static int get_num_locality_domains() {
#if CUDA_VERSION >= 13040
    int num_domains = 1;
    if (lazy_cuDeviceGetAttribute(&num_domains, CU_DEVICE_ATTRIBUTE_LOCALITY_DOMAIN_COUNT, get_device()) != CUDA_SUCCESS)
        return 1;
    return std::max(num_domains, 1);
#else
    return 1;
#endif
}

// Locality domain ordinal of the allocation holding `ptr` (-1: not localized)
static int get_locality_domain_of(const int64_t& ptr) {
#if CUDA_VERSION >= 13040
    int ordinal = -1;
    DJ_CUDA_DRIVER_CHECK(lazy_cuPointerGetAttribute(&ordinal, CU_POINTER_ATTRIBUTE_LOCALITY_DOMAIN_ORDINAL,
                                                    static_cast<CUdeviceptr>(ptr)));
    return ordinal;
#else
    return -1;
#endif
}

// Live localized reservations: base address -> (reserved bytes, (mapped address, bytes, handle) per domain)
struct Mapping {
    CUdeviceptr address;
    size_t num_bytes;
    CUmemGenericAllocationHandle handle;
};
struct Reservation {
    size_t num_bytes;
    std::vector<Mapping> mappings;
};
// NOTES: heap objects that are never destroyed, so that late tensor deleters (interpreter exit) stay valid
inline std::mutex& reservations_mutex = *new std::mutex();
inline std::map<CUdeviceptr, Reservation>& reservations = *new std::map<CUdeviceptr, Reservation>();

static void release(void* ptr) {
    Reservation reservation;
    {
        std::lock_guard<std::mutex> lock(reservations_mutex);
        const auto it = reservations.find(reinterpret_cast<CUdeviceptr>(ptr));
        if (it == reservations.end())
            return;
        reservation = std::move(it->second);
        reservations.erase(it);
    }
    // The tensor may still be read by in-flight kernels
    if (cudaDeviceSynchronize() != cudaSuccess)
        return;
    for (const auto& mapping: reservation.mappings) {
        lazy_cuMemUnmap(mapping.address, mapping.num_bytes);
        lazy_cuMemRelease(mapping.handle);
    }
    lazy_cuMemAddressFree(reinterpret_cast<CUdeviceptr>(ptr), reservation.num_bytes);
}

// `torch.empty((num_domains, *sizes), dtype=dtype)` with slice `d` homed in locality domain `d`
// (slices are padded to the allocation granularity, so `stride(0)` may exceed `prod(sizes)`)
static torch::Tensor empty_per_domain(const std::vector<int64_t>& sizes, const torch::ScalarType& dtype) {
#if CUDA_VERSION >= 13040
    const int num_domains = get_num_locality_domains();
    DG_HOST_ASSERT(num_domains > 1 and "this GPU has a single locality domain");
    const auto device = get_device();

    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE_LOCALITY_DOMAIN;
    prop.location.localized.deviceId = static_cast<unsigned char>(device);
    prop.location.localized.localityDomainId = 0;
    size_t granularity = 0;
    DJ_CUDA_DRIVER_CHECK(lazy_cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));

    const auto elem_size = static_cast<int64_t>(c10::elementSize(dtype));
    const auto num_slice_bytes = align<int64_t>(c10::multiply_integers(sizes) * elem_size, static_cast<int64_t>(granularity));
    DG_HOST_ASSERT(num_slice_bytes % elem_size == 0);

    CUdeviceptr base = 0;
    const size_t num_bytes = static_cast<size_t>(num_slice_bytes) * num_domains;
    DJ_CUDA_DRIVER_CHECK(lazy_cuMemAddressReserve(&base, num_bytes, granularity, 0, 0));
    Reservation reservation = {num_bytes, {}};
    for (int d = 0; d < num_domains; ++ d) {
        prop.location.localized.localityDomainId = static_cast<unsigned char>(d);
        CUmemGenericAllocationHandle handle;
        DJ_CUDA_DRIVER_CHECK(lazy_cuMemCreate(&handle, static_cast<size_t>(num_slice_bytes), &prop, 0));
        const auto address = base + static_cast<CUdeviceptr>(d) * num_slice_bytes;
        DJ_CUDA_DRIVER_CHECK(lazy_cuMemMap(address, static_cast<size_t>(num_slice_bytes), 0, handle, 0));
        reservation.mappings.push_back({address, static_cast<size_t>(num_slice_bytes), handle});
    }
    CUmemAccessDesc access = {};
    access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access.location.id = static_cast<int>(device);
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    DJ_CUDA_DRIVER_CHECK(lazy_cuMemSetAccess(base, num_bytes, &access, 1));
    {
        std::lock_guard<std::mutex> lock(reservations_mutex);
        reservations.emplace(base, std::move(reservation));
    }

    std::vector<int64_t> full_sizes = {num_domains}, strides = {num_slice_bytes / elem_size};
    full_sizes.insert(full_sizes.end(), sizes.begin(), sizes.end());
    const auto slice_strides = c10::contiguous_strides(sizes);
    strides.insert(strides.end(), slice_strides.begin(), slice_strides.end());
    return torch::from_blob(reinterpret_cast<void*>(base), full_sizes, strides, release,
                            torch::dtype(dtype).device(torch::kCUDA, c10::cuda::current_device()));
#else
    DG_HOST_UNREACHABLE("locality domains need CUDA >= 13.4 headers");
#endif
}

// Whether `t` is `(num_domains, ...)` with contiguous slices and slice `d` homed in domain `d`
static bool is_localized(const torch::Tensor& t) {
    const int num_domains = get_num_locality_domains();
    if (num_domains < 2 or t.dim() < 2 or t.size(0) != num_domains)
        return false;
    const auto slice_sizes = t.sizes().slice(1);
    if (t.strides().slice(1) != c10::IntArrayRef(c10::contiguous_strides(slice_sizes)))
        return false;
    for (int d = 0; d < num_domains; ++ d) {
        const auto ptr = reinterpret_cast<int64_t>(static_cast<const char*>(t.data_ptr()) + d * t.stride(0) * t.element_size());
        if (get_locality_domain_of(ptr) != d)
            return false;
    }
    return true;
}

/// Locality domain of every SM (upstream's latency probe)

static constexpr int kNumSMsPerTPC = 2;
static constexpr int kProbeChunkBytes = 4096;
static constexpr int kProbeLineBytes = 128;
static constexpr int kProbeHops = kProbeChunkBytes / kProbeLineBytes;
static constexpr int kProbeChunksPerSM = 8;
// Coprime with the line count, so that the chain visits every line
static constexpr int kChainLineStride = 7;
static constexpr float kMinFarNearRatio = 1.25f;
static constexpr int kNumMaxProbeAttempts = 5;

static void launch_probe(const torch::Tensor& buf, const torch::Tensor& latency) {
    const auto kernel = jit->compile("sm107_locality_probe_chase", std::format(R"(
#include <deep_gemm/impls/sm107_locality_probe.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm107_locality_probe_chase<{}, {}, {}>);
}};
)", kProbeHops, kProbeChunksPerSM, kProbeChunkBytes));
    // NOTES: the maximum shared memory keeps one block per SM, so every SM chases its own chunks
    jit->launch(kernel, {
            .num_smem_bytes = SM100ArchSpec::smem_capacity,
            .grid_dim = dim3(jit->device.get_num_sms(), 1, 1),
            .block_dim = dim3(32, 1, 1),
        },
        buf.data_ptr(), latency.data_ptr());
}

// `(num_domains, num_sms)` median cycles per hop
static torch::Tensor probe(const torch::Tensor& buf) {
    const auto options = torch::dtype(torch::kInt).device(buf.device());
    auto chain = torch::zeros({kProbeHops, kProbeLineBytes / 4}, options);
    chain.select(1, 0).copy_((torch::arange(kProbeHops, options) + kChainLineStride) % kProbeHops * (kProbeLineBytes / 4));
    const auto num_sms = buf.size(1);
    const auto latency = torch::zeros({buf.size(0), num_sms, kProbeChunksPerSM}, buf.options().dtype(torch::kInt16));
    for (int64_t d = 0; d < buf.size(0); ++ d)
        buf[d].view(torch::kInt).view({-1, kProbeChunkBytes / 4}).copy_(chain.view(-1).unsqueeze(0));
    // Flush L2 so that every hop is a fill from memory
    torch::zeros({256 << 20}, torch::dtype(torch::kUInt8).device(buf.device()));
    for (int64_t d = 0; d < buf.size(0); ++ d)
        launch_probe(buf[d], latency[d]);
    DG_HOST_ASSERT(latency.gt(0).all().item<bool>());
    return std::get<0>(latency.to(torch::kFloat).median(-1));
}

static torch::Tensor try_probe(const torch::Tensor& buf) {
    const auto [latency, order] = probe(buf).sort(0);
    const auto sm_domain = order[0].to(torch::kUInt8);
    const auto tpcs = sm_domain.view({-1, kNumSMsPerTPC});
    const auto clear = latency[1].ge(latency[0] * kMinFarNearRatio).all().item<bool>();
    const auto consistent = torch::equal(tpcs, tpcs.select(1, 0).unsqueeze(1).expand_as(tpcs));
    return clear and consistent ? sm_domain : torch::Tensor();
}

// Without a stable probe, an arbitrary SM mapping is still functionally correct
static torch::Tensor get_even_sm_locality_domains(const int& num_domains) {
    const auto tpc_idx = torch::arange(jit->device.get_num_sms() / kNumSMsPerTPC, torch::dtype(torch::kInt).device(torch::kCUDA));
    return (tpc_idx % num_domains).to(torch::kUInt8).repeat_interleave(kNumSMsPerTPC);
}

static torch::Tensor probe_sm_locality_domains(const int& num_domains) {
    const auto num_sms = jit->device.get_num_sms();
    const auto buf = empty_per_domain({num_sms, kProbeChunksPerSM, kProbeChunkBytes}, torch::kUInt8);
    for (int attempt = 0; attempt < kNumMaxProbeAttempts; ++ attempt) {
        if (const auto sm_domain = try_probe(buf); sm_domain.defined())
            return sm_domain;
    }
    pybind11::module_::import("warnings").attr("warn")("SM107 MegaMoE: the SM locality-domain probe found no stable mapping; using an even one");
    return get_even_sm_locality_domains(num_domains);
}

// Give both domains the same number of TPCs (the surplus TPCs of the larger domain move to the smaller one)
static torch::Tensor balance_sm_locality_domains(const torch::Tensor& sm_domain) {
    const auto tpcs = sm_domain.view({-1, kNumSMsPerTPC});
    const auto tpc_domain = tpcs.select(1, 0);
    DG_HOST_ASSERT(tpcs.size(0) % 2 == 0);
    const int64_t num_tpcs = tpcs.size(0), num_domain_1 = tpc_domain.sum().item<int64_t>();
    const uint8_t larger = num_domain_1 * 2 > num_tpcs;
    const auto surplus = (tpc_domain == larger).nonzero().flatten().slice(0, num_tpcs / 2);
    const auto balanced = sm_domain.clone();
    balanced.view({-1, kNumSMsPerTPC}).index_fill_(0, surplus, larger ^ 1);
    return balanced;
}

// Cached per process (one device per process in vLLM); never destroyed (see above)
inline torch::Tensor& sm_locality_domains = *new torch::Tensor();
inline torch::Tensor& balanced_sm_locality_domains = *new torch::Tensor();

static torch::Tensor get_sm_locality_domains() {
    if (not sm_locality_domains.defined()) {
        const int num_domains = get_num_locality_domains();
        DG_HOST_ASSERT(num_domains == 2 and "the SM107 per-die path supports exactly 2 locality domains");
        sm_locality_domains = probe_sm_locality_domains(num_domains);
    }
    return sm_locality_domains;
}

static torch::Tensor get_balanced_sm_locality_domains() {
    if (not balanced_sm_locality_domains.defined())
        balanced_sm_locality_domains = balance_sm_locality_domains(get_sm_locality_domains());
    return balanced_sm_locality_domains;
}

}  // namespace deep_gemm::sm107_locality
