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

#include <numeric>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include "../../utils/torch_compat.hpp"
#include "../../apis/locality_domain.hpp"
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
    int ordinal = 0;
    DG_CUDA_RUNTIME_CHECK(cudaGetDevice(&ordinal));
    CUdevice device;
    DJ_CUDA_DRIVER_CHECK(lazy_cuDeviceGet(&device, ordinal));
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
static torch::stable::Tensor empty_per_domain(const std::vector<int64_t>& sizes, const torch::headeronly::ScalarType& dtype) {
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

    const auto elem_size = static_cast<int64_t>(torch_compat::element_size(dtype));
    const auto num_slice_bytes = align<int64_t>(locality_domain::numel(sizes) * elem_size, static_cast<int64_t>(granularity));
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
    const auto slice_strides = locality_domain::contiguous_strides(sizes);
    strides.insert(strides.end(), slice_strides.begin(), slice_strides.end());
    // Stable ABI 2.10 has no from_blob deleter. Like the base allocator,
    // reservations are process-owned so views can outlive the original tensor.
    return torch::stable::from_blob(reinterpret_cast<void*>(base), full_sizes, strides,
                                    locality_domain::current_cuda_device(), dtype);
#else
    DG_HOST_UNREACHABLE("locality domains need CUDA >= 13.4 headers");
#endif
}

// Whether `t` is `(num_domains, ...)` with contiguous slices and slice `d` homed in domain `d`
static bool is_localized(const torch::stable::Tensor& t) {
    const int num_domains = get_num_locality_domains();
    if (num_domains < 2 or t.dim() < 2 or t.size(0) != num_domains)
        return false;
    std::vector<int64_t> slice_sizes;
    for (int i = 1; i < t.dim(); ++i)
        slice_sizes.push_back(t.size(i));
    const auto strides = locality_domain::contiguous_strides(slice_sizes);
    for (int i = 1; i < t.dim(); ++i)
        if (t.stride(i) != strides[i - 1])
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

static void launch_probe(const torch::stable::Tensor& buf, const torch::stable::Tensor& latency) {
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

inline torch::stable::Tensor& balanced_sm_locality_domains = *new torch::stable::Tensor();

static torch::stable::Tensor get_balanced_sm_locality_domains() {
    DG_HOST_ASSERT(balanced_sm_locality_domains.defined() and
                   "Warm deep_gemm.mega.sm107_locality before the first per-die kernel");
    return balanced_sm_locality_domains;
}

static void set_balanced_sm_locality_domains(const torch::stable::Tensor& domains) {
    DG_HOST_ASSERT(domains.scalar_type() == torch::headeronly::ScalarType::Byte);
    DG_HOST_ASSERT(domains.is_cuda() and domains.is_contiguous());
    DG_HOST_ASSERT(domains.numel() == jit->device.get_num_sms());
    balanced_sm_locality_domains = domains;
}

static int64_t num_domains_op() { return get_num_locality_domains(); }

} // namespace deep_gemm::sm107_locality

STABLE_TORCH_LIBRARY_FRAGMENT(deep_gemm, m) {
    m.def("sm107_get_num_locality_domains() -> int");
    m.def("sm107_empty_per_domain(int[] sizes, ScalarType dtype) -> Tensor");
    m.def("sm107_is_localized(Tensor t) -> bool");
    m.def("sm107_probe(Tensor buf, Tensor(a!) latency) -> ()");
    m.def("sm107_set_balanced_sm_locality_domains(Tensor domains) -> ()");
}
STABLE_TORCH_LIBRARY_IMPL(deep_gemm, CompositeExplicitAutograd, m) {
    m.impl("sm107_get_num_locality_domains", TORCH_BOX(&deep_gemm::sm107_locality::num_domains_op));
    m.impl("sm107_empty_per_domain", TORCH_BOX(&deep_gemm::sm107_locality::empty_per_domain));
}
STABLE_TORCH_LIBRARY_IMPL(deep_gemm, CUDA, m) {
    m.impl("sm107_is_localized", TORCH_BOX(&deep_gemm::sm107_locality::is_localized));
    m.impl("sm107_probe", TORCH_BOX(&deep_gemm::sm107_locality::launch_probe));
    m.impl("sm107_set_balanced_sm_locality_domains", TORCH_BOX(&deep_gemm::sm107_locality::set_balanced_sm_locality_domains));
}
