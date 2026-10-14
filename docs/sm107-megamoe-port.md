# Rubin MegaMoE port onto a981d8e

Source: nvzhihanj/DeepGEMM, dsv41-rubin-megamoe-sm107, c50dea4.
Target: vllm-project/DeepGEMM, a981d8e3a58e877b7f53e63707de5834312fb710.

The source branch is preserved as source/dsv41-rubin-megamoe-sm107.
rubin-megamoe-squashed contains two source-tree-equivalent commits: the
SM107 kernel/tuning/tracing changes, then per-die scheduling/localization.
Those commits were rebased onto the target on rubin-megamoe-a981d8e.
The old SM107-to-100f JIT workaround was not replayed: the target includes it.

## Compatibility changes

- Port host launchers and native CUDA locality operators from pybind/ATen to
  TORCH_STABLE_LIBRARY and the target's stable tensor API.
- Keep the SM107 kernel's original symmetric epilogue tiling; SM100's newer
  asymmetric prefill tiling is not applied to this kernel.
- Respect the target's padded scale-factor strides when constructing TMA maps.
- Use the target's contiguous per-domain task-counter arrays, including reset.
- Preserve the target's existing weight checks and dispatch when SM107 is off.
- Move locality probing orchestration into Python; retain native CUDA
  allocations rather than introducing an MPS allocation dependency.
- Qualify five compatibility-wrapper calls to avoid ambiguous overloads with
  newer PyTorch stable operators.
- Correct the target's Python locality balancer to move only surplus TPCs.
  Its previous prefix slice converted a nearly balanced map into a 2/206
  split on the test GPU and hung both SM100 and SM107 per-die launches.

## Dependencies and usage

The CUTLASS and DeepJIT submodule pins are unchanged. Default SM107 dispatch
uses the existing compatible 100f JIT target. Native 107a compilation remains
optional and requires CUTLASS >=4.8 headers installed at
deep_gemm/include/sm107_cutlass/{cutlass,cute}; this port does not change the
global CUTLASS pin. CUDA >=13.4 headers and driver support are required for
native locality allocation and oversized shared-memory features.

Set DG_MEGA_MOE_SM107=1 to enable the kernel. For localized execution, also set
DG_MEGA_MOE_SM107_PERDIE=1, localize the transformed weight tuples with
deep_gemm.mega.sm107_locality.localize_mega_moe_weights, and call
get_balanced_sm_locality_domains before CUDA graph capture. Scale tensors
remain unlocalized. The source's tuning and debug environment knobs remain
available; debug skip flags do not produce valid numerical results.

The stable ABI 2.10 from_blob API has no storage deleter callback. Native
localized reservations are therefore process-owned, as in the target's
stable-ABI locality allocator, rather than freed when individual tensors die.
Repeated model unload/reload can retain allocations until process exit.
The locality caches retain the original one-device-per-process assumption.

## Validation boundary

The extension builds in an x86_64 CUDA 13.4 / PyTorch 2.15 development container.
The seven locality-balancer regression cases pass. Single-GPU comparisons
against a separately built, unmodified c50dea4 check output tensors and expert
counters before CUDA-graph timings. SM107 and localized SM107 match the
original outputs in completed cases. Initial SM100/per-die hangs were resolved
by the separately tested Python balancing fix.

The sibling sm107-playground directory contains the harness, commands, logs,
saved reference outputs and JSON timings. Runs use w2u1g-lc-0103's single SM107
GPU. These are synthetic kernel checks, not model accuracy, end-to-end serving,
multi-rank communication validation, or exhaustive tuning-knob coverage.
Native 107a and oversized-memory configurations have not been tested.
