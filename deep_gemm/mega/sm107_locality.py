"""Rubin (VR200) locality domains for the opt-in SM107 per-die MegaMoE path.

Port of upstream DeepGEMM 26/09/30 `deep_gemm.localize` (weights split along N into the two locality domains), using
CUDA 13.4's native locality-domain allocations instead of an MPS `-mlopart` helper process.

Usage (after `transform_weights_for_mega_moe`, before the first call / CUDA graph capture)::

    os.environ['DG_MEGA_MOE_SM107'] = '1'; os.environ['DG_MEGA_MOE_SM107_PERDIE'] = '1'
    l1, l2, sl1, sl2 = localize_mega_moe_weights(l1, l2, sl1, sl2)   # SF tensors are kept as is
"""
from typing import Optional, Tuple

import torch

from .. import _C


def get_num_locality_domains() -> int:
    return _C.sm107_get_num_locality_domains()


def is_localization_available() -> bool:
    return torch.cuda.get_device_capability() == (10, 7) and get_num_locality_domains() == 2


def get_balanced_sm_locality_domains() -> torch.Tensor:
    """Per-SM locality domain (uint8, balanced to equal TPC counts), probed once and cached. Call it outside CUDA graph
    capture (it allocates and synchronizes); the per-die kernel launch reuses the cached map."""
    return _C.sm107_get_balanced_sm_locality_domains()


def localize(t: torch.Tensor, dim: int = -2) -> torch.Tensor:
    """`(*B, N, K)` -> `(2, *B, N / 2, K)`, slice `d` homed in locality domain `d` (a new allocation)."""
    num_domains, dim = get_num_locality_domains(), dim % t.dim()
    assert num_domains == 2, f'{num_domains} locality domains'
    assert t.size(dim) % num_domains == 0, f'`{tuple(t.shape)}` must split into {num_domains} parts along dim {dim}'
    sizes = [*t.shape[:dim], t.size(dim) // num_domains, *t.shape[dim + 1:]]
    localized = _C.sm107_empty_per_domain(sizes, t.dtype)
    localized.copy_(t.unflatten(dim, (num_domains, -1)).movedim(dim, 0))
    assert _C.sm107_is_localized(localized)
    return localized


def localize_mega_moe_weights(l1: Tuple[torch.Tensor, torch.Tensor], l2: Tuple[torch.Tensor, torch.Tensor],
                              shared_l1: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                              shared_l2: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
    """Localize transformed MegaMoE `(weights, sf)` tuples; the (small) SF tensors stay where they are."""
    def loc(pair):
        return None if pair is None else (localize(pair[0]), pair[1])
    return loc(l1), loc(l2), loc(shared_l1), loc(shared_l2)
