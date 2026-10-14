import pytest
import torch


@pytest.mark.parametrize("domain_one_tpcs", [0, 1, 51, 52, 53, 103, 104])
def test_balance_moves_only_surplus_tpcs(domain_one_tpcs):
    from deep_gemm.mega.sm107_locality import _balance_sm_locality_domains

    domains = torch.tensor(
        [1] * domain_one_tpcs + [0] * (104 - domain_one_tpcs),
        dtype=torch.uint8, device="cpu",
    ).repeat_interleave(2)
    original = domains.clone()
    balanced = _balance_sm_locality_domains(domains)
    assert torch.equal(domains, original)
    assert balanced.tolist().count(0) == 104
    assert balanced.tolist().count(1) == 104
    assert torch.equal(balanced[::2], balanced[1::2])
    assert int((balanced != original).sum()) == abs(domain_one_tpcs - 52) * 2
