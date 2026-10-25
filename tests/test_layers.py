"""Mixture-of-experts layers and the time ladder."""
from __future__ import annotations

import torch

from moregt.model.moe import MoEFFN, orthogonality_loss, top_k_gates
from moregt.model.time_encoding import TimeLadder


def test_gates_are_top_k_and_normalised():
    g, topi = top_k_gates(torch.randn(10, 6), 2)
    assert torch.allclose(g.sum(-1), torch.ones(10))
    assert ((g > 0).sum(-1) == 2).all()
    assert torch.equal(torch.sort(g.topk(2, -1).indices, -1).values, torch.sort(topi, -1).values)


def test_sparse_combine_matches_dense():
    torch.manual_seed(0)
    moe = MoEFFN(8, 16, 4, num_experts=4, k=2, num_shared=0)
    x, z = torch.randn(3, 5, 8), torch.randn(3, 5, 4)
    g = moe.gates(z)
    dense = sum(g[..., e:e + 1] * moe.experts[e](x) for e in range(4))
    assert torch.allclose(moe(x, z), dense, atol=1e-6)


def test_invalid_tokens_are_skipped():
    torch.manual_seed(0)
    moe = MoEFFN(8, 16, 4, num_experts=4, k=2, num_shared=1)
    x, z = torch.randn(2, 5, 8), torch.randn(2, 5, 4)
    valid = torch.ones(2, 5, dtype=torch.bool)
    valid[:, -2:] = False
    y = moe(x, z, valid=valid)
    assert (y[~valid] == 0).all()
    assert torch.allclose(y[valid], moe(x, z)[valid], atol=1e-6)


def test_orthogonality_loss():
    assert orthogonality_loss(torch.eye(4)).item() == 0.0
    assert orthogonality_loss(torch.ones(4, 3)).item() > 0


def test_time_frequencies_are_fixed_buffers():
    ladder = TimeLadder()
    assert "omega" not in ladder.state_dict()
    assert torch.allclose(ladder.omega[[0, -1]], torch.tensor([0.3, 5.0], dtype=torch.float64))


def test_rotary_scores_depend_on_relative_log_time():
    ladder = TimeLadder()
    torch.manual_seed(0)
    q, k = torch.randn(16), torch.randn(16)
    tau = torch.tensor([3.0, 5.0], dtype=torch.float64)

    def score(t):
        th = ladder.theta(t)
        return (ladder.rotate(q, th[0]) * ladder.rotate(k, th[1])).sum()

    assert torch.allclose(score(tau), score(tau + 2.5), atol=1e-5)


def test_untimed_and_clamped_states_are_separable():
    ladder = TimeLadder()
    tau = torch.zeros(3, dtype=torch.float64)
    f = ladder.feats(tau, torch.tensor([True, False, True]), torch.tensor([False, False, True]))
    assert f[1].abs().sum() == 0                                   # untimed: all zero
    assert f[0, :-1].abs().sum() > 0 and f[0, -1] == 0            # Delta = 0
    assert f[2, -1] == 1                                           # dated after the seed
