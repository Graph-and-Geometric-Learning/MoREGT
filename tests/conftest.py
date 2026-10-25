"""Synthetic relational fixtures, so the tests need no network and no real database.

Schema: ``user`` (untimed entity table: ``age`` numerical, ``tier`` categorical) and ``event`` (timed:
``amount`` and ``qty`` numerical, ``placed_at`` timestamp) with two foreign keys into ``user``
(``buyer`` and ``seller``).
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from moregt.data.graph import GraphBundle, build_vocabs

ENTITY = "user"
NODE_TYPES = ["event", "user"]
EDGE_TYPES = [
    ("event", "f2p_buyer", "user"),
    ("user", "rev_f2p_buyer", "event"),
    ("event", "f2p_seller", "user"),
    ("user", "rev_f2p_seller", "event"),
]


def _stats():
    from torch_frame.data.stats import StatType

    return {
        "user": {"age": {StatType.MEAN: 40.0, StatType.STD: 10.0},
                 "tier": {StatType.COUNT: (["bronze", "silver", "gold"], [5, 3, 2])}},
        "event": {"amount": {StatType.MEAN: 0.0, StatType.STD: 1.0},
                  "qty": {StatType.MEAN: 3.0, StatType.STD: 2.0},
                  "placed_at": {StatType.YEAR_RANGE: [2000, 2020]}},
    }


def user_tf(n: int, g: torch.Generator):
    import torch_frame

    return torch_frame.TensorFrame(
        feat_dict={torch_frame.numerical: 40.0 + 10.0 * torch.randn(n, 1, generator=g),
                   torch_frame.categorical: torch.randint(0, 3, (n, 1), generator=g)},
        col_names_dict={torch_frame.numerical: ["age"], torch_frame.categorical: ["tier"]})


def event_tf(n: int, g: torch.Generator):
    import torch_frame

    hi = torch.tensor([2020, 12, 28, 7, 24, 60, 60])
    lo = torch.tensor([2000, 0, 0, 0, 0, 0, 0])
    ts = (lo + (torch.rand(n, 7, generator=g) * (hi - lo)).long()).unsqueeze(1)    # [n, 1, 7]
    return torch_frame.TensorFrame(
        feat_dict={torch_frame.numerical: torch.randn(n, 2, generator=g),
                   torch_frame.timestamp: ts},
        col_names_dict={torch_frame.numerical: ["amount", "qty"],
                        torch_frame.timestamp: ["placed_at"]})


def make_bundle(node_types=NODE_TYPES, edge_types=EDGE_TYPES, name="synthetic") -> GraphBundle:
    from torch_geometric.data import HeteroData

    g = torch.Generator().manual_seed(0)
    d = HeteroData()
    d["user"].tf = user_tf(4, g)
    d["event"].tf = event_tf(8, g)
    node_type_id, fk_role_id = build_vocabs(node_types, edge_types)
    return GraphBundle(dataset_name=name, data=d, col_stats_dict=_stats(), node_types=list(node_types),
                       edge_types=list(edge_types), node_type_id=node_type_id, fk_role_id=fk_role_id)


def make_batch(*, seed_time: float = 100.0, event_times=(10.0, 20.0, 30.0, 40.0), values_seed: int = 0,
               targets=(1.0, 0.0)):
    """Two seeds; each has its ``user`` row and two ``event`` rows, one per FK role."""
    from torch_geometric.data import HeteroData

    g = torch.Generator().manual_seed(values_seed)
    n_ev = len(event_times)
    d = HeteroData()
    d["user"].num_nodes = 2
    d["user"].batch = torch.tensor([0, 1])
    d["user"].seed_time = torch.tensor([seed_time, seed_time], dtype=torch.float64)
    d["user"].n_id = torch.tensor([0, 1])
    d["user"].y = torch.tensor(targets)
    d["user"].tf = user_tf(2, g)
    d["event"].num_nodes = n_ev
    d["event"].batch = torch.tensor([i * 2 // n_ev for i in range(n_ev)])
    d["event"].time = torch.tensor(event_times, dtype=torch.float64)
    d["event"].tf = event_tf(n_ev, g)
    half = n_ev // 2
    # seed 0 owns events [0, half), seed 1 owns [half, n); alternate buyer / seller edges
    buyer = [i for i in range(n_ev) if i % 2 == 0]
    seller = [i for i in range(n_ev) if i % 2 == 1]
    owner = lambda i: 0 if i < half else 1
    for rel, rows in (("buyer", buyer), ("seller", seller)):
        d["event", f"f2p_{rel}", "user"].edge_index = torch.tensor([rows, [owner(i) for i in rows]])
        d["user", f"rev_f2p_{rel}", "event"].edge_index = torch.tensor([[owner(i) for i in rows], rows])
    return d


def tiny_config(**kw):
    from moregt.model.moregt import MoREGTConfig

    base = dict(d_model=32, n_blocks=2, n_heads=2, d_ff=64, d_sig=16, num_experts=4, num_shared=1,
                top_k=2, row_num_experts=2, row_num_shared=1, pool_slots=2,
                cell_attn_backend="sdpa")
    return MoREGTConfig(**(base | kw))


def schema_of(bundle, dim: int = 16):
    from moregt.text.encoders import HashEncoder
    from moregt.text.schema import embed_schema

    return embed_schema(bundle, HashEncoder(dim=dim))


@pytest.fixture
def bundle() -> GraphBundle:
    return make_bundle()


@pytest.fixture
def batch():
    return make_batch()


@pytest.fixture
def cell_batch(bundle, batch):
    from moregt.data.collate import to_cell_batch

    return to_cell_batch(batch, bundle, ENTITY, seq_len=24)


@pytest.fixture
def model(bundle):
    from moregt.model.moregt import MoREGT

    torch.manual_seed(0)
    return MoREGT(bundle, schema_of(bundle), tiny_config())


def _rel_f1_cached() -> bool:
    try:
        from relbench.utils import get_relbench_cache_dir
    except Exception:
        return False
    return (Path(get_relbench_cache_dir()) / "rel-f1" / "db").exists()


rel_f1_available = pytest.mark.skipif(not _rel_f1_cached(), reason="rel-f1 is not in the RelBench cache")
