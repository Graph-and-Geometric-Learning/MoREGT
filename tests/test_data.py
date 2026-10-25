"""Graph vocabularies, cell-token collate and the row graph."""
from __future__ import annotations

import torch

from conftest import EDGE_TYPES, ENTITY, make_batch, make_bundle
from moregt.data.collate import column_vocab, to_cell_batch
from moregt.data.graph import build_vocabs, canonical_edge_role


def test_fk_roles_are_role_triples():
    _, roles = build_vocabs(["event", "user"], EDGE_TYPES)
    assert len(roles) == 2                                         # buyer and seller stay distinct
    assert canonical_edge_role(("event", "f2p_buyer", "user")) == canonical_edge_role(
        ("user", "rev_f2p_buyer", "event"))
    # the same FK column name in another child table is another role
    extra = EDGE_TYPES + [("review", "f2p_buyer", "user"), ("user", "rev_f2p_buyer", "review")]
    _, roles = build_vocabs(["event", "review", "user"], extra)
    assert len(roles) == 3


def test_seed_cells_come_first(bundle, cell_batch):
    cb = cell_batch
    n_user_cols = 2
    assert cb.is_seed_cell[:, :n_user_cols].all()
    assert not cb.is_seed_cell[:, n_user_cols:].any()
    # each seed: 1 user row (2 cells) + 2 event rows (3 cells each)
    assert ((~cb.is_padding).sum(1) == torch.tensor([8, 8])).all()
    assert (cb.cell_col[cb.is_padding] == -1).all()
    vocab = column_vocab(bundle)
    assert set(cb.cell_col[~cb.is_padding].tolist()) == set(vocab.values())


def test_truncation_keeps_seed_row(bundle, batch):
    cb = to_cell_batch(batch, bundle, ENTITY, seq_len=3)
    assert cb.is_seed_cell[:, :2].all() and not cb.is_padding.any()


def test_cell_times_and_untimed_rows(cell_batch):
    cb = cell_batch
    assert not cb.cell_is_timed[cb.is_seed_cell].any()             # user is untimed
    assert (cb.cell_time[cb.cell_is_timed] <= cb.seed_time.view(-1, 1).expand_as(cb.cell_time)[
        cb.cell_is_timed]).all()


def test_missing_timestamps_are_untimed(bundle):
    b = make_batch(event_times=(10.0, -9223372037.0, 30.0, 40.0))
    cb = to_cell_batch(b, bundle, ENTITY, seq_len=24)
    assert int(cb.row_is_timed.sum()) == 3


def test_row_graph(cell_batch, bundle):
    cb = cell_batch
    K = bundle.num_roles
    assert cb.max_rows == 3 and cb.row_valid.all()
    assert (cb.row_is_root.sum(1) == 1).all()
    assert cb.row_hop[cb.row_is_root].eq(0).all() and cb.row_hop[~cb.row_is_root].eq(1).all()
    root = cb.row_is_root[0].nonzero().item()
    for r in range(cb.max_rows):
        if r == root:
            assert cb.adj_role[0, r, r] == 2 * K + 1
            continue
        child_code, parent_code = cb.adj_role[0, root, r], cb.adj_role[0, r, root]
        assert 1 <= child_code <= K and parent_code == child_code + K
        assert cb.row_in_role[0, r] == child_code


def test_row_graph_from_reverse_edges_only():
    """Rows reached only through reverse edges are still connected to the seed."""
    from torch_geometric.data import HeteroData

    from conftest import event_tf, user_tf

    bundle = make_bundle()
    g = torch.Generator().manual_seed(0)
    d = HeteroData()
    d["user"].num_nodes = 1
    d["user"].batch = torch.tensor([0])
    d["user"].seed_time = torch.tensor([100.0], dtype=torch.float64)
    d["user"].tf = user_tf(1, g)
    d["event"].num_nodes = 2
    d["event"].batch = torch.tensor([0, 0])
    d["event"].time = torch.tensor([1.0, 2.0], dtype=torch.float64)
    d["event"].tf = event_tf(2, g)
    d["user", "rev_f2p_buyer", "event"].edge_index = torch.tensor([[0, 0], [0, 1]])
    cb = to_cell_batch(d, bundle, ENTITY, seq_len=16)
    assert cb.row_hop[0][cb.row_is_root[0]].tolist() == [0]
    assert cb.row_hop[0][~cb.row_is_root[0]].tolist() == [1, 1]
