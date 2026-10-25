"""Sampled ``HeteroData`` minibatch -> dense, padded cell-token batch.

Every cell (row x feature column) of every sampled row is one token. Per seed, the cells of the seed
row come first, then the cells of the other rows in sampler order; the sequence is padded or truncated
to a fixed ``seq_len``. Alongside the cells the batch carries a per-seed row graph (``row_graph.py``)
for the row level of the model, and the per-table ``TensorFrame``s plus a scatter map so the cell
encoder can place encoded values into the ``[B, S, d]`` grid.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .graph import FK_NONE, GraphBundle, is_forward_relation
from .row_graph import build_row_graph

#: Timestamps before 1800-01-01 are treated as missing. relbench maps a missing timestamp (NaT) to a
#: finite sentinel in 1677, which would otherwise read as a real, very old date.
UNTIMED_FLOOR = -5364662400.0

PAD = -1


def feature_col_names(tf) -> list[str]:
    """Feature columns of a ``TensorFrame`` in canonical (sorted) order."""
    names: list[str] = []
    for cols in tf.col_names_dict.values():
        names.extend(cols)
    return sorted(names)


def column_vocab(bundle: GraphBundle) -> dict[tuple[str, str], int]:
    """Global ``(table, column) -> id`` map, sorted and cached on the bundle."""
    cached = getattr(bundle, "_column_vocab", None)
    if cached is not None:
        return cached
    pairs = [(nt, c) for nt in bundle.node_types for c in feature_col_names(bundle.data[nt].tf)]
    vocab = {p: i for i, p in enumerate(sorted(pairs))}
    bundle._column_vocab = vocab
    return vocab


@dataclass
class CellBatch:
    num_seeds: int                 # B
    seq_len: int                   # S
    max_rows: int                  # R, fitted to the batch
    # per cell [B, S]
    cell_row: Tensor               # long: row slot of the cell within its seed (pad = -1)
    cell_col: Tensor               # long: global column id (pad = -1)
    cell_time: Tensor              # float64: timestamp of the cell's row (0 if untimed)
    cell_is_timed: Tensor          # bool
    is_padding: Tensor             # bool
    is_seed_cell: Tensor           # bool: the cell belongs to the seed row
    # per seed [B]
    seed_time: Tensor              # float64
    target: Tensor                 # float32
    has_target: Tensor             # bool
    # per row [B, R] and [B, R, R]
    row_valid: Tensor              # bool
    row_table: Tensor              # long: table id (pad = -1)
    row_time: Tensor               # float64 (0 if untimed)
    row_is_timed: Tensor           # bool
    row_hop: Tensor                # long: hop distance from the seed row
    row_in_role: Tensor            # long: FK role by which the row was reached (seed = 0)
    row_is_root: Tensor            # bool: exactly one per seed
    adj_role: Tensor               # long [B, R, R]: 0 none | 1..K child | K+1..2K parent | 2K+1 self
    # value encoding, per table
    tf_dict: dict                  # table -> TensorFrame of its sampled rows
    cell_placement: dict           # table -> (b, s, row, col) index tensors

    def to(self, device) -> CellBatch:
        def mv(x):
            return x.to(device) if torch.is_tensor(x) else x

        fields = {k: mv(v) for k, v in self.__dict__.items()
                  if k not in ("tf_dict", "cell_placement")}
        return CellBatch(
            **fields,
            tf_dict={k: v.to(device) for k, v in self.tf_dict.items()},
            cell_placement={k: tuple(t.to(device) for t in v) for k, v in self.cell_placement.items()},
        )


def _seed_time_per_segment(entity_store, num_segments: int) -> Tensor:
    seed_time = entity_store.seed_time.to(torch.float64)
    batch = entity_store.batch
    out = torch.full((num_segments,), float("nan"), dtype=torch.float64)
    out[batch[: seed_time.numel()]] = seed_time
    assert not torch.isnan(out).any(), "could not recover seed_time for every segment"
    return out


def to_cell_batch(batch, bundle: GraphBundle, entity_table: str, *, seq_len: int = 1024) -> CellBatch:
    """Convert a disjoint sampled minibatch (seeds from ``entity_table``) into a :class:`CellBatch`."""
    vocab = column_vocab(bundle)
    node_types = [nt for nt in bundle.node_types if nt in batch.node_types and batch[nt].num_nodes > 0]

    # ---- flatten the sampled rows of all tables ----
    type_start: dict[str, int] = {}
    seg_list, ntype_list, rowt_list, timed_list, seed_list, intype_list, typeidx_list = ([] for _ in range(7))
    offset = 0
    for ti, nt in enumerate(node_types):
        st = batch[nt]
        n = st.num_nodes
        type_start[nt] = offset
        offset += n
        seg_list.append(st.batch.to(torch.long))
        ntype_list.append(torch.full((n,), bundle.node_type_id[nt], dtype=torch.long))
        if "time" in st:
            t = st.time.to(torch.float64)
            plausible = t > UNTIMED_FLOOR
            rowt_list.append(torch.where(plausible, t, torch.zeros_like(t)))
            timed_list.append(plausible)
        else:
            rowt_list.append(torch.zeros(n, dtype=torch.float64))
            timed_list.append(torch.zeros(n, dtype=torch.bool))
        is_seed = torch.zeros(n, dtype=torch.bool)
        if nt == entity_table and "seed_time" in st:
            is_seed[: st.seed_time.numel()] = True
        seed_list.append(is_seed)
        intype_list.append(torch.arange(n, dtype=torch.long))
        typeidx_list.append(torch.full((n,), ti, dtype=torch.long))

    seg = torch.cat(seg_list)
    ntype = torch.cat(ntype_list)
    rowt = torch.cat(rowt_list)
    timed = torch.cat(timed_list)
    is_seed_node = torch.cat(seed_list)
    intype = torch.cat(intype_list)
    typeidx = torch.cat(typeidx_list)
    total = seg.numel()

    B = int(seg.max().item()) + 1
    seed_time = _seed_time_per_segment(batch[entity_table], B)

    # row slot of each sampled row within its seed: rank in global order
    local_node = torch.empty(total, dtype=torch.long)
    order = torch.argsort(seg, stable=True)
    counts = torch.bincount(seg, minlength=B)
    starts = torch.cumsum(counts, 0) - counts
    local_node[order] = torch.arange(total, dtype=torch.long) - torch.repeat_interleave(starts, counts)

    # labels, attached to the entity store as `y`
    ent = batch[entity_table]
    target = torch.zeros(B, dtype=torch.float32)
    has_target = torch.zeros(B, dtype=torch.bool)
    if "y" in ent:
        y = ent.y.to(torch.float32)
        yb = ent.batch[: y.numel()]
        target[yb] = y
        has_target[yb] = True

    # ---- FK edges of the row graph, from every sampled edge type in both directions ----
    edge_child, edge_parent, edge_role = [], [], []
    for (src, rel, dst) in bundle.edge_types:
        if (src, rel, dst) not in batch.edge_types or src not in type_start or dst not in type_start:
            continue
        ei = batch[(src, rel, dst)].edge_index
        role = bundle.edge_fk_role((src, rel, dst))
        if ei.numel() == 0 or role == FK_NONE:
            continue
        forward = is_forward_relation(rel)
        child_nt, parent_nt = (src, dst) if forward else (dst, src)
        child_local, parent_local = (ei[0], ei[1]) if forward else (ei[1], ei[0])
        edge_child.append(child_local + type_start[child_nt])
        edge_parent.append(parent_local + type_start[parent_nt])
        edge_role.append(torch.full((child_local.numel(),), role, dtype=torch.long))

    cat = lambda xs: torch.cat(xs) if xs else torch.zeros(0, dtype=torch.long)
    rows = build_row_graph(
        seg=seg, local_node=local_node, table_id=ntype, row_t=rowt, timed=timed, is_root=is_seed_node,
        child_g=cat(edge_child), parent_g=cat(edge_parent), edge_role=cat(edge_role),
        num_seeds=B, num_roles=bundle.num_roles,
    )
    R = int(rows["adj_role"].shape[1])

    # ---- cells, seed row first ----
    feat_cols = {nt: feature_col_names(batch[nt].tf) for nt in node_types}
    col_ids = {nt: torch.tensor([vocab[(nt, c)] for c in cols], dtype=torch.long)
               for nt, cols in feat_cols.items()}
    ncols_of_type = torch.tensor([len(feat_cols[nt]) for nt in node_types], dtype=torch.long)
    cell_row = torch.full((B, seq_len), PAD, dtype=torch.long)
    cell_col = torch.full((B, seq_len), PAD, dtype=torch.long)
    is_padding = torch.ones(B, seq_len, dtype=torch.bool)
    is_seed_cell = torch.zeros(B, seq_len, dtype=torch.bool)
    cell_time = torch.zeros(B, seq_len, dtype=torch.float64)
    cell_is_timed = torch.zeros(B, seq_len, dtype=torch.bool)

    # visit order per seed: the seed row, then the other rows by slot
    key = (seg * 2 + (~is_seed_node).to(torch.long)) * (total + 1) + local_node
    visit = torch.argsort(key, stable=True)
    v_ncols = ncols_of_type[typeidx[visit]]
    v_seg = seg[visit]
    cum = torch.cumsum(v_ncols, 0)
    vc = torch.bincount(v_seg, minlength=B)
    seg_first = torch.cumsum(vc, 0) - vc
    seg_base = (cum - v_ncols)[seg_first[v_seg]]
    v_off = cum - v_ncols - seg_base                       # cells of this seed before this row
    cell_node = torch.repeat_interleave(visit, v_ncols)
    cell_off = torch.repeat_interleave(v_off, v_ncols)
    n_cells = int(v_ncols.sum())
    cell_pos = torch.arange(n_cells, dtype=torch.long) - torch.repeat_interleave(cum - v_ncols, v_ncols)
    cell_s = cell_off + cell_pos
    keep = cell_s < seq_len
    cg, cs, cp = cell_node[keep], cell_s[keep], cell_pos[keep]
    cb_ = seg[cg]
    cell_row[cb_, cs] = local_node[cg]
    is_padding[cb_, cs] = False
    is_seed_cell[cb_, cs] = is_seed_node[cg]
    cell_time[cb_, cs] = rowt[cg]
    cell_is_timed[cb_, cs] = timed[cg]
    ct = typeidx[cg]
    placement: dict[str, tuple] = {}
    for ti, nt in enumerate(node_types):
        m = ct == ti
        if not bool(m.any()):
            continue
        cell_col[cb_[m], cs[m]] = col_ids[nt][cp[m]]
        placement[nt] = (cb_[m].clone(), cs[m].clone(), intype[cg[m]].clone(), cp[m].clone())

    return CellBatch(
        num_seeds=B, seq_len=seq_len, max_rows=R,
        cell_row=cell_row, cell_col=cell_col, cell_time=cell_time, cell_is_timed=cell_is_timed,
        is_padding=is_padding, is_seed_cell=is_seed_cell,
        seed_time=seed_time, target=target, has_target=has_target,
        row_valid=rows["row_valid"], row_table=rows["row_table"], row_time=rows["row_time"],
        row_is_timed=rows["row_is_timed"], row_hop=rows["row_hop"], row_in_role=rows["row_in_role"],
        row_is_root=rows["row_is_root"], adj_role=rows["adj_role"],
        tf_dict={nt: batch[nt].tf for nt in node_types}, cell_placement=placement,
    )
