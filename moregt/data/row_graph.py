"""Per-seed row graph for the row level of the model.

A row slot ``r`` is the row's local index within its seed's subgraph (the same index a cell's
``cell_row`` carries). ``adj_role[b, r, s]`` encodes the FK edge between rows ``r`` and ``s`` and its
direction, with ``K`` FK roles::

    0           no edge
    1 .. K      s is a child of r via role k   (r holds the referenced key)
    K+1 .. 2K   s is a parent of r via role k  (r holds the foreign key)
    2K+1        r == s

``row_hop`` is the BFS distance from the seed row over the undirected edges, and ``row_in_role`` the
role of the edge by which a row was first reached.
"""
from __future__ import annotations

import torch
from torch import Tensor


def undirected_role(adj: Tensor, num_roles: int) -> Tensor:
    """``adj_role`` -> role id in ``[1, K]``; no edge and self loops -> 0."""
    out = torch.where(adj > num_roles, adj - num_roles, adj)
    return torch.where((adj <= 0) | (adj > 2 * num_roles), torch.zeros_like(out), out)


def build_row_graph(
    *,
    seg: Tensor,            # [N] seed index of each sampled row
    local_node: Tensor,     # [N] row slot within its seed
    table_id: Tensor,       # [N] table id
    row_t: Tensor,          # [N] float64 timestamp (0 if untimed)
    timed: Tensor,          # [N] bool
    is_root: Tensor,        # [N] bool, the seed row (one per seed)
    child_g: Tensor,        # [E] flat index of the FK-holding endpoint
    parent_g: Tensor,       # [E] flat index of the referenced endpoint
    edge_role: Tensor,      # [E] role id in [1, K]
    num_seeds: int,
    num_roles: int,
) -> dict[str, Tensor]:
    """The ``[B, R]`` / ``[B, R, R]`` row fields of a :class:`~moregt.data.collate.CellBatch`.

    ``R`` is the largest per-seed row count in the batch.
    """
    B, K = num_seeds, num_roles
    dev = seg.device
    num_rows = torch.bincount(seg, minlength=B)[:B]
    R = max(1, int(num_rows.max()) if num_rows.numel() else 0)

    flat = seg * R + local_node
    row_valid = torch.zeros(B * R, dtype=torch.bool, device=dev)
    row_table = torch.full((B * R,), -1, dtype=torch.long, device=dev)
    row_time = torch.zeros(B * R, dtype=torch.float64, device=dev)
    row_is_timed = torch.zeros(B * R, dtype=torch.bool, device=dev)
    row_is_root = torch.zeros(B * R, dtype=torch.bool, device=dev)
    row_valid[flat] = True
    row_table[flat] = table_id
    row_time[flat] = row_t
    row_is_timed[flat] = timed
    row_is_root[flat] = is_root
    row_valid, row_table, row_time, row_is_timed, row_is_root = (
        t.view(B, R) for t in (row_valid, row_table, row_time, row_is_timed, row_is_root))
    n_root = row_is_root.sum(dim=1)
    assert bool((n_root == 1).all()), f"expected exactly one root row per seed, got {n_root.tolist()}"

    # ---- adjacency ----
    adj = torch.zeros(B * R * R, dtype=torch.long, device=dev)
    if child_g.numel():
        b = seg[child_g]
        keep = (seg[parent_g] == b) & (child_g != parent_g)
        b, rc, rp, role = b[keep], local_node[child_g][keep], local_node[parent_g][keep], edge_role[keep]
        if rc.numel():
            # One entry per unordered pair, so both directions agree; a pair joined by several roles
            # keeps the smallest role id.
            lo, hi = torch.minimum(rc, rp), torch.maximum(rc, rp)
            pair = b * (R * R) + lo * R + hi
            _, order = torch.sort(pair * (K + 2) + role, stable=True)
            pair_s = pair[order]
            first = torch.ones_like(pair_s, dtype=torch.bool)
            first[1:] = pair_s[1:] != pair_s[:-1]
            sel = order[first]
            b, rc, rp, role = b[sel], rc[sel], rp[sel], role[sel]
            adj[b * (R * R) + rp * R + rc] = role             # from the parent: s is my child
            adj[b * (R * R) + rc * R + rp] = role + K         # from the child: s is my parent
    adj = adj.view(B, R, R)
    diag = torch.arange(R, device=dev)
    adj = adj * (row_valid.unsqueeze(2) & row_valid.unsqueeze(1)).long()
    adj[:, diag, diag] = row_valid.long() * (2 * K + 1)

    # ---- BFS from the seed row ----
    adjb = adj != 0
    adjb[:, diag, diag] = False
    row_hop = torch.full((B, R), -1, dtype=torch.long, device=dev)
    row_in_role = torch.zeros((B, R), dtype=torch.long, device=dev)
    row_hop[row_is_root] = 0
    frontier = row_is_root.clone()
    h = 0
    while bool(frontier.any()):
        h += 1
        cand = frontier.unsqueeze(2) & adjb
        new = cand.any(dim=1) & (row_hop < 0) & row_valid
        if not bool(new.any()):
            break
        pred = cand.to(torch.uint8).argmax(dim=1)
        reached_by = undirected_role(adj.gather(1, pred.unsqueeze(1)).squeeze(1), K)
        row_hop[new] = h
        row_in_role[new] = reached_by[new]
        frontier = new
    assert not bool((row_valid & (row_hop < 0)).any()), "row graph is disconnected from the seed row"

    return {"row_valid": row_valid, "row_table": row_table, "row_time": row_time,
            "row_is_timed": row_is_timed, "row_hop": row_hop, "row_in_role": row_in_role,
            "row_is_root": row_is_root, "adj_role": adj}
