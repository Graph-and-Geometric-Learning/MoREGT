"""Masked-cell pretraining targets: which cells may be masked, which are, and their labels.

A cell is maskable if it is non-missing and its column is numerical with non-zero variance or
categorical with at least two classes. Timestamps (republished as row time), free text and
multi-categorical columns are never targets.

Per sequence, one maskable cell of the seed row is the target (as in RT); ``seed_frac`` masks that
fraction of the seed row's other maskable cells, and ``p_random`` that fraction of the remaining
maskable cells of the sequence. Numerical targets are z-scored with the column statistics the
encoder uses; categorical targets are the class index within the column.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from ..text.schema import category_index, stype_id

NUMERICAL_ID = stype_id("numerical")
CATEGORICAL_ID = stype_id("categorical")

#: relbench's constant filler column for a table without features.
CONST_COL = "__const__"
_MIN_STD = 1e-8


@dataclass
class ColumnTargetSpec:
    """Per-column target description, indexed by global column id (all tensors ``[C]``)."""

    maskable: Tensor      # bool
    stype: Tensor         # long, fixed stype id
    within_idx: Tensor    # long, index of the column inside tf.feat_dict[stype]
    table: Tensor         # long, table id
    num_mean: Tensor      # float
    num_std: Tensor       # float (std + 1e-6)
    n_cat: Tensor         # long, number of classes
    cat_base: Tensor      # long, first row of the column in the category-label table

    def to(self, device) -> ColumnTargetSpec:
        return ColumnTargetSpec(**{k: v.to(device) for k, v in self.__dict__.items()})

    def summary(self) -> str:
        m = int(self.maskable.sum())
        cat = int((self.maskable & (self.stype == CATEGORICAL_ID)).sum())
        return f"{m}/{self.maskable.numel()} maskable ({m - cat} numerical, {cat} categorical)"


def build_column_target_spec(bundle) -> ColumnTargetSpec:
    import torch_frame
    from torch_frame.data.stats import StatType

    from .collate import column_vocab

    vocab = column_vocab(bundle)
    C = len(vocab)
    cat_ix, _ = category_index(bundle)
    maskable = torch.zeros(C, dtype=torch.bool)
    stype = torch.zeros(C, dtype=torch.long)
    within = torch.full((C,), -1, dtype=torch.long)
    table = torch.full((C,), -1, dtype=torch.long)
    mean = torch.zeros(C, dtype=torch.float32)
    std = torch.ones(C, dtype=torch.float32)
    n_cat = torch.zeros(C, dtype=torch.long)
    cat_base = torch.zeros(C, dtype=torch.long)

    for nt in bundle.node_types:
        stats = bundle.col_stats_dict[nt]
        for st, cols in bundle.data[nt].tf.col_names_dict.items():
            for j, c in enumerate(cols):
                if (nt, c) not in vocab:
                    continue
                gid = vocab[(nt, c)]
                stype[gid], within[gid], table[gid] = stype_id(st), j, bundle.node_type_id[nt]
                if st == torch_frame.numerical:
                    m, s = stats[c].get(StatType.MEAN), stats[c].get(StatType.STD)
                    if m is not None and s is not None:
                        mean[gid], std[gid] = float(m), float(s) + 1e-6
                        maskable[gid] = c != CONST_COL and float(s) > _MIN_STD
                elif st == torch_frame.categorical:
                    cat_base[gid], n_cat[gid] = cat_ix[(nt, c)]
                    maskable[gid] = int(n_cat[gid]) >= 2
    return ColumnTargetSpec(maskable=maskable, stype=stype, within_idx=within, table=table,
                            num_mean=mean, num_std=std, n_cat=n_cat, cat_base=cat_base)


def _raw_values(cb, spec: ColumnTargetSpec) -> tuple[Tensor, Tensor]:
    """``([B, S] raw value, [B, S] has-a-value)`` for numerical and categorical cells."""
    import torch_frame

    B, S = cb.cell_col.shape
    dev = cb.cell_col.device
    val = torch.zeros(B, S, dtype=torch.float32, device=dev)
    ok = torch.zeros(B, S, dtype=torch.bool, device=dev)
    gid = cb.cell_col.clamp_min(0).reshape(-1)
    cell_stype = spec.stype.to(dev).index_select(0, gid).view(B, S)
    cell_within = spec.within_idx.to(dev).index_select(0, gid).view(B, S)

    for nt, (b_idx, s_idx, row_idx, _col_idx) in cb.cell_placement.items():
        tf = cb.tf_dict.get(nt)
        if tf is None:
            continue
        b_idx, s_idx, row_idx = b_idx.to(dev), s_idx.to(dev), row_idx.to(dev)
        for st, key in ((torch_frame.numerical, NUMERICAL_ID), (torch_frame.categorical, CATEGORICAL_ID)):
            feat = tf.feat_dict.get(st)
            if feat is None:
                continue
            sel = cell_stype[b_idx, s_idx] == key
            if not bool(sel.any()):
                continue
            bb, ss, rr = b_idx[sel], s_idx[sel], row_idx[sel]
            v = feat[rr, cell_within[bb, ss]]
            # torch_frame marks missing values with NaN (numerical) and -1 (categorical)
            val[bb, ss] = v.to(torch.float32)
            ok[bb, ss] = ~torch.isnan(v) if v.is_floating_point() else v >= 0
    return val, ok


def maskable_cells(cb, spec: ColumnTargetSpec) -> Tensor:
    """``[B, S]`` bool: real, non-missing cells of maskable columns."""
    dev = cb.cell_col.device
    B, S = cb.cell_col.shape
    ok_col = spec.maskable.to(dev).index_select(0, cb.cell_col.clamp_min(0).reshape(-1)).view(B, S)
    _val, has_value = _raw_values(cb, spec)
    return (~cb.is_padding) & ok_col & has_value


def sample_cell_mask(cb, spec: ColumnTargetSpec, *, p_random: float = 0.0, seed_frac: float = 0.5,
                     generator: torch.Generator | None = None) -> tuple[Tensor, Tensor]:
    """-> ``(mask [B, S], is_seed_target [B, S])``. A seed row without a maskable cell gets no
    seed target."""
    cand = maskable_cells(cb, spec)
    dev = cand.device
    B, S = cand.shape

    seed_cand = cand & cb.is_seed_cell
    noise = torch.rand(B, S, device=dev, generator=generator).masked_fill(~seed_cand, -1.0)
    pick = noise.argmax(dim=1)
    has_pick = seed_cand.any(dim=1)
    is_seed_target = torch.zeros_like(cand)
    is_seed_target[torch.arange(B, device=dev)[has_pick], pick[has_pick]] = True

    mask = is_seed_target.clone()
    if seed_frac > 0:
        draw = torch.rand(B, S, device=dev, generator=generator) < seed_frac
        mask |= seed_cand & draw & ~is_seed_target
    if p_random > 0:
        draw = torch.rand(B, S, device=dev, generator=generator) < p_random
        mask |= cand & draw & ~is_seed_target

    # Never mask every visible cell of a sequence: drop the extra masks first, then the seed target.
    real = ~cb.is_padding
    blind = (mask & real).sum(1) >= real.sum(1)
    if bool(blind.any()):
        mask[blind] = is_seed_target[blind]
        still = (mask & real).sum(1) >= real.sum(1)
        if bool(still.any()):
            mask[still] = False
            is_seed_target[still] = False
    return mask, is_seed_target


@dataclass
class MaskedTargets:
    """Flat per-masked-cell targets; ``b``/``s`` index the ``[B, S]`` cell grid."""

    b: Tensor           # long [M]
    s: Tensor           # long [M]
    gid: Tensor         # long [M], global column id
    stype: Tensor       # long [M]
    y_num: Tensor       # float [M], z-scored value (0 if not numerical)
    y_cat: Tensor       # long [M], class index (-1 if not categorical)
    is_seed: Tensor     # bool [M]

    def __len__(self) -> int:
        return int(self.b.numel())


def gather_masked_targets(cb, spec: ColumnTargetSpec, mask: Tensor, is_seed_target: Tensor) -> MaskedTargets:
    dev = mask.device
    val, _ok = _raw_values(cb, spec)
    b, s = mask.nonzero(as_tuple=True)
    gid = cb.cell_col[b, s].clamp_min(0)
    st = spec.stype.to(dev).index_select(0, gid)
    mean = spec.num_mean.to(dev).index_select(0, gid)
    std = spec.num_std.to(dev).index_select(0, gid)
    raw = val[b, s]
    y_num = torch.where(st == NUMERICAL_ID, (raw - mean) / std, torch.zeros_like(raw))
    y_cat = torch.where(st == CATEGORICAL_ID, raw.to(torch.long),
                        torch.full_like(raw, -1, dtype=torch.long))
    return MaskedTargets(b=b, s=s, gid=gid, stype=st, y_num=y_num, y_cat=y_cat,
                         is_seed=is_seed_target[b, s])
