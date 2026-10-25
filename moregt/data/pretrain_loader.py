"""Label-free seed sampling for masked-cell pretraining.

Seeds are ordinary table rows instead of task rows; sampling and collation are otherwise the same as
in fine-tuning. Each batch draws all its seeds from one table (the collate needs a single seed
table), and :class:`RoundRobinLoader` interleaves the per-table streams with probability proportional
to (number of maskable columns x number of eligible rows), normalised within each database so every
database gets an equal share of the steps.

Pretraining never looks past RelBench's validation timestamp ``t_val``: train seeds have
``t* <= t_min + 0.9 (t_val - t_min)``, validation seeds lie in the remaining 10% of the window, and
the temporal sampler then keeps every sampled row before ``t*``. A timed table's seed time is the
row's own timestamp; an untimed table's seed time is drawn uniformly from the split's window.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import torch
from torch import Tensor

VAL_TAIL_FRACTION = 0.1


@dataclass
class SeedTable:
    """One table's eligible seed rows for a split, and its sampling weight."""

    node_type: str
    index: Tensor        # long [n] row ids
    time: Tensor         # long [n] seed timestamps
    weight: float        # maskable columns x eligible rows
    timed: bool

    def __len__(self) -> int:
        return int(self.index.numel())


def maskable_tables(bundle, spec) -> dict[str, int]:
    """Table -> number of maskable columns."""
    return {nt: int(((spec.table == bundle.node_type_id[nt]) & spec.maskable).sum())
            for nt in bundle.node_types}


def _time_bounds(bundle) -> tuple[int, int]:
    """``(earliest row time, val_timestamp)`` in unix seconds."""
    from relbench.datasets import get_dataset

    val = int(get_dataset(bundle.dataset_name, download=False).val_timestamp.timestamp())
    lo = None
    for nt in bundle.node_types:
        st = bundle.data[nt]
        if "time" in st and st.time.numel():
            t = st.time[st.time > 0]
            if t.numel():
                lo = int(t.min()) if lo is None else min(lo, int(t.min()))
    return (lo if lo is not None else 0), val


def build_seed_tables(bundle, spec, split: str = "train", *,
                      generator: torch.Generator | None = None) -> list[SeedTable]:
    """Eligible seed rows per table for ``split`` in {train, val}. Tables without a maskable column
    are never seeds but are still sampled as neighbours."""
    if split not in ("train", "val"):
        raise ValueError(f"split must be 'train' or 'val', got {split!r}")
    lo, val_ts = _time_bounds(bundle)
    cut = int(lo + (1.0 - VAL_TAIL_FRACTION) * (val_ts - lo))
    t_lo, t_hi = (lo, cut) if split == "train" else (cut, val_ts)

    per_table = maskable_tables(bundle, spec)
    out: list[SeedTable] = []
    for nt in bundle.node_types:
        w = per_table.get(nt, 0)
        store = bundle.data[nt]
        if w == 0 or store.num_nodes == 0:
            continue
        if "time" in store:
            t = store.time.to(torch.long)
            idx = ((t > 0) & (t >= t_lo) & (t <= t_hi)).nonzero(as_tuple=True)[0]
            if idx.numel():
                out.append(SeedTable(nt, idx, t[idx], float(w) * float(idx.numel()), True))
        else:
            n = store.num_nodes
            draw = torch.randint(0, max(t_hi - t_lo, 1), (n,), generator=generator,
                                 dtype=torch.long) + t_lo
            out.append(SeedTable(nt, torch.arange(n, dtype=torch.long), draw, float(w) * float(n), False))
    if not out:
        raise ValueError(f"{bundle.dataset_name}/{split}: no table has a maskable column and an "
                         f"eligible row in [{t_lo}, {t_hi}]")
    return out


def make_neighbor_sampler(bundle, num_neighbors=None):
    """One temporal sampler per database, shared by all its seed tables (each sampler holds a full
    CSC copy of the graph). Memoised on the bundle by fanout."""
    from relbench.modeling.loader import NeighborSampler

    fanout = tuple(num_neighbors or [12, 12])
    cache = getattr(bundle, "_nbr_samplers", None)
    if cache is None:
        cache = bundle._nbr_samplers = {}
    if fanout not in cache:
        cache[fanout] = NeighborSampler(bundle.data, num_neighbors=list(fanout), time_attr="time",
                                        temporal_strategy="last", disjoint=True)
    return cache[fanout]


def make_pretrain_loader(bundle, seed_table: SeedTable, *, sampler, batch_size: int = 64,
                         shuffle: bool = True, max_seeds: int | None = None):
    """A label-free temporal loader over one table's seed rows."""
    from relbench.modeling.loader import CustomNodeLoader

    idx, t = seed_table.index, seed_table.time
    if max_seeds is not None and idx.numel() > max_seeds:
        pick = torch.randperm(idx.numel())[:max_seeds]
        idx, t = idx[pick], t[pick]
    return CustomNodeLoader(bundle.data, node_sampler=sampler, input_nodes=(seed_table.node_type, idx),
                            input_time=t, batch_size=batch_size, shuffle=shuffle, num_workers=0)


class RoundRobinLoader:
    """Interleaves per-table loaders, drawing each step's table with probability ``weights``.

    Yields ``(key, raw_batch)`` for ``steps`` steps. The table schedule depends only on ``seed``, so
    every DDP rank draws the same table at the same step.
    """

    def __init__(self, loaders: dict[str, object], weights: dict[str, float], *, steps: int,
                 seed: int = 0):
        if not loaders:
            raise ValueError("RoundRobinLoader needs at least one table loader")
        self.names = sorted(loaders)
        self.loaders = loaders
        w = torch.tensor([max(weights.get(n, 0.0), 0.0) for n in self.names], dtype=torch.double)
        if float(w.sum()) <= 0:
            w = torch.ones_like(w)
        self.probs = w / w.sum()
        self.steps = int(steps)
        self.seed = seed

    def table_schedule(self) -> list[str]:
        g = torch.Generator().manual_seed(self.seed)
        pick = torch.multinomial(self.probs, self.steps, replacement=True, generator=g)
        return [self.names[int(i)] for i in pick]

    def __len__(self) -> int:
        return self.steps

    def next_batch(self, iters: dict, name: str):
        """One raw batch of table ``name``, restarting its stream when exhausted."""
        for _ in range(2):
            it = iters.get(name)
            if it is None:
                it = iters[name] = iter(self.loaders[name])
            try:
                return next(it)
            except StopIteration:
                iters[name] = None
        return None

    def __iter__(self):
        iters: dict = {}
        for name in self.table_schedule():
            batch = self.next_batch(iters, name)
            if batch is not None:
                yield name, batch


def build_pretrain_stream(bundles: dict, specs: dict, *, split: str = "train", steps: int,
                          batch_size: int = 64, num_neighbors=None, seed: int = 0,
                          max_seeds_per_table: int | None = None) -> RoundRobinLoader:
    """One stream over the seed tables of several databases, keyed ``"<dataset>/<table>"``."""
    loaders, weights = {}, {}
    for ds, bundle in bundles.items():
        tables = build_seed_tables(bundle, specs[ds], split,
                                   generator=torch.Generator().manual_seed(seed))
        sampler = make_neighbor_sampler(bundle, num_neighbors)
        w = torch.tensor([st.weight for st in sorted(tables, key=lambda s: s.node_type)],
                         dtype=torch.double)
        probs = (w / w.sum()).tolist()
        total = sum(probs) or 1.0
        for st, p in zip(sorted(tables, key=lambda s: s.node_type), probs, strict=True):
            key = f"{ds}/{st.node_type}"
            loaders[key] = make_pretrain_loader(bundle, st, sampler=sampler, batch_size=batch_size,
                                                shuffle=split == "train",
                                                max_seeds=max_seeds_per_table)
            weights[key] = p / total
    return RoundRobinLoader(loaders, weights, steps=steps, seed=seed)


def split_key(key: str) -> tuple[str, str]:
    """``"rel-f1/results"`` -> ``("rel-f1", "results")``."""
    ds, _, table = key.partition("/")
    return ds, table


def _rank() -> int:
    """DDP rank from the environment (safe inside a dataloader worker)."""
    for var in ("RANK", "LOCAL_RANK", "SLURM_PROCID"):
        v = os.environ.get(var)
        if v is not None and v.isdigit():
            return int(v)
    return 0


class PretrainWorkerStream(torch.utils.data.IterableDataset):
    """The whole mixture as one ``IterableDataset``, so a single worker pool samples and collates.

    Worker ``w`` of ``W`` handles the steps ``t`` with ``t % W == w``. The table of each step is the
    same on every rank, while the rows differ: the RNG is reseeded per ``(seed, rank, worker)``.
    """

    def __init__(self, rr: RoundRobinLoader, *, collate=None, seed: int = 0):
        super().__init__()
        self.rr = rr
        self.collate = collate
        self.seed = int(seed)

    def __len__(self) -> int:
        return len(self.rr)

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        wid = 0 if info is None else int(info.id)
        nw = 1 if info is None else int(info.num_workers)
        torch.manual_seed(self.seed + 9176 * _rank() + 31 * wid)
        iters: dict = {}
        for t, name in enumerate(self.rr.table_schedule()):
            if t % nw != wid:
                continue
            raw = self.rr.next_batch(iters, name)
            if raw is not None:
                yield (name, raw) if self.collate is None else (name, self.collate(name, raw))


def _identity(x):
    return x


def build_worker_loader(rr: RoundRobinLoader, *, collate=None, num_workers: int = 0, seed: int = 0):
    """Wrap a :class:`RoundRobinLoader` in one worker pool; ``num_workers=0`` iterates in-process."""
    ds = PretrainWorkerStream(rr, collate=collate, seed=seed)
    if num_workers <= 0:
        return ds
    return torch.utils.data.DataLoader(ds, batch_size=None, num_workers=num_workers,
                                       collate_fn=_identity, persistent_workers=True,
                                       prefetch_factor=2)
