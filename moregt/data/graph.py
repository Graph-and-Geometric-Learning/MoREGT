"""RelBench database -> heterogeneous temporal graph, and the leakage-safe temporal loader.

Rows are nodes, tables are node types, and every foreign key gives a pair of typed edges
(``f2p_<col>`` and ``rev_f2p_<col>``). Each FK **role** ``(child_table, fk_column, parent_table)``
gets an id in ``[1, K]`` shared by both directions, so two FKs into the same parent stay distinct.
Cell values live in one torch_frame ``TensorFrame`` per table.

The loader is PyG's temporal ``NeighborSampler`` with ``disjoint=True``: every seed gets its own
subgraph, and only rows with ``time <= seed_time`` are sampled.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from collections.abc import Callable

import torch
from torch import Tensor

FK_NONE = 0  # role id of "no FK edge"

_VALIDATE_RETRIES = 8


def _patch_multi_embedding_validate() -> None:
    """Make torch_frame's ``MultiEmbeddingTensor.validate`` robust to a shared-memory race.

    With DataLoader workers, the ``offset`` tensor of a text-embedding column can be read while its
    shared storage is being reused and fail validation spuriously. Re-reading a private copy of it
    fixes that. If the only remaining problem is a non-zero leading offset, the offsets are rebased
    onto the referenced value columns; anything else raises.
    """
    try:
        from torch_frame.data.multi_embedding_tensor import MultiEmbeddingTensor
    except Exception:
        return
    if getattr(MultiEmbeddingTensor, "_moregt_patched", False):
        return
    original = MultiEmbeddingTensor.validate

    def leading_offset_only(self) -> bool:
        off = self.offset
        return (off is not None and off.ndim == 1 and off.numel() == int(self.num_cols) + 1
                and int(off[0]) != 0 and (self.values.ndim == 2 or self.values.numel() == 0))

    def validate(self):
        for _ in range(_VALIDATE_RETRIES):
            try:
                return original(self)
            except AssertionError:
                if self.offset is None:
                    raise
                object.__setattr__(self, "offset", self.offset.clone())
        if leading_offset_only(self):
            k = int(self.offset[0])
            width = int(self.offset[-1]) - k
            w = int(self.values.shape[1]) if self.values.ndim == 2 else 0
            if self.offset.numel() == 1 or w in (width, k + width):
                if w == k + width and self.offset.numel() > 1:
                    object.__setattr__(self, "values", self.values[:, k:])
                object.__setattr__(self, "offset", self.offset - k)
        return original(self)

    MultiEmbeddingTensor.validate = validate
    MultiEmbeddingTensor._moregt_patched = True


_patch_multi_embedding_validate()


def canonical_relation(rel: str) -> str:
    """``f2p_<col>`` / ``rev_f2p_<col>`` -> ``<col>``."""
    r = rel[len("rev_"):] if rel.startswith("rev_") else rel
    for p in ("f2p_", "p2f_"):
        if r.startswith(p):
            return r[len(p):]
    return r


def is_forward_relation(rel: str) -> bool:
    return not rel.startswith("rev_")


def canonical_edge_role(edge_type: tuple[str, str, str]) -> tuple[str, str, str]:
    """PyG edge type (either direction) -> its role triple ``(child_table, fk_column, parent_table)``."""
    src, rel, dst = edge_type
    col = canonical_relation(rel)
    return (src, col, dst) if is_forward_relation(rel) else (dst, col, src)


def role_name(role: tuple[str, str, str]) -> str:
    """The string embedded for an FK role."""
    child, col, parent = role
    return f"table {child} column {col} references table {parent}"


class HashTextEmbedder:
    """Deterministic pseudo-random vectors for free-text cells. For tests only."""

    def __init__(self, dim: int = 32):
        self.dim = dim

    def __call__(self, sentences: list[str]) -> Tensor:
        out = torch.empty(len(sentences), self.dim)
        for i, s in enumerate(sentences):
            digest = hashlib.sha256(str(s).encode()).hexdigest()[:16]
            g = torch.Generator().manual_seed(int(digest, 16) & 0x7FFFFFFF)
            out[i] = torch.rand(self.dim, generator=g)
        return out


@dataclass
class GraphBundle:
    """A database as a graph, with its column statistics and table / FK-role vocabularies."""

    dataset_name: str
    data: object  # torch_geometric.data.HeteroData
    col_stats_dict: dict
    node_types: list[str]
    edge_types: list[tuple[str, str, str]]
    node_type_id: dict[str, int]
    fk_role_id: dict[tuple[str, str, str], int] = field(default_factory=dict)  # ids >= 1
    #: Name of the task whose past labels were appended as a table (see ``task_context``).
    task_context: str | None = None

    @property
    def num_roles(self) -> int:
        """K, the number of FK roles (ids ``1..K``)."""
        return len(self.fk_role_id)

    @property
    def role_triples(self) -> list[tuple[str, str, str]]:
        """Role triples ordered by id: ``role_triples[i]`` is role ``i + 1``."""
        return [t for t, _ in sorted(self.fk_role_id.items(), key=lambda kv: kv[1])]

    def edge_fk_role(self, edge_type: tuple[str, str, str]) -> int:
        return self.fk_role_id.get(canonical_edge_role(edge_type), FK_NONE)


def build_vocabs(node_types, edge_types) -> tuple[dict, dict]:
    """``(table -> id, role triple -> id)``, both sorted for determinism."""
    node_type_id = {nt: i for i, nt in enumerate(sorted(node_types))}
    roles = sorted({canonical_edge_role(et) for et in edge_types})
    fk_role_id = {r: i + 1 for i, r in enumerate(roles)}
    return node_type_id, fk_role_id


def build_graph(
    dataset_name: str = "rel-f1",
    *,
    text_encoder: str | None = None,
    text_embedder: Callable[[list[str]], Tensor] | None = None,
    text_batch_size: int = 256,
    cache_dir: str | None = None,
    download: bool = False,
    task=None,
    task_name: str | None = None,
) -> GraphBundle:
    """Load a RelBench database and build its :class:`GraphBundle`.

    ``text_encoder`` (e.g. ``"minilm"``) embeds free-text cell values; without it they fall back to
    :class:`HashTextEmbedder`. ``cache_dir`` holds the materialised tables and must be keyed by the
    same encoder (:func:`moregt.utils.graph_cache_dir`).

    With ``task`` and ``task_name``, the task's train and val rows are appended as the table
    ``task_history`` (:mod:`moregt.data.task_context`), so an entity's past labels become context
    cells. Such a bundle belongs to that one task, and ``cache_dir`` must be task-specific.
    """
    from relbench.datasets import get_dataset
    from relbench.modeling.graph import make_pkey_fkey_graph
    from relbench.modeling.utils import get_stype_proposal
    from torch_frame.config.text_embedder import TextEmbedderConfig

    db = get_dataset(dataset_name, download=download).get_db(upto_test_timestamp=False)
    stype_dict = get_stype_proposal(db)
    if task is not None:
        from .task_context import TASK_TABLE, append_task_history, task_history_stypes

        if not task_name:
            raise ValueError("task_name is required with task")
        db = append_task_history(db, task, task_name)
        stype_dict[TASK_TABLE] = task_history_stypes(task_name)
    if text_embedder is None:
        if text_encoder and text_encoder != "hash":
            from ..text.encoders import value_encoder

            text_embedder = value_encoder(text_encoder)
        else:
            text_embedder = HashTextEmbedder()
    cfg = TextEmbedderConfig(text_embedder=text_embedder, batch_size=text_batch_size)
    data, col_stats = make_pkey_fkey_graph(db, stype_dict, text_embedder_cfg=cfg, cache_dir=cache_dir)

    node_types = list(data.node_types)
    edge_types = list(data.edge_types)
    node_type_id, fk_role_id = build_vocabs(node_types, edge_types)
    return GraphBundle(
        dataset_name=dataset_name,
        data=data,
        col_stats_dict=col_stats,
        node_types=node_types,
        edge_types=edge_types,
        node_type_id=node_type_id,
        fk_role_id=fk_role_id,
        task_context=task_name if task is not None else None,
    )


_BOOL_TARGET = {"t": 1.0, "f": 0.0, "true": 1.0, "false": 0.0, "True": 1.0, "False": 0.0,
                True: 1.0, False: 0.0, "1": 1.0, "0": 0.0}


def coerce_binary_target(table, task):
    """Map boolean-string targets (``'t'``/``'f'``) of binary tasks to ``{1.0, 0.0}``."""
    from relbench.base import TaskType

    col = getattr(task, "target_col", None)
    if task.task_type != TaskType.BINARY_CLASSIFICATION or col is None or col not in table.df.columns:
        return table
    s = table.df[col]
    if s.dtype != object and s.dtype != bool:
        return table
    df = table.df.copy()
    df[col] = s.map(lambda v: _BOOL_TARGET.get(v, v)).astype("float64")
    return type(table)(df=df, fkey_col_to_pkey_table=table.fkey_col_to_pkey_table,
                       pkey_col=table.pkey_col, time_col=table.time_col)


def make_loader(
    bundle: GraphBundle,
    task,
    split: str = "train",
    *,
    num_neighbors: list[int] | None = None,
    batch_size: int = 64,
    shuffle: bool = False,
    num_workers: int = 0,
    table=None,
):
    """Temporal, per-seed disjoint loader over the rows of ``task``'s ``split`` table (or ``table``).

    Each sampled batch carries ``seed_time`` and the labels ``y`` on the entity store.
    """
    from relbench.modeling.graph import get_node_train_table_input
    from relbench.modeling.loader import CustomNodeLoader, NeighborSampler

    table = coerce_binary_target(table if table is not None else task.get_table(split), task)
    inp = get_node_train_table_input(table, task)
    sampler = NeighborSampler(
        bundle.data,
        num_neighbors=num_neighbors or [12, 12],
        time_attr="time",
        temporal_strategy="last",
        disjoint=True,
    )
    return CustomNodeLoader(
        bundle.data,
        node_sampler=sampler,
        input_nodes=inp.nodes,
        input_time=inp.time,
        transform=inp.transform,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )
