"""Per-database frozen tables, bound into one set of shared weights for multi-database training.

The parameters of :class:`~moregt.model.moregt.MoREGT` are identical across schemas; only the frozen
non-persistent buffers (name tables, column statistics, column layout) differ. Pretraining on several
databases therefore uses one model and re-points those buffers before each batch, which is safe
because every batch comes from one table of one database and the table schedule is the same on all
ranks.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

from torch import Tensor, nn

from .encoder import column_tables

_ENCODER_BUFFERS = ("name_emb", "cat_emb", "num_mean", "num_std", "cat_base", "modality_id",
                    "year_stats")


@dataclass
class SchemaTables:
    """Everything about one database that the shared weights read but do not learn."""

    name_emb: Tensor                       # [C, d_text]
    cat_emb: Tensor                        # [1 + n_cat, d_text]
    num_mean: Tensor                       # [C]
    num_std: Tensor                        # [C]
    cat_base: Tensor                       # [C]
    modality_id: Tensor                    # [C]
    year_stats: Tensor                     # [2]
    table_name_emb: Tensor                 # [n_tables, d_text]
    role_name_emb: Tensor                  # [K + 1, d_text]
    layout: dict = field(default_factory=dict)
    gids: dict = field(default_factory=dict)

    @classmethod
    def build(cls, bundle, schema) -> SchemaTables:
        t = column_tables(bundle)
        return cls(name_emb=schema.column, cat_emb=schema.category, table_name_emb=schema.table,
                   role_name_emb=schema.role, **t)

    def apply(self, fn) -> SchemaTables:
        tensors = {k: fn(v) for k, v in self.__dict__.items() if isinstance(v, Tensor)}
        return replace(self, gids={k: fn(v) for k, v in self.gids.items()}, **tensors)


def _set_buffer(module: nn.Module, name: str, value: Tensor) -> None:
    assert name in module._buffers, f"{type(module).__name__}.{name} is not a buffer"
    module._buffers[name] = value


def bind(model, tables: SchemaTables) -> None:
    """Point ``model``'s frozen buffers at ``tables`` (references only, no copies)."""
    enc = model.encoder
    for attr in _ENCODER_BUFFERS:
        _set_buffer(enc, attr, getattr(tables, attr))
    enc._layout, enc._gids = tables.layout, tables.gids
    _set_buffer(model.signature, "name_emb", tables.name_emb)
    _set_buffer(model.signature, "modality_id", tables.modality_id)
    sub = model.substrate
    _set_buffer(sub.row_sig, "table_name_emb", tables.table_name_emb)
    _set_buffer(sub.row_sig, "role_name_emb", tables.role_name_emb)
    for blk in sub.blocks:
        _set_buffer(blk.pool, "col_name_emb", tables.name_emb)
        _set_buffer(blk.row_attn, "role_name_emb", tables.role_name_emb)


class SchemaBank(nn.Module):
    """Holds every database's :class:`SchemaTables` and binds one at a time. A module only so the
    tables follow the model across devices; it adds nothing to the ``state_dict``."""

    def __init__(self, tables: dict[str, SchemaTables]):
        super().__init__()
        if not tables:
            raise ValueError("SchemaBank needs at least one database")
        if len({t.name_emb.shape[-1] for t in tables.values()}) > 1:
            raise ValueError("all databases must use the same name encoder")
        self._tables = dict(tables)
        self.current: str | None = None

    @property
    def datasets(self) -> list[str]:
        return sorted(self._tables)

    def _apply(self, fn, *args, **kwargs):  # type: ignore[override]
        # `nn.Module.to` reaches children through `_apply`, so this is what moves the tables.
        out = super()._apply(fn, *args, **kwargs)
        self._tables = {k: t.apply(fn) for k, t in self._tables.items()}
        self.current = None
        return out

    def bind(self, model, dataset: str) -> None:
        if dataset != self.current:
            bind(model, self._tables[dataset])
            self.current = dataset
