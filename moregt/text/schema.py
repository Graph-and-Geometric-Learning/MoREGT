"""Frozen embeddings of schema names, and the fixed modality (stype) id space.

Every name is embedded from a table-qualified string, so two same-named columns of different tables
land at different points and similar columns of different databases land close together:

* column    ``"table <t>, column <c>"``                         -> ``[C, d_text]``
* table     ``"table <t>"``                                      -> ``[n_tables, d_text]``
* FK role   ``"table <child> column <fk> references table <p>"`` -> ``[K + 1, d_text]`` (row 0 = none)
* category  ``"table <t>, column <c>, value <v>"``               -> ``[1 + n_cat, d_text]`` (row 0 = missing)

These tables are data, not parameters: they are rebuilt for any database without gradients, which is
what lets one set of weights serve every schema.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from ..data.collate import column_vocab
from ..data.graph import GraphBundle, role_name


def column_name_text(node_type: str, column: str) -> str:
    return f"table {node_type}, column {column}"


def table_name_text(node_type: str) -> str:
    return f"table {node_type}"


def category_value_text(node_type: str, column: str, value) -> str:
    return f"table {node_type}, column {column}, value {value}"


def _embed(texts: list[str], encode, kind: str) -> Tensor:
    emb = encode(texts, kind=kind)
    if not torch.is_tensor(emb):
        emb = torch.as_tensor(emb)
    return emb.detach().to(torch.float32).cpu()


def build_column_name_embeddings(bundle: GraphBundle, encode, *, kind: str = "query") -> Tensor:
    """``[C, d_text]``, row ``i`` = global column id ``i`` of :func:`column_vocab`."""
    vocab = column_vocab(bundle)
    texts = [""] * len(vocab)
    for (nt, col), i in vocab.items():
        texts[i] = column_name_text(nt, col)
    return _embed(texts, encode, kind)


def build_table_name_embeddings(bundle: GraphBundle, encode, *, kind: str = "query") -> Tensor:
    """``[n_tables, d_text]``, row ``i`` = table id ``i`` of ``bundle.node_type_id``."""
    texts = [""] * len(bundle.node_type_id)
    for nt, i in bundle.node_type_id.items():
        texts[i] = table_name_text(nt)
    return _embed(texts, encode, kind)


def build_role_name_embeddings(bundle: GraphBundle, encode, *, kind: str = "query") -> Tensor:
    """``[K + 1, d_text]``, row ``r`` = FK role id ``r``; row 0 (no role) is all zeros."""
    emb = _embed([role_name(t) for t in bundle.role_triples], encode, kind)
    return torch.cat([torch.zeros(1, emb.shape[-1], dtype=emb.dtype), emb], dim=0)


def category_index(bundle) -> tuple[dict[tuple[str, str], tuple[int, int]], list[str]]:
    """``({(table, column): (base, n_cat)}, texts)`` over one database-wide category table.

    Class ``i`` of column ``c`` is row ``base[c] + i``; row 0 is reserved for missing values.
    """
    from torch_frame import categorical
    from torch_frame.data.stats import StatType

    index: dict[tuple[str, str], tuple[int, int]] = {}
    texts: list[str] = [""]
    for nt in bundle.node_types:
        for c in bundle.data[nt].tf.col_names_dict.get(categorical, []):
            values = bundle.col_stats_dict[nt][c][StatType.COUNT][0]
            index[(nt, c)] = (len(texts), len(values))
            texts.extend(category_value_text(nt, c, v) for v in values)
    return index, texts


def build_category_name_embeddings(bundle, encode, *, kind: str = "query") -> Tensor:
    """``[1 + n_categories, d_text]``. Row 0 is embedded and then zeroed, so the table has the
    encoder's width even for a database without categorical columns."""
    _index, texts = category_index(bundle)
    emb = _embed(texts, encode, kind)
    emb[0] = 0.0
    return emb


@dataclass
class SchemaEmbeddings:
    """The four frozen name tables of one database."""

    column: Tensor
    table: Tensor
    role: Tensor
    category: Tensor


def embed_schema(bundle: GraphBundle, encode) -> SchemaEmbeddings:
    """Embed every column, table, FK role and category label of ``bundle`` with ``encode``."""
    return SchemaEmbeddings(
        column=build_column_name_embeddings(bundle, encode),
        table=build_table_name_embeddings(bundle, encode),
        role=build_role_name_embeddings(bundle, encode),
        category=build_category_name_embeddings(bundle, encode),
    )


# Modality ids. The order is fixed so a stype has the same id in every database; id 0 is reserved
# for stypes not listed. Append only: reordering changes the meaning of trained embeddings.
STYPE_ORDER: tuple[str, ...] = (
    "numerical",
    "categorical",
    "multicategorical",
    "timestamp",
    "text_embedded",
    "text_tokenized",
    "embedding",
    "image_embedded",
    "sequence_numerical",
)
MODALITY_UNKNOWN = 0
N_STYPES = 1 + len(STYPE_ORDER)
_STYPE_ID = {name: i + 1 for i, name in enumerate(STYPE_ORDER)}


def stype_id(st) -> int:
    """Fixed id of a torch_frame stype (or its name); unknown stypes map to 0."""
    name = getattr(st, "value", None) or str(st).rsplit(".", 1)[-1]
    return _STYPE_ID.get(name, MODALITY_UNKNOWN)


def build_column_modality_ids(bundle: GraphBundle) -> Tensor:
    """``[C]`` modality id of every column, indexed by global column id."""
    vocab = column_vocab(bundle)
    modality = torch.full((len(vocab),), MODALITY_UNKNOWN, dtype=torch.long)
    for nt in bundle.node_types:
        for st, cols in bundle.data[nt].tf.col_names_dict.items():
            for c in cols:
                if (nt, c) in vocab:
                    modality[vocab[(nt, c)]] = stype_id(st)
    return modality
