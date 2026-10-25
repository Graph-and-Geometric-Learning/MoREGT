"""Integration on the cached rel-f1 database (skipped when it is not cached)."""
from __future__ import annotations

import functools

import torch

from conftest import rel_f1_available, schema_of, tiny_config


@functools.lru_cache(maxsize=1)
def _bundle_and_task(cache: str):
    from relbench.tasks import get_task

    from moregt.data.graph import build_graph

    return build_graph("rel-f1", cache_dir=cache), get_task("rel-f1", "driver-dnf", download=False)


@rel_f1_available
def test_sampled_rows_precede_the_seed(tmp_path_factory):
    from moregt.data.collate import to_cell_batch
    from moregt.data.graph import make_loader

    bundle, task = _bundle_and_task(str(tmp_path_factory.getbasetemp() / "rel-f1"))
    raw = next(iter(make_loader(bundle, task, "train", batch_size=32, shuffle=False)))
    for nt in raw.node_types:
        st = raw[nt]
        if "time" in st and st.num_nodes:
            seed_t = raw[task.entity_table].seed_time[st.batch]
            assert (st.time <= seed_t).all(), nt
    cb = to_cell_batch(raw, bundle, task.entity_table, seq_len=512)
    t = cb.seed_time.view(-1, 1).expand_as(cb.cell_time)
    assert (cb.cell_time[cb.cell_is_timed] <= t[cb.cell_is_timed]).all()


@rel_f1_available
def test_model_and_pretraining_on_rel_f1(tmp_path_factory):
    from moregt.data.collate import to_cell_batch
    from moregt.data.pretrain_loader import build_pretrain_stream
    from moregt.model.moregt import MoREGT
    from moregt.train.pretrain import PretrainModule

    bundle, task = _bundle_and_task(str(tmp_path_factory.getbasetemp() / "rel-f1"))
    schema = schema_of(bundle)
    module = PretrainModule({"rel-f1": bundle}, {"rel-f1": schema}, tiny_config(), seq_len=256,
                            max_time_s=None)
    rr = build_pretrain_stream({"rel-f1": bundle}, module.specs, split="train", steps=3, batch_size=8)
    for key, raw in rr:
        cb = module.collate(key, raw)
        loss, aux, stats = module._masked_loss(key, cb, step_seed=0)
        assert torch.isfinite(loss)
    model = MoREGT(bundle, schema, tiny_config())
    from moregt.data.graph import make_loader

    raw = next(iter(make_loader(bundle, task, "val", batch_size=8, shuffle=False)))
    logits, aux = model(to_cell_batch(raw, bundle, task.entity_table, seq_len=256))
    assert logits.shape == (8, 1) and torch.isfinite(logits).all()


@rel_f1_available
def test_agent_columns_on_rel_f1(tmp_path_factory):
    from pathlib import Path

    from moregt.agent.columns import apply_agent_columns
    from moregt.data.collate import to_cell_batch
    from moregt.data.graph import make_loader
    from moregt.model.moregt import MoREGT

    bundle, task = _bundle_and_task(str(tmp_path_factory.getbasetemp() / "rel-f1"))
    out = apply_agent_columns(bundle, Path(__file__).resolve().parents[1] / "agent_columns", "driver-dnf")
    assert sum(len(v) for v in out.derived_columns.values()) == 14
    raw = next(iter(make_loader(out, task, "val", batch_size=8, shuffle=False)))
    model = MoREGT(out, schema_of(out), tiny_config())
    logits, _ = model(to_cell_batch(raw, out, task.entity_table, seq_len=256))
    assert logits.shape == (8, 1) and torch.isfinite(logits).all()
