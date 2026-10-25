"""Masking, the reconstruction head, checkpoints and multi-database binding."""
from __future__ import annotations

import pytest
import torch
import torch_frame

from conftest import ENTITY, make_batch, make_bundle, schema_of, tiny_config
from moregt.data.collate import to_cell_batch
from moregt.data.masking import (CATEGORICAL_ID, NUMERICAL_ID, build_column_target_spec,
                                 gather_masked_targets, maskable_cells, sample_cell_mask)
from moregt.model.moregt import MoREGT
from moregt.model.schema_bank import SchemaBank, SchemaTables
from moregt.train.pretrain import DivergenceGuard, PretrainModule, linear_warmup_decay, load_trunk


def test_target_spec(bundle):
    spec = build_column_target_spec(bundle)
    assert set(spec.stype[spec.maskable].tolist()) == {NUMERICAL_ID, CATEGORICAL_ID}
    assert int(spec.maskable.sum()) == 4                           # age, tier, amount, qty
    assert int(spec.n_cat[spec.stype == CATEGORICAL_ID][0]) == 3


def test_one_seed_target_per_seed(bundle, cell_batch):
    spec = build_column_target_spec(bundle)
    g = torch.Generator().manual_seed(0)
    mask, seed = sample_cell_mask(cell_batch, spec, p_random=0.0, seed_frac=0.0, generator=g)
    assert (seed.sum(1) == 1).all() and torch.equal(mask, seed)
    assert (seed & ~cell_batch.is_seed_cell).sum() == 0
    assert (mask & ~maskable_cells(cell_batch, spec)).sum() == 0


def test_missing_values_are_never_targets(bundle):
    b = make_batch()
    b["user"].tf.feat_dict[torch_frame.numerical][:] = float("nan")
    cb = to_cell_batch(b, bundle, ENTITY, seq_len=24)
    spec = build_column_target_spec(bundle)
    mask, seed = sample_cell_mask(cb, spec, p_random=1.0, seed_frac=1.0)
    t = gather_masked_targets(cb, spec, mask, seed)
    assert torch.isfinite(t.y_num).all()
    assert ((mask & ~cb.is_padding).sum(1) < (~cb.is_padding).sum(1)).all()   # never all masked


def test_numerical_targets_are_z_scored(bundle, cell_batch):
    spec = build_column_target_spec(bundle)
    mask, seed = sample_cell_mask(cell_batch, spec, p_random=1.0, seed_frac=1.0)
    t = gather_masked_targets(cell_batch, spec, mask, seed)
    ages = t.y_num[(t.stype == NUMERICAL_ID) & t.is_seed]
    assert ages.abs().max() < 10


def _module(bundles, **kw):
    torch.manual_seed(0)
    return PretrainModule(bundles, {ds: schema_of(b) for ds, b in bundles.items()}, tiny_config(),
                          seq_len=24, max_time_s=None, max_steps=100, **kw)


def test_masked_loss_trains(bundle):
    m = _module({"synthetic": bundle})
    cb = m.collate("synthetic/user", make_batch())
    loss, aux, stats = m._masked_loss("synthetic/user", cb, step_seed=0)
    assert torch.isfinite(loss) and loss > 0 and stats["n_num"] + stats["n_cat"] >= 2
    (loss + aux).backward()
    assert m.model.encoder.mask_emb.weight.grad.abs().sum() > 0


def test_trunk_round_trip(tmp_path, bundle):
    m = _module({"synthetic": bundle})
    torch.save(m.state_dict(), tmp_path / "trunk.pt")
    fresh = _module({"synthetic": bundle})
    for p in fresh.parameters():
        torch.nn.init.zeros_(p)
    load_trunk(fresh.model, tmp_path / "trunk.pt", head=fresh.head)
    for k, v in m.state_dict().items():
        assert torch.equal(fresh.state_dict()[k], v)
    other = MoREGT(bundle, schema_of(bundle), tiny_config(n_blocks=3))
    with pytest.raises(RuntimeError):
        load_trunk(other, tmp_path / "trunk.pt")


def test_schema_bank_binds_each_database():
    a, b = make_bundle(name="db-a"), make_bundle(name="db-b")
    sa, sb = schema_of(a, 16), schema_of(b, 16)
    sb.column = torch.randn_like(sb.column)
    model = MoREGT(a, sa, tiny_config())
    bank = SchemaBank({"db-a": SchemaTables.build(a, sa), "db-b": SchemaTables.build(b, sb)})
    keys = set(model.state_dict())
    bank.bind(model, "db-b")
    assert torch.equal(model.encoder.name_emb, sb.column)
    assert model.substrate.blocks[1].pool.col_name_emb is model.encoder.name_emb
    bank.bind(model, "db-a")
    assert torch.equal(model.signature.name_emb, sa.column)
    assert set(model.state_dict()) == keys


def test_multi_database_module(bundle):
    m = _module({"db-a": make_bundle(name="db-a"), "db-b": make_bundle(name="db-b")})
    cb = m.collate("db-b/user", make_batch())
    loss, aux, _ = m._masked_loss("db-b/user", cb, step_seed=1)
    assert m.bank.current == "db-b" and torch.isfinite(loss)


def test_warmup_decay_schedule():
    fn = linear_warmup_decay(100, 0.1)
    assert fn(0) == pytest.approx(0.1) and fn(9) == 1.0
    assert fn(55) == pytest.approx(0.5) and fn(100) == 0.0


class _Trainer:
    sanity_checking = False
    global_step = 0

    def __init__(self, metrics):
        self.callback_metrics = metrics


def test_guard_stops_stagnation_and_collapse():
    g = DivergenceGuard(patience=2, warmup_evals=0)
    g.on_validation_end(_Trainer({"val/skill": torch.tensor(0.5)}), None)
    g.on_validation_end(_Trainer({"val/skill": torch.tensor(0.6)}), None)
    with pytest.raises(RuntimeError, match="not improved"):
        g.on_validation_end(_Trainer({"val/skill": torch.tensor(0.7)}), None)
    g = DivergenceGuard(liveness_patience=2, warmup_evals=0)
    dead = {"val/loss_cat": torch.tensor(1.10), "val/loss_cat_uniform": torch.tensor(1.0986),
            "val/cat_acc": torch.tensor(0.33)}
    g.on_validation_end(_Trainer(dead), None)
    with pytest.raises(RuntimeError, match="collapsed"):
        g.on_validation_end(_Trainer(dead), None)
