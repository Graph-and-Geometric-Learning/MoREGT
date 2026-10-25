"""The MoREGT model: shapes, gradients, value-free routing and schema independence."""
from __future__ import annotations

import pytest
import torch

from conftest import EDGE_TYPES, ENTITY, make_batch, make_bundle, schema_of, tiny_config
from moregt.data.collate import to_cell_batch
from moregt.model.moregt import MoREGT, MoREGTConfig


def test_forward_shapes(model, cell_batch):
    logits, aux, cells = model(cell_batch, return_cells=True)
    assert logits.shape == (2, 1)
    assert cells.shape == (2, cell_batch.seq_len, 32)
    assert torch.isfinite(logits).all() and torch.isfinite(aux) and aux > 0


def test_gradients_reach_routers_and_signatures(model, cell_batch):
    logits, aux = model(cell_batch)
    (logits.sum() + aux).backward()
    blk = model.substrate.blocks[0]
    for p in (blk.cell_ffn.router.weight, blk.row_ffn.w_g, model.signature.schema_proj.weight,
              model.substrate.row_sig.w_tab.weight, model.encoder.proj_num.weight):
        assert p.grad is not None and p.grad.abs().sum() > 0


def test_signature_is_value_free(model, bundle):
    a = to_cell_batch(make_batch(values_seed=0), bundle, ENTITY, seq_len=24)
    b = to_cell_batch(make_batch(values_seed=1), bundle, ENTITY, seq_len=24)
    assert torch.equal(model.signature(a), model.signature(b))
    assert not torch.allclose(model.encoder(a), model.encoder(b))


def test_seed_signature_ignores_sampled_neighbours(model, bundle):
    a = to_cell_batch(make_batch(event_times=(10.0, 20.0, 30.0, 40.0)), bundle, ENTITY, seq_len=24)
    b = to_cell_batch(make_batch(event_times=(5.0, 7.0, 9.0, 11.0, 50.0, 60.0)), bundle, ENTITY,
                      seq_len=24)
    za, zb = model.signature(a), model.signature(b)
    assert torch.equal(za[a.is_seed_cell], zb[b.is_seed_cell])


def test_masking_keeps_the_name_component(model, cell_batch):
    mask = torch.zeros_like(cell_batch.is_padding)
    mask[:, 0] = True
    x = model.encoder(cell_batch, mask)
    assert torch.allclose(x[:, 0], model.encoder.masked_token(cell_batch)[:, 0])


def test_state_dict_is_schema_independent():
    """No parameter is sized by the schema: the same weights load on another database."""
    other = make_bundle(["event", "review", "user"],
                        EDGE_TYPES + [("review", "f2p_buyer", "user"), ("user", "rev_f2p_buyer", "review")])
    from conftest import _stats
    import torch_frame
    from torch_frame.data.stats import StatType

    other.data["review"].tf = torch_frame.TensorFrame(
        feat_dict={torch_frame.numerical: torch.randn(3, 1)},
        col_names_dict={torch_frame.numerical: ["score"]})
    other.col_stats_dict = _stats() | {"review": {"score": {StatType.MEAN: 0.0, StatType.STD: 1.0}}}
    a = MoREGT(make_bundle(), schema_of(make_bundle()), tiny_config())
    b = MoREGT(other, schema_of(other), tiny_config())
    sa, sb = a.state_dict(), b.state_dict()
    assert sa.keys() == sb.keys()
    assert all(sa[k].shape == sb[k].shape for k in sa)
    b.load_state_dict(sa, strict=True)


def test_default_config_is_the_134m_model(bundle):
    m = MoREGT(bundle, schema_of(bundle, dim=2560), MoREGTConfig(cell_attn_backend="sdpa"))
    assert abs(sum(p.numel() for p in m.parameters()) / 1e6 - 133.3) < 1.0


def test_config_round_trip(tmp_path):
    import json

    cfg = tiny_config(n_blocks=3)
    (tmp_path / "config.json").write_text(json.dumps({"model": cfg.to_dict()}))
    assert MoREGTConfig.from_checkpoint(tmp_path / "best_trunk.pt") == cfg


@pytest.mark.skipif(not torch.cuda.is_available(), reason="flex attention runs on CUDA")
def test_flex_matches_sdpa(bundle, cell_batch):
    torch.manual_seed(0)
    a = MoREGT(bundle, schema_of(bundle), tiny_config(cell_attn_backend="sdpa")).cuda().eval()
    b = MoREGT(bundle, schema_of(bundle), tiny_config(cell_attn_backend="flex")).cuda().eval()
    b.load_state_dict(a.state_dict())
    cb = cell_batch.to("cuda")
    with torch.no_grad():
        assert torch.allclose(a(cb)[0], b(cb)[0], atol=1e-4)
