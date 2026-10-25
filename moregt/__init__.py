"""MoREGT: a two-level relational graph transformer whose feed-forward layers are mixtures of
experts routed on a value-free relational signature of each cell and row.

Subpackages:
  data/   relational graph, temporal sampler, cell-token collate, past-label context, masking
  text/   frozen text encoders and the schema-name embedding tables
  model/  cell encoder, signatures, two-level blocks, mixture-of-experts layers, heads
  train/  masked-cell pretraining and supervised fine-tuning
  eval/   metrics and RelBench test-set evaluation
"""

__version__ = "1.0.0"
