# BERT-compatible encoder

The approved model family is additive and does not alter the immutable
NN-1--NN-7 Pre-Norm family. Its future contract is BERT Post-Norm with biased
Q/K/V/output projections, exact GELU, absolute learned position embeddings,
token-type embeddings and embedding LayerNorm.

The first checkpoint target is `BAAI/bge-small-en-v1.5`; the fundamental model
output will be `last_hidden_state [B,S,D]`. Pooling and L2 normalization belong
to a later sentence-embedding wrapper.

Safetensors metadata, selective payload reading, strict Float32
materialization and closed manifest-to-Parameter mapping are implemented. The
exact BGE manifest, `config.json` validation and BertModel construction remain
pending. See `docs/model-loading.md`.
