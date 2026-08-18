# BERT-compatible encoder

The approved model family is additive and does not alter the immutable
NN-1--NN-7 Pre-Norm family. Its contract is BERT Post-Norm with biased
Q/K/V/output projections, exact GELU, absolute learned position embeddings,
token-type embeddings and embedding LayerNorm.

The first checkpoint target is `BAAI/bge-small-en-v1.5`; the fundamental model
output will be `last_hidden_state [B,S,D]`. Pooling and L2 normalization belong
to a later sentence-embedding wrapper.

Safetensors metadata, selective payload reading, strict Float32
materialization, the closed BGE manifest, `config.json` validation and atomic
`BertModel` construction are implemented. Exact GELU and biased BERT attention
use isolated additive backend/ABI operations; the original NN family is
unchanged. See `docs/model-loading.md`.
