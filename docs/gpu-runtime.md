# GPU-R1 — BGE CUDA runtime

The CUDA path is additive and specialized for the validated
`BAAI/bge-small-en-v1.5` topology. The CPU Tensor, NN-1–NN-7, BERT and
Safetensors contracts are unchanged. PHP performs tokenization, then one CUDA
FFI call owns the complete numerical pipeline:

```text
PHP tokenizer
  -> input_ids + attention_mask + token_type_ids
  -> one H2D boundary
  -> embeddings -> 12 BERT blocks -> CLS pooling -> L2
  -> one [B,384] D2H boundary
  -> list<float> / list<list<float>>
```

All 197 Float32 parameters are uploaded once during atomic model construction
and stay allocated on the selected device until the model is destroyed. No
Tensor, activation or hidden state is materialized in host memory between
layers. cuBLAS executes every projection and dedicated CUDA kernels execute the
remaining operations.

## Operation inventory

Percentages use the MODEL-R6 optimized CPU profile for one nine-token sentence
(`38.214 ms`). Attention sub-operation percentages are diagnostic kernel shares;
the 34.64% aggregate also includes CPU validation, allocation and orchestration.
“None” in the transfer column means the value remains device-resident.

| Operation | CPU implementation | GPU implementation | % CPU baseline | Calls/encode | H2D/D2H required? | Decision | Reason |
|---|---|---|---:|---:|---|---|---|
| Tokenization | PHP WordPiece | unchanged PHP | outside profile | 1 | initial H2D | CPU-only acceptable | String handling does not consume a device value. |
| IDs/type/mask copy | native inputs | three `cudaMemcpy` uploads | <0.1% | 1 boundary | H2D only | mandatory boundary | Small integer inputs are the sole forward inputs. |
| Word lookup | Rust embedding | `embeddings` kernel | in 0.43% | 1 | none | mandatory GPU | Starts the resident activation chain. |
| Position lookup | Rust embedding | same kernel | in 0.43% | 1 | none | mandatory GPU | Avoids downloading the summed embedding. |
| Token-type lookup | Rust embedding | same kernel | in 0.43% | 1 | none | mandatory GPU | Same resident fusion. |
| Embedding sums | two adds | fused in `embeddings` | in 0.43% | 2 logical | none | covered by embedding GPU | Fusion removes intermediates. |
| Embedding LayerNorm | Welford Float64 | `layer_norm` kernel | in 0.43% | 1 | none | mandatory GPU | Output feeds block 0. |
| Q projection + bias | `linear_last_dim` | cuBLAS + bias kernel | 2.72% diagnostic | 12 | none | GPU implemented | Repeated dense operation. |
| K projection + bias | `linear_last_dim` | cuBLAS + bias kernel | 2.69% diagnostic | 12 | none | GPU implemented | Repeated dense operation. |
| V projection + bias | `linear_last_dim` | cuBLAS + bias kernel | 2.56% diagnostic | 12 | none | GPU implemented | Repeated dense operation. |
| Q×K scores | attention kernel | `fused_attention` kernel | 1.00% diagnostic | 12 | none | GPU implemented | Consumes resident Q/K without a global probability buffer. |
| Score scaling | attention kernel | fused in scores | included above | 12 | none | covered by score GPU | No separate buffer or launch. |
| Padding mask | attention kernel | fused in scores | included above | 12 | none | mandatory GPU | A CPU mask step would force round trips. |
| Attention softmax | stable softmax | fused in `fused_attention` | 0.20% diagnostic | 12 | none | mandatory GPU | Probabilities stay in shared memory. |
| Attention × V | attention kernel | fused in `fused_attention` | 0.93% diagnostic | 12 | none | covered by fused attention | Produces resident merged heads in the same launch. |
| Output projection + bias | `linear_last_dim` | cuBLAS + bias kernel | 2.65% diagnostic | 12 | none | GPU implemented | Dense operation before residual. |
| Attention residual add | add kernel | fused into LayerNorm | in 2.73% | 12 | none | covered by LayerNorm GPU | Removes a launch and temporary. |
| Attention LayerNorm | Welford Float64 | `layer_norm` kernel | in 2.73% | 12 | none | mandatory GPU | Post-Norm output feeds FFN. |
| FFN input Linear + bias | `linear_last_dim` | cuBLAS + bias kernel | 13.89% | 12 | none | GPU implemented | Major dense hotspot. |
| ExactGELU (erf) | exact GELU | `gelu` kernel with `erff` | 28.38% | 12 | none | GPU implemented | Largest non-GEMM CPU hotspot. |
| FFN output Linear + bias | `linear_last_dim` | cuBLAS + bias kernel | 15.10% | 12 | none | GPU implemented | Major dense hotspot. |
| FFN residual add | add kernel | fused into LayerNorm | in 2.65% | 12 | none | covered by LayerNorm GPU | No extra activation transfer. |
| FFN LayerNorm | Welford Float64 | `layer_norm` kernel | in 2.65% | 12 | none | mandatory GPU | Keeps the next layer resident. |
| CLS pooling | slice extraction | `cls_l2` kernel | 0.61% | 1 | none | mandatory GPU | Avoids downloading last hidden state. |
| Mean pooling | generic component | not in official CUDA path | 0% official path | 0 | none | CPU-only acceptable | Available for CPU generic models; no GPU value enters it. |
| L2 normalization | PHP component | fused in `cls_l2` | 0.20% | 1 | none | mandatory GPU | Only the normalized vector is downloaded. |
| Final materialization | CData to PHP | `[B,384]` decode | 0.05% | 1 | D2H only | mandatory boundary | This is the public result. |
| Parameters | resident CPU Tensors | 197 device allocations | load-only | 197/model | H2D at load only | mandatory residence | No weight upload during forward. |
| Activations | at least 177 Rust Vecs | 12 persistent workspace buffers | structural | 0 allocations steady-state | none | GPU internal | Capacity grows when required and never shrinks between forwards. |
| Synchronization | CPU FFI lifecycle | one final stream synchronization | structural | 1 explicit | no layer transfer | GPU internal | One persistent stream orders kernels/cuBLAS without layer host sync. |

The only numerical component not ported is generic `MeanPooling`, which is not
executed by the official CLS-pooled BGE path. Tokenization is intentionally on
the CPU because its output is the initial device input rather than an
intermediate GPU activation. Neither causes an internal round trip.

## Build and use

CUDA is opt-in so ordinary CPU builds do not require the toolkit:

```bash
cargo build --release --manifest-path runtime/Cargo.toml --features cuda
```

```php
use Omegaalfa\Transformer\Backend\BackendType;
use Omegaalfa\Transformer\Backend\Ffi\FfiBackend;
use Omegaalfa\Transformer\Backend\Ffi\NativeLibrary;
use Omegaalfa\Transformer\Model\Loader\CudaBgeEmbeddingModelLoader;
use Omegaalfa\Transformer\Runtime\Runtime;
use Omegaalfa\Transformer\Runtime\RuntimeConfig;

$library = NativeLibrary::defaultPath(__DIR__);
$runtime = new Runtime(
    new FfiBackend(new NativeLibrary($library)),
    new RuntimeConfig(BackendType::Ffi),
);
$model = (new CudaBgeEmbeddingModelLoader($runtime, $library))
    ->load('/path/to/bge-small-en-v1.5');

$embedding = $model->encode('How do I request annual leave?'); // list<float>[384]
$batch = $model->encodeBatch(['first text', 'second text']);   // [2,384]
```

The CPU runtime above is used only by the existing validated Safetensors
materializer during model construction. Forward execution uses the CUDA handle
owned by `CudaBgeEmbeddingModel`.

## Validation result

On an RTX 3060 with CUDA 12, the real checkpoint loaded all 197 parameters.
For `hello world`, 15 warmed samples measured p50/p95 `8.675/12.759 ms`, versus
MODEL-R6 CPU p50/p95 `34.682/41.169 ms`. Official-reference parity passed with
maximum absolute error `1.0430813e-7` and maximum relative error `3.5754e-5`
(relative error is reported only where `|expected| >= 1e-3`). A 100-forward
soak remained deterministic and CUDA free memory was identical before and
after (`delta = 0` bytes).

## GPU-R2 optimization result

GPU-R2 keeps strict FP32 as the default and adds a persistent per-model stream
and workspace. The first/growing shape performs one workspace reallocation;
every steady-state forward performs zero `cudaMalloc` and zero `cudaFree`.
Checked arithmetic covers `B*S`, `rows*384`, `B*12*S*S`, and `rows*1536` before
allocation or launch. Failed parameter uploads free their device allocation and
leave the parameter unpublished.

The numerical kernels now use 128-thread block reductions for LayerNorm,
four-warp query blocks for fused scores/mask/softmax, one 32-thread warp per
attention output head, and a 128-thread CLS/L2 reduction. Residual+LayerNorm,
scale+mask+softmax, and CLS+L2 remain fused. Logical launch count remains 207:
cuBLASLt bias epilogues had no supported row-major heuristic on this CUDA 12.0
host, while strided-batched attention regressed batch latency and was removed.

Paired serial/parallel measurements showed p50 speedups from `2.33x` to `5.34x`
across the required shapes. The final 25-sample `hello world` run measured
GPU-R1 `8.675/12.759 ms` versus GPU-R2 `3.260/10.195 ms` p50/p95, or
`2.66x/1.25x`. Official-reference parity passed with maximum absolute error
`1.4901161e-7` and maximum relative error `3.0390721e-5`. The 100-forward soak
was deterministic with zero VRAM growth.

Explicit TF32 was rejected. Across the final matrix it was inconsistent, did
not provide the required additional 10%, and reached maximum absolute error
`5.5848062e-4` and maximum relative error `0.26857` for B=32/S=74. Strict FP32
therefore remains production default.

CUDA Events expose 137 timing intervals to the benchmark profiler. For
B=1/S=11, the warmed GPU total was `3.856 ms`; aggregate block costs were Q/K/V
`1.123 ms`, FFN1 `0.517 ms`, FFN2 `0.547 ms`, output projections `0.369 ms`,
LayerNorms `0.457 ms`, fused scores/softmax `0.198 ms`, GELU `0.255 ms`, and
Attention×V `0.180 ms`. At B=8/S=74, fused scores/softmax becomes the largest
single stage at `3.232 ms` of a `13.418 ms` GPU forward.

```bash
TRANSFORMER_BGE_CUDA_CHECKPOINT=/path/to/bge-small-en-v1.5 \
TRANSFORMER_BGE_CUDA_SAMPLES=15 \
TRANSFORMER_BGE_CUDA_SOAK=100 \
php -d xdebug.mode=off runtime/benches/bge_cuda.php
```
## GPU-R3: CUDA Graph e atenção fundida

O forward BGE mantém H2D e D2H fora do grafo e captura somente o pipeline
device-resident. O cache guarda o último par exato `[batch, sequence]`; mudança
de shape, crescimento do workspace ou alteração do modo invalida o executável.
Após um forward de aquecimento e a captura, cada forward usa uma chamada de
`cudaGraphLaunch`, contendo 195 operações internas, sem sincronização entre
layers e com uma única sincronização antes de entregar o embedding ao host.

A atenção FP32 combina QKᵀ, máscara, softmax e Attention×V em um kernel por
query/head. As probabilidades ficam em memória compartilhada, portanto o
workspace não mantém mais o buffer global `[B,H,S,S]`. A implementação continua
especializada para `head_dim=32` e `S <= 512`; os 197 Parameters permanecem
residentes.

O caminho de aplicação não muda:

```php
$model = (new CudaBgeEmbeddingModelLoader($runtime, $library))->load($checkpoint);
$embedding = $model->encode('hello world'); // list<float> com 384 elementos
```

`encode()` tokeniza no host, envia IDs/máscara/tipos uma vez, executa embeddings,
12 blocos, CLS pooling e L2 normalization no dispositivo, e copia somente o
vetor final. O controle `setGraphEnabled()` existe no wrapper CUDA apenas para
benchmarks A/B; produção mantém o grafo habilitado.

O candidato `SGEMM -> bias_exact_gelu` foi rejeitado: a pequena melhora de
mediana não compensou a regressão de p95. O caminho preserva os kernels separados
de bias e ExactGELU.

### GPU-R3.1: baseline público reconciliado

O baseline oficial mede exclusivamente `$model->encode($text)` com `hrtime()`,
incluindo tokenizer, preparação FFI, H2D, graph, D2H e materialização da lista.
CUDA Events são usados apenas para decompor o mesmo caminho e nunca substituem
o tempo público end-to-end. Para `hello world`, o tokenizer produz `[1,4]`.

O benchmark executa 100 warmups, 250 amostras e um A/B graph OFF/ON no mesmo
modelo. Também comprova a sequência normal → captura → reuse, mede 1000 forwards
com shape constante e separa crescimento esperado do workspace de vazamento:

```bash
TRANSFORMER_BGE_CUDA_CHECKPOINT=/path/to/bge-small-en-v1.5 \
php -d xdebug.mode=off runtime/benches/bge_cuda.php
```

Na RTX 3060 usada para o gate, o baseline `PUBLIC BGE CUDA WARM ENCODE` mediu
`1.246/1.838/2.789 ms` p50/p95/p99. Graph OFF mediu `2.340/8.774 ms`
p50/p95. O soak de shape constante manteve a VRAM idêntica; crescer de S=4
para a sequência longa reservou 4 MiB uma única vez e as repetições seguintes
tiveram delta zero.
