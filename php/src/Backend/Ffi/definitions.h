/* C ABI declarations consumed by PHP; keep synchronized with Rust exports. */
typedef struct TransformerTensor TransformerTensor;

const char* transformer_native_version(void);

int transformer_tensor_create_f32(
    const float* data,
    const size_t* shape,
    size_t rank,
    TransformerTensor** output
);

void transformer_tensor_destroy(TransformerTensor* tensor);

int transformer_tensor_rank(
    const TransformerTensor* tensor,
    size_t* output
);

int transformer_tensor_numel(
    const TransformerTensor* tensor,
    size_t* output
);

int transformer_tensor_shape(
    const TransformerTensor* tensor,
    size_t* output,
    size_t capacity
);

int transformer_tensor_dtype(
    const TransformerTensor* tensor,
    int* output
);

int transformer_tensor_copy_data_f32(
    const TransformerTensor* tensor,
    float* output,
    size_t capacity
);

int transformer_tensor_add(
    const TransformerTensor* a,
    const TransformerTensor* b,
    TransformerTensor** output
);

int transformer_tensor_matmul(
    const TransformerTensor* a,
    const TransformerTensor* b,
    TransformerTensor** output
);

int transformer_tensor_linear_last_dim(
    const TransformerTensor* input,
    const TransformerTensor* weight,
    const TransformerTensor* bias,
    TransformerTensor** output
);

int transformer_tensor_layer_norm(
    const TransformerTensor* input,
    const TransformerTensor* weight,
    const TransformerTensor* bias,
    float epsilon,
    TransformerTensor** output
);

int transformer_tensor_gelu(
    const TransformerTensor* input,
    TransformerTensor** output
);

int transformer_tensor_multi_head_attention(
    const TransformerTensor* input,
    const TransformerTensor* q_weight,
    const TransformerTensor* k_weight,
    const TransformerTensor* v_weight,
    const TransformerTensor* out_weight,
    size_t heads,
    const uint8_t* mask,
    size_t mask_length,
    TransformerTensor** output
);

int transformer_tensor_embedding(
    const int64_t* token_ids,
    size_t batch,
    size_t sequence,
    const TransformerTensor* weight,
    TransformerTensor** output
);

int transformer_tensor_transpose(
    const TransformerTensor* input,
    TransformerTensor** output
);

int transformer_tensor_softmax(
    const TransformerTensor* input,
    TransformerTensor** output
);

int transformer_tensor_softmax_last_dim(
    const TransformerTensor* input,
    TransformerTensor** output
);

int transformer_tensor_add_f32(
    const float* a,
    const float* b,
    float* output,
    size_t length
);

int transformer_matmul_f32(
    const float* a,
    const float* b,
    float* output,
    size_t m,
    size_t k,
    size_t n
);

int transformer_transpose_f32(
    const float* input,
    float* output,
    size_t rows,
    size_t columns
);

int transformer_softmax_f32(
    const float* input,
    float* output,
    size_t length
);
