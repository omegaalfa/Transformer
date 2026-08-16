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
