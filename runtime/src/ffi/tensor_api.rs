use std::ffi::c_int;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::ptr;

use crate::kernels::add::add_f32;
use crate::kernels::attention::multi_head_attention_f32;
use crate::kernels::embedding::embedding_f32;
use crate::kernels::gelu::gelu_f32;
use crate::kernels::layer_norm::layer_norm_f32;
use crate::kernels::linear::linear_last_dim_f32;
use crate::kernels::matmul_dispatch::matmul_dispatch_f32;
use crate::kernels::softmax::{softmax_f32, softmax_last_dim_f32};
use crate::kernels::transpose::transpose_f32;
use crate::tensor::{DType, Shape, Strides, Tensor};

use super::handle::TransformerTensor;
use super::{STATUS_INSUFFICIENT_BUFFER, STATUS_INVALID_ARGUMENT, STATUS_OK, STATUS_PANIC};

const DTYPE_FLOAT32: c_int = 0;

#[cfg(test)]
thread_local! {
    static PANIC_LAYER_NORM: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    static PANIC_GELU: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    static PANIC_ATTENTION: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Creates an exclusively owned native Float32 Tensor.
///
/// # Safety
///
/// `output` must be valid for one writable handle pointer. For non-zero rank,
/// `shape` must be valid for `rank` elements. When the resulting shape has at
/// least one element, `data` must be valid for `numel` readable floats. Input
/// buffers only need to remain valid for this call because their contents are
/// copied into native storage.
#[no_mangle]
pub unsafe extern "C" fn transformer_tensor_create_f32(
    data: *const f32,
    shape: *const usize,
    rank: usize,
    output: *mut *mut TransformerTensor,
) -> c_int {
    if output.is_null() || rank > isize::MAX as usize / size_of::<usize>() {
        return STATUS_INVALID_ARGUMENT;
    }

    // SAFETY: `output` is non-null and writable by the caller contract.
    unsafe { output.write(ptr::null_mut()) };

    if rank > 0 && shape.is_null() {
        return STATUS_INVALID_ARGUMENT;
    }

    catch_unwind(AssertUnwindSafe(|| {
        let dimensions = if rank == 0 {
            Vec::new()
        } else {
            // SAFETY: Representable rank and non-null shape were checked;
            // allocation validity remains the caller's responsibility.
            unsafe { std::slice::from_raw_parts(shape, rank) }.to_vec()
        };

        let Ok(shape) = Shape::new(dimensions) else {
            return STATUS_INVALID_ARGUMENT;
        };
        let length = shape.numel();

        if length > isize::MAX as usize / size_of::<f32>() || (length > 0 && data.is_null()) {
            return STATUS_INVALID_ARGUMENT;
        }

        let values = if length == 0 {
            Vec::new()
        } else {
            // SAFETY: Representable length and non-null data were checked;
            // allocation validity remains the caller's responsibility.
            unsafe { std::slice::from_raw_parts(data, length) }.to_vec()
        };

        let Ok(tensor) = Tensor::from_vec(values, shape) else {
            return STATUS_INVALID_ARGUMENT;
        };
        let handle = Box::into_raw(Box::new(TransformerTensor::new(tensor)));

        // SAFETY: `output` is valid by the caller contract and initialized once.
        unsafe { output.write(handle) };
        STATUS_OK
    }))
    .unwrap_or(STATUS_PANIC)
}

/// Releases a handle returned by `transformer_tensor_create_f32`.
///
/// # Safety
///
/// A non-null pointer must be a live handle produced by this library and must
/// be passed to this function exactly once. Null is accepted as a no-op.
#[no_mangle]
pub unsafe extern "C" fn transformer_tensor_destroy(tensor: *mut TransformerTensor) {
    if tensor.is_null() {
        return;
    }

    let _ = catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: The caller guarantees unique ownership of a live handle.
        drop(unsafe { Box::from_raw(tensor) });
    }));
}

/// Writes the Tensor rank to caller-owned memory.
///
/// # Safety
///
/// `tensor` must point to a live handle and `output` must be writable.
#[no_mangle]
pub unsafe extern "C" fn transformer_tensor_rank(
    tensor: *const TransformerTensor,
    output: *mut usize,
) -> c_int {
    if tensor.is_null() || output.is_null() {
        return STATUS_INVALID_ARGUMENT;
    }

    catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: Both pointers satisfy the caller contract and null checks.
        let tensor = unsafe { &*tensor };
        unsafe { output.write(tensor.tensor().rank()) };
        STATUS_OK
    }))
    .unwrap_or(STATUS_PANIC)
}

/// Writes the Tensor element count to caller-owned memory.
///
/// # Safety
///
/// `tensor` must point to a live handle and `output` must be writable.
#[no_mangle]
pub unsafe extern "C" fn transformer_tensor_numel(
    tensor: *const TransformerTensor,
    output: *mut usize,
) -> c_int {
    if tensor.is_null() || output.is_null() {
        return STATUS_INVALID_ARGUMENT;
    }

    catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: Both pointers satisfy the caller contract and null checks.
        let tensor = unsafe { &*tensor };
        unsafe { output.write(tensor.tensor().numel()) };
        STATUS_OK
    }))
    .unwrap_or(STATUS_PANIC)
}

/// Copies the Tensor shape into a caller-owned buffer.
///
/// # Safety
///
/// `tensor` must point to a live handle. For non-zero rank, `output` must be
/// writable for at least `capacity` dimensions and must not overlap the handle.
#[no_mangle]
pub unsafe extern "C" fn transformer_tensor_shape(
    tensor: *const TransformerTensor,
    output: *mut usize,
    capacity: usize,
) -> c_int {
    if tensor.is_null() {
        return STATUS_INVALID_ARGUMENT;
    }

    catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: The caller guarantees a live handle.
        let tensor = unsafe { &*tensor };
        let dimensions = tensor.tensor().shape().as_slice();

        if capacity < dimensions.len() {
            return STATUS_INSUFFICIENT_BUFFER;
        }
        if dimensions.is_empty() {
            return STATUS_OK;
        }
        if output.is_null() {
            return STATUS_INVALID_ARGUMENT;
        }

        // SAFETY: Capacity and non-null output are checked; writable allocation
        // and non-overlap are required by the caller contract.
        unsafe { ptr::copy_nonoverlapping(dimensions.as_ptr(), output, dimensions.len()) };
        STATUS_OK
    }))
    .unwrap_or(STATUS_PANIC)
}

/// Writes the stable C ABI dtype discriminant (`0` for Float32).
///
/// # Safety
///
/// `tensor` must point to a live handle and `output` must be writable.
#[no_mangle]
pub unsafe extern "C" fn transformer_tensor_dtype(
    tensor: *const TransformerTensor,
    output: *mut c_int,
) -> c_int {
    if tensor.is_null() || output.is_null() {
        return STATUS_INVALID_ARGUMENT;
    }

    catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: Both pointers satisfy the caller contract and null checks.
        let tensor = unsafe { &*tensor };
        let dtype = match tensor.tensor().dtype() {
            DType::Float32 => DTYPE_FLOAT32,
        };
        unsafe { output.write(dtype) };
        STATUS_OK
    }))
    .unwrap_or(STATUS_PANIC)
}

/// Copies Tensor data into a caller-owned Float32 buffer.
///
/// # Safety
///
/// `tensor` must point to a live handle. For a non-empty Tensor, `output` must
/// be writable for at least `capacity` floats and must not overlap the handle.
/// The handle remains valid and retains ownership of its native storage.
#[no_mangle]
pub unsafe extern "C" fn transformer_tensor_copy_data_f32(
    tensor: *const TransformerTensor,
    output: *mut f32,
    capacity: usize,
) -> c_int {
    if tensor.is_null() {
        return STATUS_INVALID_ARGUMENT;
    }

    catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: The caller guarantees a live handle.
        let tensor = unsafe { &*tensor };
        let data = tensor.tensor().as_slice();

        if capacity < data.len() {
            return STATUS_INSUFFICIENT_BUFFER;
        }
        if data.is_empty() {
            return STATUS_OK;
        }
        if output.is_null() {
            return STATUS_INVALID_ARGUMENT;
        }

        // SAFETY: Capacity and non-null output are checked; writable allocation
        // and non-overlap are required by the caller contract.
        unsafe { ptr::copy_nonoverlapping(data.as_ptr(), output, data.len()) };
        STATUS_OK
    }))
    .unwrap_or(STATUS_PANIC)
}

/// Adds two identically shaped Float32 Tensors into a new owned Tensor.
///
/// # Safety
///
/// `a` and `b` must point to live handles for the duration of the call.
/// `output` must be writable for one handle pointer and must not alias either
/// input handle. The input handles remain valid and unchanged on success.
#[no_mangle]
pub unsafe extern "C" fn transformer_tensor_add(
    a: *const TransformerTensor,
    b: *const TransformerTensor,
    output: *mut *mut TransformerTensor,
) -> c_int {
    if output.is_null() {
        return STATUS_INVALID_ARGUMENT;
    }

    // SAFETY: `output` is non-null and writable by the caller contract.
    unsafe { output.write(ptr::null_mut()) };

    if a.is_null() || b.is_null() {
        return STATUS_INVALID_ARGUMENT;
    }

    catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: The caller guarantees two live immutable handles.
        let a = unsafe { &*a }.tensor();
        let b = unsafe { &*b }.tensor();

        if a.dtype() != b.dtype()
            || a.shape().as_slice() != b.shape().as_slice()
            || a.numel() != b.numel()
        {
            return STATUS_INVALID_ARGUMENT;
        }

        let mut values = vec![0.0; a.numel()];
        if add_f32(a.as_slice(), b.as_slice(), &mut values).is_err() {
            return STATUS_INVALID_ARGUMENT;
        }

        let shape = a.shape().clone();
        let Ok(tensor) = Tensor::from_vec(values, shape) else {
            return STATUS_INVALID_ARGUMENT;
        };
        let handle = Box::into_raw(Box::new(TransformerTensor::new(tensor)));

        // SAFETY: `output` is valid by the caller contract and written once.
        unsafe { output.write(handle) };
        STATUS_OK
    }))
    .unwrap_or(STATUS_PANIC)
}

/// Multiplies two rank-2 Float32 Tensors into a new owned Tensor.
///
/// # Safety
///
/// `a` and `b` must point to live handles for the duration of the call.
/// `output` must be writable for one handle pointer and must not alias either
/// input handle. The input handles remain valid and unchanged on success.
#[no_mangle]
pub unsafe extern "C" fn transformer_tensor_matmul(
    a: *const TransformerTensor,
    b: *const TransformerTensor,
    output: *mut *mut TransformerTensor,
) -> c_int {
    if output.is_null() {
        return STATUS_INVALID_ARGUMENT;
    }

    // SAFETY: `output` is non-null and writable by the caller contract.
    unsafe { output.write(ptr::null_mut()) };

    if a.is_null() || b.is_null() {
        return STATUS_INVALID_ARGUMENT;
    }

    catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: The caller guarantees two live immutable handles.
        let a = unsafe { &*a }.tensor();
        let b = unsafe { &*b }.tensor();

        if a.dtype() != DType::Float32
            || b.dtype() != DType::Float32
            || a.rank() != 2
            || b.rank() != 2
        {
            return STATUS_INVALID_ARGUMENT;
        }

        let a_shape = a.shape().as_slice();
        let b_shape = b.shape().as_slice();
        let (m, k, n) = (a_shape[0], a_shape[1], b_shape[1]);
        if k != b_shape[0] {
            return STATUS_INVALID_ARGUMENT;
        }

        let Ok(shape) = Shape::new(vec![m, n]) else {
            return STATUS_INVALID_ARGUMENT;
        };
        let mut values = vec![0.0; shape.numel()];
        if matmul_dispatch_f32(a.as_slice(), b.as_slice(), &mut values, m, k, n).is_err() {
            return STATUS_INVALID_ARGUMENT;
        }

        let Ok(tensor) = Tensor::from_vec(values, shape) else {
            return STATUS_INVALID_ARGUMENT;
        };
        let handle = Box::into_raw(Box::new(TransformerTensor::new(tensor)));

        // SAFETY: `output` is valid by the caller contract and written once.
        unsafe { output.write(handle) };
        STATUS_OK
    }))
    .unwrap_or(STATUS_PANIC)
}

/// Selects rows from a resident rank-2 Float32 embedding weight into a new
/// owned Tensor with shape `[batch, sequence, embedding_dim]`.
///
/// # Safety
///
/// `weight` must point to a live handle and `output` must be writable for one
/// handle pointer. For non-zero `batch * sequence`, `token_ids` must remain
/// readable for that many `i64` elements during the call. Inputs remain alive
/// and unchanged on success or error.
#[no_mangle]
pub unsafe extern "C" fn transformer_tensor_embedding(
    token_ids: *const i64,
    batch: usize,
    sequence: usize,
    weight: *const TransformerTensor,
    output: *mut *mut TransformerTensor,
) -> c_int {
    if output.is_null() {
        return STATUS_INVALID_ARGUMENT;
    }
    // SAFETY: `output` is non-null and writable by the caller contract.
    unsafe { output.write(ptr::null_mut()) };

    let Some(token_count) = batch.checked_mul(sequence) else {
        return STATUS_INVALID_ARGUMENT;
    };
    if weight.is_null()
        || token_count > isize::MAX as usize / size_of::<i64>()
        || (token_count > 0 && token_ids.is_null())
    {
        return STATUS_INVALID_ARGUMENT;
    }

    catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: The caller guarantees a live immutable handle.
        let weight = unsafe { &*weight }.tensor();
        if weight.dtype() != DType::Float32 || weight.rank() != 2 {
            return STATUS_INVALID_ARGUMENT;
        }

        let weight_shape = weight.shape().as_slice();
        let (vocab_size, embedding_dim) = (weight_shape[0], weight_shape[1]);
        if vocab_size == 0 || embedding_dim == 0 {
            return STATUS_INVALID_ARGUMENT;
        }
        let Some(output_length) = token_count.checked_mul(embedding_dim) else {
            return STATUS_INVALID_ARGUMENT;
        };
        if output_length > isize::MAX as usize / size_of::<f32>() {
            return STATUS_INVALID_ARGUMENT;
        }

        let token_ids = if token_count == 0 {
            &[]
        } else {
            // SAFETY: Representable length and non-null pointer were checked;
            // allocation validity remains the caller's responsibility.
            unsafe { std::slice::from_raw_parts(token_ids, token_count) }
        };
        let Ok(shape) = Shape::new(vec![batch, sequence, embedding_dim]) else {
            return STATUS_INVALID_ARGUMENT;
        };
        if shape.numel() != output_length {
            return STATUS_INVALID_ARGUMENT;
        }

        let mut values = vec![0.0; output_length];
        if embedding_f32(
            token_ids,
            weight.as_slice(),
            &mut values,
            vocab_size,
            embedding_dim,
        )
        .is_err()
        {
            return STATUS_INVALID_ARGUMENT;
        }

        let Ok(tensor) = Tensor::from_vec(values, shape) else {
            return STATUS_INVALID_ARGUMENT;
        };
        let handle = Box::into_raw(Box::new(TransformerTensor::new(tensor)));
        // SAFETY: `output` is valid by the caller contract and written once.
        unsafe { output.write(handle) };
        STATUS_OK
    }))
    .unwrap_or(STATUS_PANIC)
}

/// Projects the last dimension of `input` using a resident rank-2 weight and
/// an optional resident rank-1 bias, returning a new owned Tensor.
///
/// # Safety
///
/// Every non-null input must be a live handle for the complete call. `bias`
/// may be null. `output` must be writable for one handle pointer and must not
/// alias an input handle. Inputs remain alive and unchanged on success/error.
#[no_mangle]
pub unsafe extern "C" fn transformer_tensor_linear_last_dim(
    input: *const TransformerTensor,
    weight: *const TransformerTensor,
    bias: *const TransformerTensor,
    output: *mut *mut TransformerTensor,
) -> c_int {
    if output.is_null() {
        return STATUS_INVALID_ARGUMENT;
    }
    // SAFETY: `output` is non-null and writable by the caller contract.
    unsafe { output.write(ptr::null_mut()) };
    if input.is_null() || weight.is_null() {
        return STATUS_INVALID_ARGUMENT;
    }

    catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: Required pointers were checked and the caller guarantees live handles.
        let input = unsafe { &*input }.tensor();
        let weight = unsafe { &*weight }.tensor();
        let bias = if bias.is_null() {
            None
        } else {
            // SAFETY: A non-null optional bias is live by the caller contract.
            Some(unsafe { &*bias }.tensor())
        };
        if input.dtype() != DType::Float32
            || weight.dtype() != DType::Float32
            || input.rank() < 1
            || weight.rank() != 2
            || bias.is_some_and(|tensor| tensor.dtype() != DType::Float32 || tensor.rank() != 1)
        {
            return STATUS_INVALID_ARGUMENT;
        }
        let input_features = *input.shape().as_slice().last().unwrap_or(&0);
        let weight_shape = weight.shape().as_slice();
        let output_features = weight_shape[1];
        if input_features != weight_shape[0]
            || bias.is_some_and(|tensor| tensor.shape().as_slice() != [output_features])
        {
            return STATUS_INVALID_ARGUMENT;
        }
        let mut dimensions = input.shape().as_slice().to_vec();
        if let Some(last) = dimensions.last_mut() {
            *last = output_features;
        }
        let Ok(shape) = Shape::new(dimensions) else {
            return STATUS_INVALID_ARGUMENT;
        };
        let mut values = vec![0.0; shape.numel()];
        if linear_last_dim_f32(
            input.as_slice(),
            weight.as_slice(),
            bias.map(Tensor::as_slice),
            &mut values,
            input_features,
            output_features,
        )
        .is_err()
        {
            return STATUS_INVALID_ARGUMENT;
        }
        let Ok(tensor) = Tensor::from_vec(values, shape) else {
            return STATUS_INVALID_ARGUMENT;
        };
        let handle = Box::into_raw(Box::new(TransformerTensor::new(tensor)));
        // SAFETY: `output` is valid by the caller contract and written once.
        unsafe { output.write(handle) };
        STATUS_OK
    }))
    .unwrap_or(STATUS_PANIC)
}

/// Applies inference LayerNorm over the last dimension into a new owned Tensor.
#[no_mangle]
pub unsafe extern "C" fn transformer_tensor_layer_norm(
    input: *const TransformerTensor,
    weight: *const TransformerTensor,
    bias: *const TransformerTensor,
    epsilon: f32,
    output: *mut *mut TransformerTensor,
) -> c_int {
    if output.is_null() {
        return STATUS_INVALID_ARGUMENT;
    }
    // SAFETY: `output` is non-null and writable by the caller contract.
    unsafe { output.write(ptr::null_mut()) };
    if input.is_null() || weight.is_null() || bias.is_null() {
        return STATUS_INVALID_ARGUMENT;
    }

    catch_unwind(AssertUnwindSafe(|| {
        #[cfg(test)]
        PANIC_LAYER_NORM.with(|flag| {
            if flag.replace(false) {
                panic!("controlled LayerNorm ABI panic");
            }
        });
        // SAFETY: Required pointers were checked and must remain live for this call.
        let input = unsafe { &*input }.tensor();
        let weight = unsafe { &*weight }.tensor();
        let bias = unsafe { &*bias }.tensor();
        if input.dtype() != DType::Float32
            || weight.dtype() != DType::Float32
            || bias.dtype() != DType::Float32
            || input.rank() < 1
            || weight.rank() != 1
            || bias.rank() != 1
            || !epsilon.is_finite()
            || epsilon <= 0.0
            || input.strides() != &Strides::contiguous(input.shape())
            || weight.strides() != &Strides::contiguous(weight.shape())
            || bias.strides() != &Strides::contiguous(bias.shape())
        {
            return STATUS_INVALID_ARGUMENT;
        }
        let Some(&d) = input.shape().as_slice().last() else {
            return STATUS_INVALID_ARGUMENT;
        };
        if d == 0 || weight.shape().as_slice() != [d] || bias.shape().as_slice() != [d] {
            return STATUS_INVALID_ARGUMENT;
        }
        let shape = input.shape().clone();
        let mut values = vec![0.0; input.numel()];
        if layer_norm_f32(
            input.as_slice(),
            weight.as_slice(),
            bias.as_slice(),
            &mut values,
            d,
            epsilon,
        )
        .is_err()
        {
            return STATUS_INVALID_ARGUMENT;
        }
        let Ok(tensor) = Tensor::from_vec(values, shape) else {
            return STATUS_INVALID_ARGUMENT;
        };
        let handle = Box::into_raw(Box::new(TransformerTensor::new(tensor)));
        // SAFETY: Publication happens exactly once after complete validation/execution.
        unsafe { output.write(handle) };
        STATUS_OK
    }))
    .unwrap_or(STATUS_PANIC)
}

/// Applies the canonical tanh GELU elementwise into a new owned Tensor.
#[no_mangle]
pub unsafe extern "C" fn transformer_tensor_gelu(
    input: *const TransformerTensor,
    output: *mut *mut TransformerTensor,
) -> c_int {
    if output.is_null() {
        return STATUS_INVALID_ARGUMENT;
    }
    // SAFETY: `output` is non-null and writable by the caller contract.
    unsafe { output.write(ptr::null_mut()) };
    if input.is_null() {
        return STATUS_INVALID_ARGUMENT;
    }

    catch_unwind(AssertUnwindSafe(|| {
        #[cfg(test)]
        PANIC_GELU.with(|flag| {
            if flag.replace(false) {
                panic!("controlled GELU ABI panic");
            }
        });
        // SAFETY: The caller guarantees a live immutable handle.
        let input = unsafe { &*input }.tensor();
        if input.dtype() != DType::Float32
            || input.strides() != &Strides::contiguous(input.shape())
            || input.as_slice().len() != input.numel()
        {
            return STATUS_INVALID_ARGUMENT;
        }

        let shape = input.shape().clone();
        let mut values = vec![0.0; input.numel()];
        if gelu_f32(input.as_slice(), &mut values).is_err() {
            return STATUS_INVALID_ARGUMENT;
        }
        let Ok(tensor) = Tensor::from_vec(values, shape) else {
            return STATUS_INVALID_ARGUMENT;
        };
        let handle = Box::into_raw(Box::new(TransformerTensor::new(tensor)));
        // SAFETY: Publication happens once, only after successful execution.
        unsafe { output.write(handle) };
        STATUS_OK
    }))
    .unwrap_or(STATUS_PANIC)
}

/// Materializes the transpose of a rank-2 Float32 Tensor.
///
/// # Safety
///
/// `input` must point to a live handle for the duration of the call. `output`
/// must be writable for one handle pointer and must not alias the input handle.
/// The input handle remains valid and unchanged on success.
#[no_mangle]
pub unsafe extern "C" fn transformer_tensor_transpose(
    input: *const TransformerTensor,
    output: *mut *mut TransformerTensor,
) -> c_int {
    if output.is_null() {
        return STATUS_INVALID_ARGUMENT;
    }

    // SAFETY: `output` is non-null and writable by the caller contract.
    unsafe { output.write(ptr::null_mut()) };

    if input.is_null() {
        return STATUS_INVALID_ARGUMENT;
    }

    catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: The caller guarantees a live immutable handle.
        let input = unsafe { &*input }.tensor();
        if input.dtype() != DType::Float32 || input.rank() != 2 {
            return STATUS_INVALID_ARGUMENT;
        }

        let input_shape = input.shape().as_slice();
        let (rows, columns) = (input_shape[0], input_shape[1]);
        let Ok(shape) = Shape::new(vec![columns, rows]) else {
            return STATUS_INVALID_ARGUMENT;
        };
        let mut values = vec![0.0; shape.numel()];
        if transpose_f32(input.as_slice(), &mut values, rows, columns).is_err() {
            return STATUS_INVALID_ARGUMENT;
        }

        let Ok(tensor) = Tensor::from_vec(values, shape) else {
            return STATUS_INVALID_ARGUMENT;
        };
        let handle = Box::into_raw(Box::new(TransformerTensor::new(tensor)));

        // SAFETY: `output` is valid by the caller contract and written once.
        unsafe { output.write(handle) };
        STATUS_OK
    }))
    .unwrap_or(STATUS_PANIC)
}

/// Computes a numerically stable softmax over a non-empty rank-1 Tensor.
///
/// # Safety
///
/// `input` must point to a live handle for the duration of the call. `output`
/// must be writable for one handle pointer and must not alias the input handle.
/// The input handle remains valid and unchanged on success.
#[no_mangle]
pub unsafe extern "C" fn transformer_tensor_softmax(
    input: *const TransformerTensor,
    output: *mut *mut TransformerTensor,
) -> c_int {
    if output.is_null() {
        return STATUS_INVALID_ARGUMENT;
    }

    // SAFETY: `output` is non-null and writable by the caller contract.
    unsafe { output.write(ptr::null_mut()) };

    if input.is_null() {
        return STATUS_INVALID_ARGUMENT;
    }

    catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: The caller guarantees a live immutable handle.
        let input = unsafe { &*input }.tensor();
        if input.dtype() != DType::Float32 || input.rank() != 1 || input.is_empty() {
            return STATUS_INVALID_ARGUMENT;
        }

        let mut values = vec![0.0; input.numel()];
        if softmax_f32(input.as_slice(), &mut values).is_err() {
            return STATUS_INVALID_ARGUMENT;
        }

        let shape = input.shape().clone();
        let Ok(tensor) = Tensor::from_vec(values, shape) else {
            return STATUS_INVALID_ARGUMENT;
        };
        let handle = Box::into_raw(Box::new(TransformerTensor::new(tensor)));

        // SAFETY: `output` is valid by the caller contract and written once.
        unsafe { output.write(handle) };
        STATUS_OK
    }))
    .unwrap_or(STATUS_PANIC)
}

/// Computes a numerically stable softmax over the last dimension of a
/// contiguous, non-empty rank-N Float32 Tensor and returns a new owned Tensor.
///
/// # Safety
///
/// `input` must point to a live handle for the duration of the call. `output`
/// must be writable for one handle pointer and must not alias the input handle.
/// The input handle remains valid and unchanged on success.
#[no_mangle]
pub unsafe extern "C" fn transformer_tensor_softmax_last_dim(
    input: *const TransformerTensor,
    output: *mut *mut TransformerTensor,
) -> c_int {
    if output.is_null() {
        return STATUS_INVALID_ARGUMENT;
    }

    // SAFETY: `output` is non-null and writable by the caller contract.
    unsafe { output.write(ptr::null_mut()) };

    if input.is_null() {
        return STATUS_INVALID_ARGUMENT;
    }

    catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: The caller guarantees a live immutable handle.
        let input = unsafe { &*input }.tensor();
        if input.dtype() != DType::Float32 || input.rank() == 0 || input.is_empty() {
            return STATUS_INVALID_ARGUMENT;
        }

        let Some(&last_dim) = input.shape().as_slice().last() else {
            return STATUS_INVALID_ARGUMENT;
        };
        if last_dim == 0 {
            return STATUS_INVALID_ARGUMENT;
        }

        let mut values = vec![0.0; input.numel()];
        if softmax_last_dim_f32(input.as_slice(), &mut values, last_dim).is_err() {
            return STATUS_INVALID_ARGUMENT;
        }

        let shape = input.shape().clone();
        let Ok(tensor) = Tensor::from_vec(values, shape) else {
            return STATUS_INVALID_ARGUMENT;
        };
        let handle = Box::into_raw(Box::new(TransformerTensor::new(tensor)));

        // SAFETY: `output` is valid by the caller contract and written once.
        unsafe { output.write(handle) };
        STATUS_OK
    }))
    .unwrap_or(STATUS_PANIC)
}

/// Computes non-causal multi-head self-attention into a new owned Tensor.
///
/// # Safety
///
/// Tensor pointers must identify live immutable handles for the complete call.
/// A non-null mask must be readable for `mask_length` bytes. `output` must be
/// writable for one handle pointer and must not alias any input pointer.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn transformer_tensor_multi_head_attention(
    input: *const TransformerTensor,
    q_weight: *const TransformerTensor,
    k_weight: *const TransformerTensor,
    v_weight: *const TransformerTensor,
    out_weight: *const TransformerTensor,
    heads: usize,
    mask: *const u8,
    mask_length: usize,
    output: *mut *mut TransformerTensor,
) -> c_int {
    if output.is_null() {
        return STATUS_INVALID_ARGUMENT;
    }
    // SAFETY: `output` is non-null and writable by the caller contract.
    unsafe { output.write(ptr::null_mut()) };
    if input.is_null()
        || q_weight.is_null()
        || k_weight.is_null()
        || v_weight.is_null()
        || out_weight.is_null()
        || (mask.is_null() && mask_length != 0)
    {
        return STATUS_INVALID_ARGUMENT;
    }

    catch_unwind(AssertUnwindSafe(|| {
        #[cfg(test)]
        PANIC_ATTENTION.with(|flag| {
            if flag.replace(false) {
                panic!("injected attention panic");
            }
        });

        // SAFETY: Required pointers were checked and are live by caller contract.
        let input = unsafe { &*input }.tensor();
        let q_weight = unsafe { &*q_weight }.tensor();
        let k_weight = unsafe { &*k_weight }.tensor();
        let v_weight = unsafe { &*v_weight }.tensor();
        let out_weight = unsafe { &*out_weight }.tensor();
        let weights = [q_weight, k_weight, v_weight, out_weight];
        if input.dtype() != DType::Float32
            || input.rank() != 3
            || input.strides() != &Strides::contiguous(input.shape())
            || weights.iter().any(|weight| {
                weight.dtype() != DType::Float32
                    || weight.rank() != 2
                    || weight.strides() != &Strides::contiguous(weight.shape())
            })
        {
            return STATUS_INVALID_ARGUMENT;
        }
        let input_shape = input.shape().as_slice();
        let (batch, sequence, dimensions) = (input_shape[0], input_shape[1], input_shape[2]);
        if dimensions == 0
            || heads == 0
            || dimensions % heads != 0
            || weights
                .iter()
                .any(|weight| weight.shape().as_slice() != [dimensions, dimensions])
        {
            return STATUS_INVALID_ARGUMENT;
        }
        let Some(expected_mask_length) = batch.checked_mul(sequence) else {
            return STATUS_INVALID_ARGUMENT;
        };
        let mask = if mask.is_null() {
            None
        } else {
            if mask_length != expected_mask_length
                || mask_length > isize::MAX as usize / size_of::<u8>()
            {
                return STATUS_INVALID_ARGUMENT;
            }
            // SAFETY: Non-null pointer and representable exact length were validated.
            Some(unsafe { std::slice::from_raw_parts(mask, mask_length) })
        };

        let mut values = vec![0.0; input.numel()];
        if multi_head_attention_f32(
            input.as_slice(),
            q_weight.as_slice(),
            k_weight.as_slice(),
            v_weight.as_slice(),
            out_weight.as_slice(),
            mask,
            &mut values,
            batch,
            sequence,
            dimensions,
            heads,
        )
        .is_err()
        {
            return STATUS_INVALID_ARGUMENT;
        }
        let Ok(tensor) = Tensor::from_vec(values, input.shape().clone()) else {
            return STATUS_INVALID_ARGUMENT;
        };
        let handle = Box::into_raw(Box::new(TransformerTensor::new(tensor)));
        // SAFETY: `output` is writable and published once after complete success.
        unsafe { output.write(handle) };
        STATUS_OK
    }))
    .unwrap_or(STATUS_PANIC)
}

#[cfg(test)]
mod tests {
    use std::ptr::{null, null_mut};

    use super::*;
    use crate::ffi::{
        transformer_matmul_f32, transformer_softmax_f32, transformer_tensor_add_f32,
        transformer_transpose_f32,
    };

    unsafe fn destroy(handle: *mut TransformerTensor) {
        // SAFETY: Test handles are destroyed exactly once.
        unsafe { transformer_tensor_destroy(handle) };
    }

    unsafe fn create(data: &[f32], shape: &[usize]) -> *mut TransformerTensor {
        let mut handle = null_mut();
        // SAFETY: Test slices match their declared shape and remain live for the call.
        assert_eq!(
            unsafe {
                transformer_tensor_create_f32(
                    data.as_ptr(),
                    shape.as_ptr(),
                    shape.len(),
                    &mut handle,
                )
            },
            STATUS_OK
        );
        handle
    }

    #[test]
    fn tensor_attention_executes_masks_and_preserves_inputs() {
        // SAFETY: Handles and mask remain live for the calls and are destroyed once.
        unsafe {
            let input = create(&[1.0, 0.0, 0.0, 1.0], &[1, 2, 2]);
            let identity = [1.0, 0.0, 0.0, 1.0];
            let q = create(&identity, &[2, 2]);
            let k = create(&identity, &[2, 2]);
            let v = create(&identity, &[2, 2]);
            let out = create(&identity, &[2, 2]);
            let mask = [1u8, 0];
            let mut output = null_mut();
            assert_eq!(
                transformer_tensor_multi_head_attention(
                    input,
                    q,
                    k,
                    v,
                    out,
                    1,
                    mask.as_ptr(),
                    mask.len(),
                    &mut output,
                ),
                STATUS_OK
            );
            assert_eq!((*output).tensor().shape().as_slice(), &[1, 2, 2]);
            assert_eq!((*output).tensor().as_slice(), &[1.0, 0.0, 1.0, 0.0]);
            assert_eq!((*input).tensor().as_slice(), &[1.0, 0.0, 0.0, 1.0]);
            assert_eq!((*q).tensor().as_slice(), identity);
            for handle in [input, q, k, v, out, output] {
                destroy(handle);
            }
        }
    }

    #[test]
    fn tensor_attention_rejects_invalid_mask_without_publication_and_supports_empty() {
        // SAFETY: Handles and buffers remain live and are destroyed once.
        unsafe {
            let identity = [1.0, 0.0, 0.0, 1.0];
            let input = create(&[1.0, 2.0], &[1, 1, 2]);
            let empty = create(&[], &[1, 0, 2]);
            let q = create(&identity, &[2, 2]);
            let k = create(&identity, &[2, 2]);
            let v = create(&identity, &[2, 2]);
            let out = create(&identity, &[2, 2]);
            let invalid = [2u8];
            let mut output = 1usize as *mut TransformerTensor;
            assert_eq!(
                transformer_tensor_multi_head_attention(
                    input,
                    q,
                    k,
                    v,
                    out,
                    1,
                    invalid.as_ptr(),
                    invalid.len(),
                    &mut output,
                ),
                STATUS_INVALID_ARGUMENT
            );
            assert!(output.is_null());
            let empty_mask: [u8; 0] = [];
            assert_eq!(
                transformer_tensor_multi_head_attention(
                    empty,
                    q,
                    k,
                    v,
                    out,
                    1,
                    empty_mask.as_ptr(),
                    0,
                    &mut output,
                ),
                STATUS_OK
            );
            assert_eq!((*output).tensor().shape().as_slice(), &[1, 0, 2]);
            destroy(output);
            for handle in [input, empty, q, k, v, out] {
                destroy(handle);
            }
        }
    }

    #[test]
    fn tensor_attention_contains_panic_and_recovers() {
        // SAFETY: Handles remain live and are destroyed exactly once.
        unsafe {
            let input = create(&[1.0], &[1, 1, 1]);
            let weight = create(&[1.0], &[1, 1]);
            let mut output = null_mut();
            PANIC_ATTENTION.with(|flag| flag.set(true));
            assert_eq!(
                transformer_tensor_multi_head_attention(
                    input,
                    weight,
                    weight,
                    weight,
                    weight,
                    1,
                    null(),
                    0,
                    &mut output,
                ),
                STATUS_PANIC
            );
            assert!(output.is_null());
            assert_eq!(
                transformer_tensor_multi_head_attention(
                    input,
                    weight,
                    weight,
                    weight,
                    weight,
                    1,
                    null(),
                    0,
                    &mut output,
                ),
                STATUS_OK
            );
            destroy(output);
            destroy(input);
            destroy(weight);
        }
    }

    #[test]
    fn tensor_linear_projects_last_dimension_and_preserves_inputs() {
        // SAFETY: All handles are live and destroyed exactly once below.
        unsafe {
            let input = create(&[1.0, 2.0, 3.0, 4.0], &[2, 1, 2]);
            let weight = create(&[1.0, -1.0, 2.0, 0.5, 3.0, 2.0], &[2, 3]);
            let bias = create(&[0.25, -0.5, 1.0], &[3]);
            let mut output = null_mut();
            assert_eq!(
                transformer_tensor_linear_last_dim(input, weight, bias, &mut output),
                STATUS_OK
            );
            assert_eq!((*output).tensor().shape().as_slice(), &[2, 1, 3]);
            assert_eq!(
                (*output).tensor().as_slice(),
                &[2.25, 4.5, 7.0, 5.25, 8.5, 15.0]
            );
            assert_eq!((*input).tensor().as_slice(), &[1.0, 2.0, 3.0, 4.0]);
            destroy(output);
            destroy(bias);
            destroy(weight);
            destroy(input);
        }
    }

    #[test]
    fn tensor_embedding_gathers_batches_and_preserves_weight() {
        // SAFETY: All buffers and handles are live and destroyed exactly once.
        unsafe {
            let weight_data = [0.0, 0.1, 0.2, 1.0, 1.1, 1.2, 2.0, 2.1, 2.2, 3.0, 3.1, 3.2];
            let weight = create(&weight_data, &[4, 3]);
            let ids = [3_i64, 0, 2, 1];
            let mut output = null_mut();

            assert_eq!(
                transformer_tensor_embedding(ids.as_ptr(), 2, 2, weight, &mut output),
                STATUS_OK
            );
            assert_eq!((*output).tensor().shape().as_slice(), &[2, 2, 3]);
            assert_eq!(
                (*output).tensor().as_slice(),
                [3.0, 3.1, 3.2, 0.0, 0.1, 0.2, 2.0, 2.1, 2.2, 1.0, 1.1, 1.2]
            );
            assert_eq!((*weight).tensor().as_slice(), weight_data);

            destroy(output);
            destroy(weight);
        }
    }

    #[test]
    fn tensor_embedding_supports_all_approved_empty_shapes() {
        // SAFETY: Empty inputs are never read and handles are destroyed once.
        unsafe {
            let weight = create(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
            for (batch, sequence) in [(0, 3), (2, 0), (0, 0)] {
                let mut output = null_mut();
                assert_eq!(
                    transformer_tensor_embedding(null(), batch, sequence, weight, &mut output),
                    STATUS_OK
                );
                assert_eq!((*output).tensor().shape().as_slice(), &[batch, sequence, 2]);
                assert!((*output).tensor().is_empty());
                destroy(output);
            }
            destroy(weight);
        }
    }

    #[test]
    fn tensor_embedding_rejects_ids_without_publishing_or_consuming_weight() {
        // SAFETY: Inputs are valid for each call and the weight is destroyed once.
        unsafe {
            let weight = create(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
            for ids in [[0_i64, -1], [0_i64, 2]] {
                let mut output = usize::MAX as *mut TransformerTensor;
                assert_eq!(
                    transformer_tensor_embedding(ids.as_ptr(), 1, 2, weight, &mut output),
                    STATUS_INVALID_ARGUMENT
                );
                assert!(output.is_null());
                assert_eq!((*weight).tensor().as_slice(), &[1.0, 2.0, 3.0, 4.0]);
            }

            let valid = [1_i64];
            let mut output = null_mut();
            assert_eq!(
                transformer_tensor_embedding(valid.as_ptr(), 1, 1, weight, &mut output),
                STATUS_OK
            );
            destroy(output);
            destroy(weight);
        }
    }

    #[test]
    fn tensor_embedding_rejects_invalid_metadata_and_overflow() {
        // SAFETY: Invalid arguments are rejected before unsafe reads.
        unsafe {
            let rank_one_weight = create(&[1.0, 2.0], &[2]);
            let mut output = usize::MAX as *mut TransformerTensor;
            assert_eq!(
                transformer_tensor_embedding(null(), 0, 1, rank_one_weight, &mut output),
                STATUS_INVALID_ARGUMENT
            );
            assert!(output.is_null());
            assert_eq!(
                transformer_tensor_embedding(null(), usize::MAX, 2, rank_one_weight, &mut output,),
                STATUS_INVALID_ARGUMENT
            );
            assert!(output.is_null());
            assert_eq!(
                transformer_tensor_embedding(null(), 1, 1, rank_one_weight, null_mut()),
                STATUS_INVALID_ARGUMENT
            );
            destroy(rank_one_weight);
        }
    }

    #[test]
    fn tensor_linear_rejects_incompatible_shapes_without_consuming_inputs() {
        // SAFETY: All handles are live and destroyed exactly once below.
        unsafe {
            let input = create(&[1.0, 2.0, 3.0], &[3]);
            let weight = create(&[1.0, 0.0, 0.0, 1.0], &[2, 2]);
            let mut output = null_mut();
            assert_eq!(
                transformer_tensor_linear_last_dim(input, weight, null(), &mut output),
                STATUS_INVALID_ARGUMENT
            );
            assert!(output.is_null());
            assert_eq!((*weight).tensor().as_slice(), &[1.0, 0.0, 0.0, 1.0]);
            destroy(weight);
            destroy(input);
        }
    }

    #[test]
    fn creates_tensor_and_reads_metadata() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let shape = [2, 3];
        let mut handle = null_mut();

        // SAFETY: All buffers have the lengths declared to the ABI.
        let status =
            unsafe { transformer_tensor_create_f32(data.as_ptr(), shape.as_ptr(), 2, &mut handle) };
        assert_eq!(status, STATUS_OK);
        assert!(!handle.is_null());

        let mut rank = 0;
        let mut numel = 0;
        let mut dtype = -1;
        let mut copied_shape = [0; 2];
        // SAFETY: Handle is live and outputs have sufficient capacity.
        assert_eq!(
            unsafe { transformer_tensor_rank(handle, &mut rank) },
            STATUS_OK
        );
        assert_eq!(
            unsafe { transformer_tensor_numel(handle, &mut numel) },
            STATUS_OK
        );
        assert_eq!(
            unsafe { transformer_tensor_dtype(handle, &mut dtype) },
            STATUS_OK
        );
        assert_eq!(
            unsafe { transformer_tensor_shape(handle, copied_shape.as_mut_ptr(), 2) },
            STATUS_OK
        );

        assert_eq!(rank, 2);
        assert_eq!(numel, 6);
        assert_eq!(dtype, DTYPE_FLOAT32);
        assert_eq!(copied_shape, [2, 3]);
        unsafe { destroy(handle) };
    }

    #[test]
    fn supports_scalar_shape() {
        let data = [42.0];
        let mut handle = null_mut();

        // SAFETY: Rank zero requires no shape buffer and data has one scalar.
        assert_eq!(
            unsafe { transformer_tensor_create_f32(data.as_ptr(), null(), 0, &mut handle) },
            STATUS_OK
        );
        assert_eq!(
            unsafe { transformer_tensor_shape(handle, null_mut(), 0) },
            STATUS_OK
        );
        unsafe { destroy(handle) };
    }

    #[test]
    fn supports_zero_sized_shape_with_null_data() {
        let shape = [2, 0, 3];
        let mut handle = null_mut();

        // SAFETY: Shape has zero elements, so no data is read.
        assert_eq!(
            unsafe { transformer_tensor_create_f32(null(), shape.as_ptr(), 3, &mut handle) },
            STATUS_OK
        );
        unsafe { destroy(handle) };
    }

    #[test]
    fn rejects_numel_overflow_and_leaves_output_null() {
        let shape = [usize::MAX, 2];
        let mut handle = null_mut();

        // SAFETY: Shape buffer is valid; overflow is rejected before data use.
        assert_eq!(
            unsafe { transformer_tensor_create_f32(null(), shape.as_ptr(), 2, &mut handle) },
            STATUS_INVALID_ARGUMENT
        );
        assert!(handle.is_null());
    }

    #[test]
    fn reports_insufficient_shape_capacity_without_writing() {
        let data = [1.0, 2.0];
        let shape = [2];
        let mut handle = null_mut();
        // SAFETY: Input buffers are valid.
        assert_eq!(
            unsafe { transformer_tensor_create_f32(data.as_ptr(), shape.as_ptr(), 1, &mut handle) },
            STATUS_OK
        );

        // SAFETY: Handle is live; capacity zero causes no output write.
        assert_eq!(
            unsafe { transformer_tensor_shape(handle, null_mut(), 0) },
            STATUS_INSUFFICIENT_BUFFER
        );
        unsafe { destroy(handle) };
    }

    #[test]
    fn rejects_null_arguments() {
        let mut handle = null_mut();
        // SAFETY: Functions reject null before dereferencing it.
        assert_eq!(
            unsafe { transformer_tensor_create_f32(null(), null(), 1, &mut handle) },
            STATUS_INVALID_ARGUMENT
        );
        assert_eq!(
            unsafe { transformer_tensor_create_f32(null(), null(), 0, null_mut()) },
            STATUS_INVALID_ARGUMENT
        );
        assert_eq!(
            unsafe { transformer_tensor_rank(null(), null_mut()) },
            STATUS_INVALID_ARGUMENT
        );
        assert_eq!(
            unsafe { transformer_tensor_dtype(null(), null_mut()) },
            STATUS_INVALID_ARGUMENT
        );
        assert_eq!(
            unsafe { transformer_tensor_numel(null(), null_mut()) },
            STATUS_INVALID_ARGUMENT
        );
        assert_eq!(
            unsafe { transformer_tensor_shape(null(), null_mut(), 0) },
            STATUS_INVALID_ARGUMENT
        );
    }

    #[test]
    fn destroy_accepts_null() {
        // SAFETY: Null destroy is explicitly a no-op.
        unsafe { transformer_tensor_destroy(null_mut()) };
    }

    #[test]
    fn copies_data_without_consuming_handle() {
        let data = [1.5, -2.0, 3.25, 4.0];
        let shape = [2, 2];
        let mut handle = null_mut();
        // SAFETY: Input buffers are valid for the declared shape.
        assert_eq!(
            unsafe { transformer_tensor_create_f32(data.as_ptr(), shape.as_ptr(), 2, &mut handle) },
            STATUS_OK
        );

        let mut output = [0.0; 4];
        // SAFETY: Output has exact capacity and handle is live.
        assert_eq!(
            unsafe { transformer_tensor_copy_data_f32(handle, output.as_mut_ptr(), output.len()) },
            STATUS_OK
        );
        assert_eq!(output, data);

        let mut numel = 0;
        // SAFETY: A successful copy does not consume the live handle.
        assert_eq!(
            unsafe { transformer_tensor_numel(handle, &mut numel) },
            STATUS_OK
        );
        assert_eq!(numel, 4);
        unsafe { destroy(handle) };
    }

    #[test]
    fn copies_scalar_data() {
        let data = [42.0];
        let mut handle = null_mut();
        // SAFETY: Rank zero represents one scalar value.
        assert_eq!(
            unsafe { transformer_tensor_create_f32(data.as_ptr(), null(), 0, &mut handle) },
            STATUS_OK
        );

        let mut output = [0.0];
        // SAFETY: Output has capacity for the scalar.
        assert_eq!(
            unsafe { transformer_tensor_copy_data_f32(handle, output.as_mut_ptr(), 1) },
            STATUS_OK
        );
        assert_eq!(output, [42.0]);
        unsafe { destroy(handle) };
    }

    #[test]
    fn copies_empty_tensor_without_output_buffer() {
        let shape = [2, 0, 3];
        let mut handle = null_mut();
        // SAFETY: Empty shape requires no data buffer.
        assert_eq!(
            unsafe { transformer_tensor_create_f32(null(), shape.as_ptr(), 3, &mut handle) },
            STATUS_OK
        );

        // SAFETY: Empty Tensor causes no output write.
        assert_eq!(
            unsafe { transformer_tensor_copy_data_f32(handle, null_mut(), 0) },
            STATUS_OK
        );
        unsafe { destroy(handle) };
    }

    #[test]
    fn rejects_insufficient_copy_capacity_without_writing() {
        let data = [1.0, 2.0];
        let shape = [2];
        let mut handle = null_mut();
        // SAFETY: Input buffers are valid.
        assert_eq!(
            unsafe { transformer_tensor_create_f32(data.as_ptr(), shape.as_ptr(), 1, &mut handle) },
            STATUS_OK
        );

        let mut output = [99.0];
        // SAFETY: The function detects insufficient capacity before writing.
        assert_eq!(
            unsafe { transformer_tensor_copy_data_f32(handle, output.as_mut_ptr(), 1) },
            STATUS_INSUFFICIENT_BUFFER
        );
        assert_eq!(output, [99.0]);
        unsafe { destroy(handle) };
    }

    #[test]
    fn copy_rejects_null_arguments_for_non_empty_tensor() {
        let data = [1.0];
        let shape = [1];
        let mut handle = null_mut();
        // SAFETY: Input buffers are valid.
        assert_eq!(
            unsafe { transformer_tensor_create_f32(data.as_ptr(), shape.as_ptr(), 1, &mut handle) },
            STATUS_OK
        );

        // SAFETY: Null output is rejected before a write.
        assert_eq!(
            unsafe { transformer_tensor_copy_data_f32(handle, null_mut(), 1) },
            STATUS_INVALID_ARGUMENT
        );
        // SAFETY: Null handle is rejected before dereference.
        assert_eq!(
            unsafe { transformer_tensor_copy_data_f32(null(), null_mut(), 0) },
            STATUS_INVALID_ARGUMENT
        );
        unsafe { destroy(handle) };
    }

    #[test]
    fn tensor_add_matches_buffer_api_and_preserves_inputs() {
        let a_data = [1.0, 2.0, 3.0];
        let b_data = [10.0, 20.0, 30.0];
        let shape = [3];
        let mut a = null_mut();
        let mut b = null_mut();
        let mut output = null_mut();
        // SAFETY: All input buffers are valid for shape [3].
        assert_eq!(
            unsafe { transformer_tensor_create_f32(a_data.as_ptr(), shape.as_ptr(), 1, &mut a) },
            STATUS_OK
        );
        assert_eq!(
            unsafe { transformer_tensor_create_f32(b_data.as_ptr(), shape.as_ptr(), 1, &mut b) },
            STATUS_OK
        );

        // SAFETY: Both handles are live and output is writable.
        assert_eq!(
            unsafe { transformer_tensor_add(a, b, &mut output) },
            STATUS_OK
        );
        assert!(!output.is_null());

        let mut tensor_result = [0.0; 3];
        let mut buffer_result = [0.0; 3];
        // SAFETY: Output buffers have exact capacity and all handles/buffers are valid.
        assert_eq!(
            unsafe {
                transformer_tensor_copy_data_f32(
                    output,
                    tensor_result.as_mut_ptr(),
                    tensor_result.len(),
                )
            },
            STATUS_OK
        );
        assert_eq!(
            unsafe {
                transformer_tensor_add_f32(
                    a_data.as_ptr(),
                    b_data.as_ptr(),
                    buffer_result.as_mut_ptr(),
                    buffer_result.len(),
                )
            },
            STATUS_OK
        );
        assert_eq!(tensor_result, buffer_result);
        assert_eq!(tensor_result, [11.0, 22.0, 33.0]);

        let mut a_after = [0.0; 3];
        let mut b_after = [0.0; 3];
        // SAFETY: Successful add leaves both input handles live and unchanged.
        assert_eq!(
            unsafe { transformer_tensor_copy_data_f32(a, a_after.as_mut_ptr(), a_after.len()) },
            STATUS_OK
        );
        assert_eq!(
            unsafe { transformer_tensor_copy_data_f32(b, b_after.as_mut_ptr(), b_after.len()) },
            STATUS_OK
        );
        assert_eq!(a_after, a_data);
        assert_eq!(b_after, b_data);

        unsafe { destroy(output) };
        unsafe { destroy(a) };
        unsafe { destroy(b) };
    }

    #[test]
    fn tensor_add_rejects_different_shapes_with_same_numel() {
        let data = [1.0; 6];
        let a_shape = [2, 3];
        let b_shape = [3, 2];
        let mut a = null_mut();
        let mut b = null_mut();
        let mut output = null_mut();
        // SAFETY: Input buffers are valid for their declared shapes.
        assert_eq!(
            unsafe { transformer_tensor_create_f32(data.as_ptr(), a_shape.as_ptr(), 2, &mut a) },
            STATUS_OK
        );
        assert_eq!(
            unsafe { transformer_tensor_create_f32(data.as_ptr(), b_shape.as_ptr(), 2, &mut b) },
            STATUS_OK
        );

        // SAFETY: Handles are live; incompatible shapes are rejected.
        assert_eq!(
            unsafe { transformer_tensor_add(a, b, &mut output) },
            STATUS_INVALID_ARGUMENT
        );
        assert!(output.is_null());
        unsafe { destroy(a) };
        unsafe { destroy(b) };
    }

    #[test]
    fn tensor_add_rejects_broadcasting() {
        let a_data = [1.0; 6];
        let b_data = [1.0; 3];
        let a_shape = [2, 3];
        let b_shape = [3];
        let mut a = null_mut();
        let mut b = null_mut();
        let mut output = null_mut();
        // SAFETY: Input buffers are valid for their declared shapes.
        assert_eq!(
            unsafe { transformer_tensor_create_f32(a_data.as_ptr(), a_shape.as_ptr(), 2, &mut a) },
            STATUS_OK
        );
        assert_eq!(
            unsafe { transformer_tensor_create_f32(b_data.as_ptr(), b_shape.as_ptr(), 1, &mut b) },
            STATUS_OK
        );

        // SAFETY: Handles are live; broadcasting is intentionally unsupported.
        assert_eq!(
            unsafe { transformer_tensor_add(a, b, &mut output) },
            STATUS_INVALID_ARGUMENT
        );
        assert!(output.is_null());
        unsafe { destroy(a) };
        unsafe { destroy(b) };
    }

    #[test]
    fn tensor_add_supports_scalars_and_empty_tensors() {
        let scalar_a = [2.0];
        let scalar_b = [5.0];
        let mut a = null_mut();
        let mut b = null_mut();
        let mut scalar_output = null_mut();
        // SAFETY: Rank zero inputs each contain one scalar.
        assert_eq!(
            unsafe { transformer_tensor_create_f32(scalar_a.as_ptr(), null(), 0, &mut a) },
            STATUS_OK
        );
        assert_eq!(
            unsafe { transformer_tensor_create_f32(scalar_b.as_ptr(), null(), 0, &mut b) },
            STATUS_OK
        );
        // SAFETY: Scalar handles are live and output is writable.
        assert_eq!(
            unsafe { transformer_tensor_add(a, b, &mut scalar_output) },
            STATUS_OK
        );
        let mut value = [0.0];
        // SAFETY: Output buffer has capacity for one scalar.
        assert_eq!(
            unsafe { transformer_tensor_copy_data_f32(scalar_output, value.as_mut_ptr(), 1) },
            STATUS_OK
        );
        assert_eq!(value, [7.0]);

        let empty_shape = [2, 0, 3];
        let mut empty_a = null_mut();
        let mut empty_b = null_mut();
        let mut empty_output = null_mut();
        // SAFETY: Empty shapes read no data.
        assert_eq!(
            unsafe { transformer_tensor_create_f32(null(), empty_shape.as_ptr(), 3, &mut empty_a) },
            STATUS_OK
        );
        assert_eq!(
            unsafe { transformer_tensor_create_f32(null(), empty_shape.as_ptr(), 3, &mut empty_b) },
            STATUS_OK
        );
        // SAFETY: Empty handles are live and identically shaped.
        assert_eq!(
            unsafe { transformer_tensor_add(empty_a, empty_b, &mut empty_output) },
            STATUS_OK
        );
        let mut numel = 1;
        // SAFETY: Result handle is live.
        assert_eq!(
            unsafe { transformer_tensor_numel(empty_output, &mut numel) },
            STATUS_OK
        );
        assert_eq!(numel, 0);

        unsafe { destroy(scalar_output) };
        unsafe { destroy(a) };
        unsafe { destroy(b) };
        unsafe { destroy(empty_output) };
        unsafe { destroy(empty_a) };
        unsafe { destroy(empty_b) };
    }

    #[test]
    fn tensor_add_rejects_null_arguments_and_clears_output() {
        let mut output = std::ptr::NonNull::<TransformerTensor>::dangling().as_ptr();

        // SAFETY: Null handles are rejected before dereference.
        assert_eq!(
            unsafe { transformer_tensor_add(null(), null(), &mut output) },
            STATUS_INVALID_ARGUMENT
        );
        assert!(output.is_null());
        // SAFETY: Null output is rejected before any write.
        assert_eq!(
            unsafe { transformer_tensor_add(null(), null(), null_mut()) },
            STATUS_INVALID_ARGUMENT
        );
    }

    #[test]
    fn tensor_matmul_matches_manual_result_and_buffer_api() {
        let a_data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b_data = [7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let a_shape = [2, 3];
        let b_shape = [3, 2];
        let mut a = null_mut();
        let mut b = null_mut();
        let mut output = null_mut();
        // SAFETY: Input buffers match their declared matrix shapes.
        assert_eq!(
            unsafe { transformer_tensor_create_f32(a_data.as_ptr(), a_shape.as_ptr(), 2, &mut a) },
            STATUS_OK
        );
        assert_eq!(
            unsafe { transformer_tensor_create_f32(b_data.as_ptr(), b_shape.as_ptr(), 2, &mut b) },
            STATUS_OK
        );

        // SAFETY: Both matrix handles are live and output is writable.
        assert_eq!(
            unsafe { transformer_tensor_matmul(a, b, &mut output) },
            STATUS_OK
        );

        let mut tensor_result = [0.0; 4];
        let mut buffer_result = [0.0; 4];
        let mut output_shape = [0; 2];
        // SAFETY: All output buffers have exact capacity.
        assert_eq!(
            unsafe {
                transformer_tensor_copy_data_f32(
                    output,
                    tensor_result.as_mut_ptr(),
                    tensor_result.len(),
                )
            },
            STATUS_OK
        );
        assert_eq!(
            unsafe { transformer_tensor_shape(output, output_shape.as_mut_ptr(), 2) },
            STATUS_OK
        );
        assert_eq!(
            unsafe {
                transformer_matmul_f32(
                    a_data.as_ptr(),
                    b_data.as_ptr(),
                    buffer_result.as_mut_ptr(),
                    2,
                    3,
                    2,
                )
            },
            STATUS_OK
        );

        assert_eq!(tensor_result, [58.0, 64.0, 139.0, 154.0]);
        assert_eq!(tensor_result, buffer_result);
        assert_eq!(output_shape, [2, 2]);

        let mut a_after = [0.0; 6];
        let mut b_after = [0.0; 6];
        // SAFETY: Matmul leaves both input handles valid and unchanged.
        assert_eq!(
            unsafe { transformer_tensor_copy_data_f32(a, a_after.as_mut_ptr(), a_after.len()) },
            STATUS_OK
        );
        assert_eq!(
            unsafe { transformer_tensor_copy_data_f32(b, b_after.as_mut_ptr(), b_after.len()) },
            STATUS_OK
        );
        assert_eq!(a_after, a_data);
        assert_eq!(b_after, b_data);

        unsafe { destroy(output) };
        unsafe { destroy(a) };
        unsafe { destroy(b) };
    }

    #[test]
    fn tensor_matmul_handles_zero_inner_dimension() {
        let a_shape = [2, 0];
        let b_shape = [0, 2];
        let mut a = null_mut();
        let mut b = null_mut();
        let mut output = null_mut();
        // SAFETY: Zero-sized inputs require no data buffers.
        assert_eq!(
            unsafe { transformer_tensor_create_f32(null(), a_shape.as_ptr(), 2, &mut a) },
            STATUS_OK
        );
        assert_eq!(
            unsafe { transformer_tensor_create_f32(null(), b_shape.as_ptr(), 2, &mut b) },
            STATUS_OK
        );
        // SAFETY: Inner dimensions match at zero.
        assert_eq!(
            unsafe { transformer_tensor_matmul(a, b, &mut output) },
            STATUS_OK
        );

        let mut values = [1.0; 4];
        let mut shape = [0; 2];
        // SAFETY: Result buffers have exact capacity.
        assert_eq!(
            unsafe { transformer_tensor_copy_data_f32(output, values.as_mut_ptr(), 4) },
            STATUS_OK
        );
        assert_eq!(
            unsafe { transformer_tensor_shape(output, shape.as_mut_ptr(), 2) },
            STATUS_OK
        );
        assert_eq!(values, [0.0; 4]);
        assert_eq!(shape, [2, 2]);

        unsafe { destroy(output) };
        unsafe { destroy(a) };
        unsafe { destroy(b) };
    }

    #[test]
    fn tensor_matmul_rejects_non_matrices_and_incompatible_dimensions() {
        let vector_data = [1.0, 2.0, 3.0];
        let vector_shape = [3];
        let matrix_data = [1.0; 6];
        let matrix_shape = [2, 3];
        let other_shape = [2, 3];
        let mut vector = null_mut();
        let mut matrix = null_mut();
        let mut other = null_mut();
        let mut output = null_mut();
        // SAFETY: All buffers match their declared shapes.
        assert_eq!(
            unsafe {
                transformer_tensor_create_f32(
                    vector_data.as_ptr(),
                    vector_shape.as_ptr(),
                    1,
                    &mut vector,
                )
            },
            STATUS_OK
        );
        assert_eq!(
            unsafe {
                transformer_tensor_create_f32(
                    matrix_data.as_ptr(),
                    matrix_shape.as_ptr(),
                    2,
                    &mut matrix,
                )
            },
            STATUS_OK
        );
        assert_eq!(
            unsafe {
                transformer_tensor_create_f32(
                    matrix_data.as_ptr(),
                    other_shape.as_ptr(),
                    2,
                    &mut other,
                )
            },
            STATUS_OK
        );

        // SAFETY: Live handles with invalid ranks/dimensions are rejected.
        assert_eq!(
            unsafe { transformer_tensor_matmul(vector, matrix, &mut output) },
            STATUS_INVALID_ARGUMENT
        );
        assert!(output.is_null());
        assert_eq!(
            unsafe { transformer_tensor_matmul(matrix, other, &mut output) },
            STATUS_INVALID_ARGUMENT
        );
        assert!(output.is_null());

        unsafe { destroy(vector) };
        unsafe { destroy(matrix) };
        unsafe { destroy(other) };
    }

    #[test]
    fn tensor_matmul_rejects_null_arguments_and_clears_output() {
        let mut output = std::ptr::NonNull::<TransformerTensor>::dangling().as_ptr();

        // SAFETY: Null handles are rejected before dereference.
        assert_eq!(
            unsafe { transformer_tensor_matmul(null(), null(), &mut output) },
            STATUS_INVALID_ARGUMENT
        );
        assert!(output.is_null());
        // SAFETY: Null output is rejected before any write.
        assert_eq!(
            unsafe { transformer_tensor_matmul(null(), null(), null_mut()) },
            STATUS_INVALID_ARGUMENT
        );
    }

    #[test]
    fn tensor_transpose_matches_manual_result_and_buffer_api() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let shape = [2, 3];
        let mut input = null_mut();
        let mut output = null_mut();
        // SAFETY: Input buffer matches shape [2, 3].
        assert_eq!(
            unsafe { transformer_tensor_create_f32(data.as_ptr(), shape.as_ptr(), 2, &mut input) },
            STATUS_OK
        );

        // SAFETY: Input handle is live and output is writable.
        assert_eq!(
            unsafe { transformer_tensor_transpose(input, &mut output) },
            STATUS_OK
        );

        let mut tensor_result = [0.0; 6];
        let mut buffer_result = [0.0; 6];
        let mut output_shape = [0; 2];
        // SAFETY: Output buffers have exact capacity.
        assert_eq!(
            unsafe {
                transformer_tensor_copy_data_f32(
                    output,
                    tensor_result.as_mut_ptr(),
                    tensor_result.len(),
                )
            },
            STATUS_OK
        );
        assert_eq!(
            unsafe { transformer_tensor_shape(output, output_shape.as_mut_ptr(), 2) },
            STATUS_OK
        );
        assert_eq!(
            unsafe { transformer_transpose_f32(data.as_ptr(), buffer_result.as_mut_ptr(), 2, 3,) },
            STATUS_OK
        );

        assert_eq!(tensor_result, [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
        assert_eq!(tensor_result, buffer_result);
        assert_eq!(output_shape, [3, 2]);

        let mut input_after = [0.0; 6];
        // SAFETY: Transpose leaves the input handle valid and unchanged.
        assert_eq!(
            unsafe {
                transformer_tensor_copy_data_f32(input, input_after.as_mut_ptr(), input_after.len())
            },
            STATUS_OK
        );
        assert_eq!(input_after, data);

        unsafe { destroy(output) };
        unsafe { destroy(input) };
    }

    #[test]
    fn tensor_transpose_materializes_square_matrix() {
        let data = [1.0, 2.0, 3.0, 4.0];
        let shape = [2, 2];
        let mut input = null_mut();
        let mut output = null_mut();
        // SAFETY: Input buffer matches shape [2, 2].
        assert_eq!(
            unsafe { transformer_tensor_create_f32(data.as_ptr(), shape.as_ptr(), 2, &mut input) },
            STATUS_OK
        );
        // SAFETY: Input handle is live and output is writable.
        assert_eq!(
            unsafe { transformer_tensor_transpose(input, &mut output) },
            STATUS_OK
        );

        let mut values = [0.0; 4];
        // SAFETY: Output buffer has exact capacity.
        assert_eq!(
            unsafe { transformer_tensor_copy_data_f32(output, values.as_mut_ptr(), 4) },
            STATUS_OK
        );
        assert_eq!(values, [1.0, 3.0, 2.0, 4.0]);

        unsafe { destroy(output) };
        unsafe { destroy(input) };
    }

    #[test]
    fn tensor_transpose_supports_zero_sized_matrix() {
        let shape = [2, 0];
        let mut input = null_mut();
        let mut output = null_mut();
        // SAFETY: Empty shape requires no data buffer.
        assert_eq!(
            unsafe { transformer_tensor_create_f32(null(), shape.as_ptr(), 2, &mut input) },
            STATUS_OK
        );
        // SAFETY: Empty matrix handle is live and output is writable.
        assert_eq!(
            unsafe { transformer_tensor_transpose(input, &mut output) },
            STATUS_OK
        );

        let mut output_shape = [usize::MAX; 2];
        let mut numel = usize::MAX;
        // SAFETY: Metadata outputs have sufficient capacity.
        assert_eq!(
            unsafe { transformer_tensor_shape(output, output_shape.as_mut_ptr(), 2) },
            STATUS_OK
        );
        assert_eq!(
            unsafe { transformer_tensor_numel(output, &mut numel) },
            STATUS_OK
        );
        assert_eq!(output_shape, [0, 2]);
        assert_eq!(numel, 0);

        unsafe { destroy(output) };
        unsafe { destroy(input) };
    }

    #[test]
    fn tensor_transpose_rejects_non_matrix_and_null_arguments() {
        let data = [1.0, 2.0, 3.0];
        let shape = [3];
        let mut vector = null_mut();
        let mut output = std::ptr::NonNull::<TransformerTensor>::dangling().as_ptr();
        // SAFETY: Input buffer matches vector shape.
        assert_eq!(
            unsafe { transformer_tensor_create_f32(data.as_ptr(), shape.as_ptr(), 1, &mut vector) },
            STATUS_OK
        );

        // SAFETY: Live non-matrix handle is rejected.
        assert_eq!(
            unsafe { transformer_tensor_transpose(vector, &mut output) },
            STATUS_INVALID_ARGUMENT
        );
        assert!(output.is_null());
        output = std::ptr::NonNull::<TransformerTensor>::dangling().as_ptr();
        // SAFETY: Null input is rejected before dereference.
        assert_eq!(
            unsafe { transformer_tensor_transpose(null(), &mut output) },
            STATUS_INVALID_ARGUMENT
        );
        assert!(output.is_null());
        // SAFETY: Null output is rejected before any write.
        assert_eq!(
            unsafe { transformer_tensor_transpose(vector, null_mut()) },
            STATUS_INVALID_ARGUMENT
        );

        unsafe { destroy(vector) };
    }

    #[test]
    fn tensor_softmax_matches_reference_and_buffer_api() {
        let data = [1.0, 2.0, 3.0];
        let shape = [3];
        let mut input = null_mut();
        let mut output = null_mut();
        // SAFETY: Input buffer matches shape [3].
        assert_eq!(
            unsafe { transformer_tensor_create_f32(data.as_ptr(), shape.as_ptr(), 1, &mut input) },
            STATUS_OK
        );
        // SAFETY: Input handle is live and output is writable.
        assert_eq!(
            unsafe { transformer_tensor_softmax(input, &mut output) },
            STATUS_OK
        );

        let mut tensor_result = [0.0; 3];
        let mut buffer_result = [0.0; 3];
        // SAFETY: Output buffers have exact capacity.
        assert_eq!(
            unsafe {
                transformer_tensor_copy_data_f32(
                    output,
                    tensor_result.as_mut_ptr(),
                    tensor_result.len(),
                )
            },
            STATUS_OK
        );
        assert_eq!(
            unsafe {
                transformer_softmax_f32(
                    data.as_ptr(),
                    buffer_result.as_mut_ptr(),
                    buffer_result.len(),
                )
            },
            STATUS_OK
        );

        for (actual, expected) in
            tensor_result
                .iter()
                .zip([0.090_030_57, 0.244_728_48, 0.665_240_94])
        {
            assert!((actual - expected).abs() < 1e-6);
        }
        assert_eq!(tensor_result, buffer_result);
        assert!((tensor_result.iter().sum::<f32>() - 1.0).abs() < 1e-6);

        let mut input_after = [0.0; 3];
        // SAFETY: Softmax leaves the input handle valid and unchanged.
        assert_eq!(
            unsafe {
                transformer_tensor_copy_data_f32(input, input_after.as_mut_ptr(), input_after.len())
            },
            STATUS_OK
        );
        assert_eq!(input_after, data);

        unsafe { destroy(output) };
        unsafe { destroy(input) };
    }

    #[test]
    fn tensor_softmax_last_dim_normalizes_each_matrix_row() {
        let data = [1.0, 2.0, 3.0, -3.0, -2.0, -1.0];
        let shape = [2, 3];
        let mut input = null_mut();
        let mut output = null_mut();
        assert_eq!(
            unsafe { transformer_tensor_create_f32(data.as_ptr(), shape.as_ptr(), 2, &mut input) },
            STATUS_OK
        );
        assert_eq!(
            unsafe { transformer_tensor_softmax_last_dim(input, &mut output) },
            STATUS_OK
        );

        let mut output_shape = [0; 2];
        let mut values = [0.0; 6];
        assert_eq!(
            unsafe { transformer_tensor_shape(output, output_shape.as_mut_ptr(), 2) },
            STATUS_OK
        );
        assert_eq!(
            unsafe { transformer_tensor_copy_data_f32(output, values.as_mut_ptr(), values.len()) },
            STATUS_OK
        );
        assert_eq!(output_shape, shape);
        for row in values.chunks_exact(3) {
            assert!((row.iter().sum::<f32>() - 1.0).abs() < 1.0e-6);
            for (actual, expected) in row.iter().zip([0.090_030_57, 0.244_728_48, 0.665_240_94]) {
                assert!((actual - expected).abs() < 1.0e-6);
            }
        }
        let mut input_after = [0.0; 6];
        assert_eq!(
            unsafe {
                transformer_tensor_copy_data_f32(input, input_after.as_mut_ptr(), input_after.len())
            },
            STATUS_OK
        );
        assert_eq!(input_after, data);

        unsafe { destroy(output) };
        unsafe { destroy(input) };
    }

    #[test]
    fn tensor_softmax_last_dim_matches_rank_one_operation_and_supports_rank_n() {
        let vector = [1000.0, 1001.0, 1002.0];
        let vector_shape = [3];
        let rank_three_data = [1.0, 2.0, 3.0, 3.0, 2.0, 1.0];
        let rank_three_shape = [1, 2, 3];
        let mut vector_input = null_mut();
        let mut old_output = null_mut();
        let mut new_output = null_mut();
        let mut rank_three_input = null_mut();
        let mut rank_three_output = null_mut();

        assert_eq!(
            unsafe {
                transformer_tensor_create_f32(
                    vector.as_ptr(),
                    vector_shape.as_ptr(),
                    1,
                    &mut vector_input,
                )
            },
            STATUS_OK
        );
        assert_eq!(
            unsafe { transformer_tensor_softmax(vector_input, &mut old_output) },
            STATUS_OK
        );
        assert_eq!(
            unsafe { transformer_tensor_softmax_last_dim(vector_input, &mut new_output) },
            STATUS_OK
        );
        let mut old_values = [0.0; 3];
        let mut new_values = [0.0; 3];
        assert_eq!(
            unsafe { transformer_tensor_copy_data_f32(old_output, old_values.as_mut_ptr(), 3) },
            STATUS_OK
        );
        assert_eq!(
            unsafe { transformer_tensor_copy_data_f32(new_output, new_values.as_mut_ptr(), 3) },
            STATUS_OK
        );
        assert_eq!(new_values, old_values);

        assert_eq!(
            unsafe {
                transformer_tensor_create_f32(
                    rank_three_data.as_ptr(),
                    rank_three_shape.as_ptr(),
                    3,
                    &mut rank_three_input,
                )
            },
            STATUS_OK
        );
        assert_eq!(
            unsafe {
                transformer_tensor_softmax_last_dim(rank_three_input, &mut rank_three_output)
            },
            STATUS_OK
        );
        let mut values = [0.0; 6];
        assert_eq!(
            unsafe {
                transformer_tensor_copy_data_f32(
                    rank_three_output,
                    values.as_mut_ptr(),
                    values.len(),
                )
            },
            STATUS_OK
        );
        for row in values.chunks_exact(3) {
            assert!((row.iter().sum::<f32>() - 1.0).abs() < 1.0e-6);
        }

        for handle in [
            rank_three_output,
            rank_three_input,
            new_output,
            old_output,
            vector_input,
        ] {
            unsafe { destroy(handle) };
        }
    }

    #[test]
    fn tensor_softmax_last_dim_rejects_empty_scalar_non_finite_and_null_arguments() {
        let empty_shape = [2, 0];
        let scalar_data = [1.0];
        let invalid_data = [1.0, f32::INFINITY];
        let invalid_shape = [1, 2];
        let mut empty = null_mut();
        let mut scalar = null_mut();
        let mut invalid = null_mut();
        assert_eq!(
            unsafe { transformer_tensor_create_f32(null(), empty_shape.as_ptr(), 2, &mut empty) },
            STATUS_OK
        );
        assert_eq!(
            unsafe { transformer_tensor_create_f32(scalar_data.as_ptr(), null(), 0, &mut scalar) },
            STATUS_OK
        );
        assert_eq!(
            unsafe {
                transformer_tensor_create_f32(
                    invalid_data.as_ptr(),
                    invalid_shape.as_ptr(),
                    2,
                    &mut invalid,
                )
            },
            STATUS_OK
        );

        for input in [empty, scalar, invalid] {
            let mut output = std::ptr::NonNull::<TransformerTensor>::dangling().as_ptr();
            assert_eq!(
                unsafe { transformer_tensor_softmax_last_dim(input, &mut output) },
                STATUS_INVALID_ARGUMENT
            );
            assert!(output.is_null());
        }
        let mut output = std::ptr::NonNull::<TransformerTensor>::dangling().as_ptr();
        assert_eq!(
            unsafe { transformer_tensor_softmax_last_dim(null(), &mut output) },
            STATUS_INVALID_ARGUMENT
        );
        assert!(output.is_null());
        assert_eq!(
            unsafe { transformer_tensor_softmax_last_dim(invalid, null_mut()) },
            STATUS_INVALID_ARGUMENT
        );

        for handle in [invalid, scalar, empty] {
            unsafe { destroy(handle) };
        }
    }

    #[test]
    fn tensor_softmax_is_stable_for_large_positive_and_negative_values() {
        for data in [[1000.0, 1001.0, 1002.0], [-1000.0, -1001.0, -1002.0]] {
            let shape = [3];
            let mut input = null_mut();
            let mut output = null_mut();
            // SAFETY: Input buffer matches shape [3].
            assert_eq!(
                unsafe {
                    transformer_tensor_create_f32(data.as_ptr(), shape.as_ptr(), 1, &mut input)
                },
                STATUS_OK
            );
            // SAFETY: Input handle is live and output is writable.
            assert_eq!(
                unsafe { transformer_tensor_softmax(input, &mut output) },
                STATUS_OK
            );

            let mut values = [0.0; 3];
            // SAFETY: Output buffer has exact capacity.
            assert_eq!(
                unsafe { transformer_tensor_copy_data_f32(output, values.as_mut_ptr(), 3) },
                STATUS_OK
            );
            assert!(values.iter().all(|value| value.is_finite()));
            assert!((values.iter().sum::<f32>() - 1.0).abs() < 1e-6);

            unsafe { destroy(output) };
            unsafe { destroy(input) };
        }
    }

    #[test]
    fn tensor_softmax_distributes_equal_values_uniformly() {
        let data = [0.0, 0.0, 0.0];
        let shape = [3];
        let mut input = null_mut();
        let mut output = null_mut();
        // SAFETY: Input buffer matches shape [3].
        assert_eq!(
            unsafe { transformer_tensor_create_f32(data.as_ptr(), shape.as_ptr(), 1, &mut input) },
            STATUS_OK
        );
        // SAFETY: Input handle is live and output is writable.
        assert_eq!(
            unsafe { transformer_tensor_softmax(input, &mut output) },
            STATUS_OK
        );

        let mut values = [0.0; 3];
        // SAFETY: Output buffer has exact capacity.
        assert_eq!(
            unsafe { transformer_tensor_copy_data_f32(output, values.as_mut_ptr(), 3) },
            STATUS_OK
        );
        assert!(values
            .iter()
            .all(|value| (*value - (1.0 / 3.0)).abs() < 1e-6));

        unsafe { destroy(output) };
        unsafe { destroy(input) };
    }

    #[test]
    fn tensor_softmax_rejects_empty_non_vector_and_non_finite_inputs() {
        let empty_shape = [0];
        let matrix_data = [1.0, 2.0];
        let matrix_shape = [1, 2];
        let invalid_data = [1.0, f32::NAN];
        let invalid_shape = [2];
        let mut empty = null_mut();
        let mut matrix = null_mut();
        let mut invalid = null_mut();
        let mut output = null_mut();
        // SAFETY: Creation buffers match their shapes.
        assert_eq!(
            unsafe { transformer_tensor_create_f32(null(), empty_shape.as_ptr(), 1, &mut empty) },
            STATUS_OK
        );
        assert_eq!(
            unsafe {
                transformer_tensor_create_f32(
                    matrix_data.as_ptr(),
                    matrix_shape.as_ptr(),
                    2,
                    &mut matrix,
                )
            },
            STATUS_OK
        );
        assert_eq!(
            unsafe {
                transformer_tensor_create_f32(
                    invalid_data.as_ptr(),
                    invalid_shape.as_ptr(),
                    1,
                    &mut invalid,
                )
            },
            STATUS_OK
        );

        // SAFETY: Live invalid inputs are rejected without producing a handle.
        for input in [empty, matrix, invalid] {
            assert_eq!(
                unsafe { transformer_tensor_softmax(input, &mut output) },
                STATUS_INVALID_ARGUMENT
            );
            assert!(output.is_null());
        }

        unsafe { destroy(empty) };
        unsafe { destroy(matrix) };
        unsafe { destroy(invalid) };
    }

    #[test]
    fn tensor_softmax_rejects_null_arguments_and_clears_output() {
        let mut output = std::ptr::NonNull::<TransformerTensor>::dangling().as_ptr();

        // SAFETY: Null input is rejected before dereference.
        assert_eq!(
            unsafe { transformer_tensor_softmax(null(), &mut output) },
            STATUS_INVALID_ARGUMENT
        );
        assert!(output.is_null());
        // SAFETY: Null output is rejected before any write.
        assert_eq!(
            unsafe { transformer_tensor_softmax(null(), null_mut()) },
            STATUS_INVALID_ARGUMENT
        );
    }

    #[test]
    fn tensor_layer_norm_preserves_shape_and_publishes_only_on_success() {
        let input = unsafe { create(&[-2.0, 0.0, 2.0, 4.0, 4.0, 4.0], &[2, 3]) };
        let weight = unsafe { create(&[0.5, 2.0, -1.0], &[3]) };
        let bias = unsafe { create(&[1.0, -2.0, 3.0], &[3]) };
        let invalid = unsafe { create(&[f32::NAN, 0.0, 1.0], &[3]) };
        let mut output = null_mut();
        assert_eq!(
            unsafe { transformer_tensor_layer_norm(input, weight, bias, 1e-5, &mut output) },
            STATUS_OK
        );
        assert!(!output.is_null());
        assert_eq!(unsafe { &*output }.tensor().shape().as_slice(), [2, 3]);
        assert!(unsafe { &*output }
            .tensor()
            .as_slice()
            .iter()
            .all(|v| v.is_finite()));
        unsafe { destroy(output) };

        output = std::ptr::NonNull::<TransformerTensor>::dangling().as_ptr();
        assert_eq!(
            unsafe { transformer_tensor_layer_norm(invalid, weight, bias, 1e-5, &mut output) },
            STATUS_INVALID_ARGUMENT
        );
        assert!(output.is_null());
        assert_eq!(
            unsafe { transformer_tensor_layer_norm(input, weight, bias, 0.0, &mut output) },
            STATUS_INVALID_ARGUMENT
        );
        assert!(output.is_null());
        for handle in [input, weight, bias, invalid] {
            unsafe { destroy(handle) };
        }
    }

    #[test]
    fn tensor_layer_norm_supports_empty_outer_dimensions() {
        for shape in [&[0, 3][..], &[2, 0, 3], &[0, 0, 3]] {
            let input = unsafe { create(&[], shape) };
            let weight = unsafe { create(&[1.0, 1.0, 1.0], &[3]) };
            let bias = unsafe { create(&[0.0, 0.0, 0.0], &[3]) };
            let mut output = null_mut();
            assert_eq!(
                unsafe { transformer_tensor_layer_norm(input, weight, bias, 1e-5, &mut output) },
                STATUS_OK
            );
            assert_eq!(unsafe { &*output }.tensor().shape().as_slice(), shape);
            assert!(unsafe { &*output }.tensor().is_empty());
            for handle in [input, weight, bias, output] {
                unsafe { destroy(handle) };
            }
        }
    }

    #[test]
    fn tensor_layer_norm_contains_panics_and_recovers_for_next_call() {
        let input = unsafe { create(&[1.0, 2.0], &[2]) };
        let weight = unsafe { create(&[1.0, 1.0], &[2]) };
        let bias = unsafe { create(&[0.0, 0.0], &[2]) };
        let mut output = std::ptr::NonNull::<TransformerTensor>::dangling().as_ptr();
        PANIC_LAYER_NORM.with(|flag| flag.set(true));
        assert_eq!(
            unsafe { transformer_tensor_layer_norm(input, weight, bias, 1e-5, &mut output) },
            STATUS_PANIC
        );
        assert!(output.is_null());
        assert_eq!(
            unsafe { transformer_tensor_layer_norm(input, weight, bias, 1e-5, &mut output) },
            STATUS_OK
        );
        for handle in [input, weight, bias, output] {
            unsafe { destroy(handle) };
        }
    }

    #[test]
    fn tensor_gelu_preserves_scalar_rank_n_empty_shape_and_input() {
        for (data, shape) in [
            (&[1.0][..], &[][..]),
            (&[-1.0, 0.0, 1.0], &[3][..]),
            (&[-2.0, -1.0, 0.0, 1.0, 2.0, 3.0], &[1, 2, 1, 3][..]),
            (&[][..], &[2, 0, 3][..]),
        ] {
            let input = unsafe { create(data, shape) };
            let before = unsafe { &*input }.tensor().as_slice().to_vec();
            let mut output = null_mut();
            assert_eq!(
                unsafe { transformer_tensor_gelu(input, &mut output) },
                STATUS_OK
            );
            assert_eq!(unsafe { &*output }.tensor().shape().as_slice(), shape);
            assert_eq!(unsafe { &*input }.tensor().as_slice(), before);
            assert_ne!(input, output);
            for handle in [input, output] {
                unsafe { destroy(handle) };
            }
        }
    }

    #[test]
    fn tensor_gelu_rejects_non_finite_without_publication_and_recovers() {
        let invalid = unsafe { create(&[1.0, f32::NAN], &[2]) };
        let valid = unsafe { create(&[-1.0, 1.0], &[2]) };
        let mut output = std::ptr::NonNull::<TransformerTensor>::dangling().as_ptr();
        assert_eq!(
            unsafe { transformer_tensor_gelu(invalid, &mut output) },
            STATUS_INVALID_ARGUMENT
        );
        assert!(output.is_null());
        assert_eq!(
            unsafe { transformer_tensor_gelu(valid, &mut output) },
            STATUS_OK
        );
        for handle in [invalid, valid, output] {
            unsafe { destroy(handle) };
        }
    }

    #[test]
    fn tensor_gelu_contains_panic_and_rejects_null_arguments() {
        let input = unsafe { create(&[1.0], &[1]) };
        let mut output = std::ptr::NonNull::<TransformerTensor>::dangling().as_ptr();
        PANIC_GELU.with(|flag| flag.set(true));
        assert_eq!(
            unsafe { transformer_tensor_gelu(input, &mut output) },
            STATUS_PANIC
        );
        assert!(output.is_null());
        assert_eq!(
            unsafe { transformer_tensor_gelu(input, &mut output) },
            STATUS_OK
        );
        unsafe { destroy(output) };
        assert_eq!(
            unsafe { transformer_tensor_gelu(null(), &mut output) },
            STATUS_INVALID_ARGUMENT
        );
        assert!(output.is_null());
        assert_eq!(
            unsafe { transformer_tensor_gelu(input, null_mut()) },
            STATUS_INVALID_ARGUMENT
        );
        unsafe { destroy(input) };
    }
}
