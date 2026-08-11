mod handle;
mod tensor_api;

use std::ffi::{c_char, c_int};
use std::panic::{catch_unwind, AssertUnwindSafe};

use crate::kernels::add::add_f32;
use crate::kernels::matmul::matmul_f32;
use crate::kernels::softmax::softmax_f32;
use crate::kernels::transpose::transpose_f32;

pub(super) const STATUS_OK: c_int = 0;
pub(super) const STATUS_INVALID_ARGUMENT: c_int = 1;
pub(super) const STATUS_PANIC: c_int = 2;
pub(super) const STATUS_INSUFFICIENT_BUFFER: c_int = 3;
const VERSION: &[u8] = b"0.1.0-dev\0";

#[no_mangle]
pub extern "C" fn transformer_native_version() -> *const c_char {
    VERSION.as_ptr().cast()
}

/// Adds two float32 buffers through the C ABI.
///
/// # Safety
///
/// For non-zero `length`, every pointer must be valid for `length` elements.
/// `a` and `b` must remain readable for the entire call. `output` must remain
/// writable for the entire call and must not overlap `a` or `b`. A non-null
/// pointer alone does not prove any of these requirements; the caller owns
/// this contract.
#[no_mangle]
pub unsafe extern "C" fn transformer_tensor_add_f32(
    a: *const f32,
    b: *const f32,
    output: *mut f32,
    length: usize,
) -> c_int {
    if length == 0 {
        return STATUS_OK;
    }

    if a.is_null() || b.is_null() || output.is_null() {
        return STATUS_INVALID_ARGUMENT;
    }

    if length > isize::MAX as usize / size_of::<f32>() {
        return STATUS_INVALID_ARGUMENT;
    }

    catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: The caller contract is checked for null and representable
        // length above; validity and non-overlap are required by the C ABI.
        let a = unsafe { std::slice::from_raw_parts(a, length) };
        let b = unsafe { std::slice::from_raw_parts(b, length) };
        let output = unsafe { std::slice::from_raw_parts_mut(output, length) };

        match add_f32(a, b, output) {
            Ok(()) => STATUS_OK,
            Err(_) => STATUS_INVALID_ARGUMENT,
        }
    }))
    .unwrap_or(STATUS_PANIC)
}

/// Multiplies row-major matrices A[M x K] and B[K x N].
///
/// # Safety
///
/// The caller guarantees that `a`, `b`, and `output` remain valid for M*K,
/// K*N, and M*N float32 elements respectively for the complete call. `output`
/// must be writable and must not overlap either input buffer.
#[no_mangle]
pub unsafe extern "C" fn transformer_matmul_f32(
    a: *const f32,
    b: *const f32,
    output: *mut f32,
    m: usize,
    k: usize,
    n: usize,
) -> c_int {
    let Some(a_length) = m.checked_mul(k) else {
        return STATUS_INVALID_ARGUMENT;
    };
    let Some(b_length) = k.checked_mul(n) else {
        return STATUS_INVALID_ARGUMENT;
    };
    let Some(output_length) = m.checked_mul(n) else {
        return STATUS_INVALID_ARGUMENT;
    };

    if a_length > isize::MAX as usize / size_of::<f32>()
        || b_length > isize::MAX as usize / size_of::<f32>()
        || output_length > isize::MAX as usize / size_of::<f32>()
    {
        return STATUS_INVALID_ARGUMENT;
    }

    if output_length == 0 {
        return STATUS_OK;
    }

    if output.is_null() || (a_length > 0 && a.is_null()) || (b_length > 0 && b.is_null()) {
        return STATUS_INVALID_ARGUMENT;
    }

    catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: Lengths and required non-null pointers are checked above;
        // allocation validity and non-overlap remain the caller's contract.
        let a = if a_length == 0 {
            &[]
        } else {
            unsafe { std::slice::from_raw_parts(a, a_length) }
        };
        let b = if b_length == 0 {
            &[]
        } else {
            unsafe { std::slice::from_raw_parts(b, b_length) }
        };
        let output = unsafe { std::slice::from_raw_parts_mut(output, output_length) };

        match matmul_f32(a, b, output, m, k, n) {
            Ok(()) => STATUS_OK,
            Err(_) => STATUS_INVALID_ARGUMENT,
        }
    }))
    .unwrap_or(STATUS_PANIC)
}

/// Transposes a row-major [rows x columns] float32 matrix.
///
/// # Safety
///
/// The caller guarantees that `input` and `output` remain valid for
/// `rows * columns` float32 elements for the complete call. `output` must be
/// writable and must not overlap `input`.
#[no_mangle]
pub unsafe extern "C" fn transformer_transpose_f32(
    input: *const f32,
    output: *mut f32,
    rows: usize,
    columns: usize,
) -> c_int {
    let Some(length) = rows.checked_mul(columns) else {
        return STATUS_INVALID_ARGUMENT;
    };

    if length > isize::MAX as usize / size_of::<f32>() {
        return STATUS_INVALID_ARGUMENT;
    }

    if length == 0 {
        return STATUS_OK;
    }

    if input.is_null() || output.is_null() {
        return STATUS_INVALID_ARGUMENT;
    }

    catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: Length and non-null pointers are checked above; allocation
        // validity and non-overlap remain the caller's contract.
        let input = unsafe { std::slice::from_raw_parts(input, length) };
        let output = unsafe { std::slice::from_raw_parts_mut(output, length) };

        match transpose_f32(input, output, rows, columns) {
            Ok(()) => STATUS_OK,
            Err(_) => STATUS_INVALID_ARGUMENT,
        }
    }))
    .unwrap_or(STATUS_PANIC)
}

/// Computes a numerically stable softmax over one float32 vector.
///
/// # Safety
///
/// The caller guarantees that `input` and `output` remain valid for `length`
/// float32 elements for the complete call. `output` must be writable and must
/// not overlap `input`.
#[no_mangle]
pub unsafe extern "C" fn transformer_softmax_f32(
    input: *const f32,
    output: *mut f32,
    length: usize,
) -> c_int {
    if length == 0
        || length > isize::MAX as usize / size_of::<f32>()
        || input.is_null()
        || output.is_null()
    {
        return STATUS_INVALID_ARGUMENT;
    }

    catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: Length and non-null pointers are checked above; allocation
        // validity and non-overlap remain the caller's contract.
        let input = unsafe { std::slice::from_raw_parts(input, length) };
        let output = unsafe { std::slice::from_raw_parts_mut(output, length) };

        match softmax_f32(input, output) {
            Ok(()) => STATUS_OK,
            Err(_) => STATUS_INVALID_ARGUMENT,
        }
    }))
    .unwrap_or(STATUS_PANIC)
}

#[cfg(test)]
mod tests {
    use std::ptr::{null, null_mut};

    use super::{
        transformer_matmul_f32, transformer_softmax_f32, transformer_tensor_add_f32,
        transformer_transpose_f32, STATUS_INVALID_ARGUMENT, STATUS_OK,
    };

    #[test]
    fn accepts_empty_null_buffers() {
        // SAFETY: Zero length makes the ABI explicitly accept null pointers.
        let status = unsafe { transformer_tensor_add_f32(null(), null(), null_mut(), 0) };

        assert_eq!(status, STATUS_OK);
    }

    #[test]
    fn rejects_null_buffers_for_non_empty_input() {
        // SAFETY: The function rejects the pointers before constructing slices.
        let status = unsafe { transformer_tensor_add_f32(null(), null(), null_mut(), 1) };

        assert_eq!(status, STATUS_INVALID_ARGUMENT);
    }

    #[test]
    fn matmul_accepts_empty_output_with_null_buffers() {
        // SAFETY: M x N is empty, so the ABI dereferences no pointers.
        let status = unsafe { transformer_matmul_f32(null(), null(), null_mut(), 0, 3, 2) };

        assert_eq!(status, STATUS_OK);
    }

    #[test]
    fn matmul_rejects_null_buffers_for_non_empty_matrices() {
        // SAFETY: The function rejects null pointers before constructing slices.
        let status = unsafe { transformer_matmul_f32(null(), null(), null_mut(), 1, 1, 1) };

        assert_eq!(status, STATUS_INVALID_ARGUMENT);
    }

    #[test]
    fn matmul_rejects_dimension_overflow() {
        // SAFETY: Overflow is rejected before any pointer is dereferenced.
        let status =
            unsafe { transformer_matmul_f32(null(), null(), null_mut(), usize::MAX, 2, 1) };

        assert_eq!(status, STATUS_INVALID_ARGUMENT);
    }

    #[test]
    fn transpose_accepts_empty_shape_with_null_buffers() {
        // SAFETY: The empty shape causes no pointer dereference.
        let status = unsafe { transformer_transpose_f32(null(), null_mut(), 0, 3) };

        assert_eq!(status, STATUS_OK);
    }

    #[test]
    fn transpose_rejects_null_non_empty_buffers() {
        // SAFETY: Null pointers are rejected before slices are constructed.
        let status = unsafe { transformer_transpose_f32(null(), null_mut(), 2, 3) };

        assert_eq!(status, STATUS_INVALID_ARGUMENT);
    }

    #[test]
    fn transpose_rejects_dimension_overflow() {
        // SAFETY: Overflow is rejected before pointers are dereferenced.
        let status = unsafe { transformer_transpose_f32(null(), null_mut(), usize::MAX, 2) };

        assert_eq!(status, STATUS_INVALID_ARGUMENT);
    }

    #[test]
    fn softmax_rejects_empty_vector() {
        // SAFETY: Empty input is rejected before pointers are dereferenced.
        let status = unsafe { transformer_softmax_f32(null(), null_mut(), 0) };

        assert_eq!(status, STATUS_INVALID_ARGUMENT);
    }

    #[test]
    fn softmax_rejects_null_buffers() {
        // SAFETY: Null pointers are rejected before slices are constructed.
        let status = unsafe { transformer_softmax_f32(null(), null_mut(), 3) };

        assert_eq!(status, STATUS_INVALID_ARGUMENT);
    }

    #[test]
    fn softmax_processes_valid_buffers() {
        let input = [1.0, 2.0, 3.0];
        let mut output = [0.0; 3];

        // SAFETY: Both arrays are valid, non-overlapping, and have length 3.
        let status =
            unsafe { transformer_softmax_f32(input.as_ptr(), output.as_mut_ptr(), input.len()) };

        assert_eq!(status, STATUS_OK);
        assert!((output.iter().sum::<f32>() - 1.0).abs() <= 1.0e-5);
    }
}
