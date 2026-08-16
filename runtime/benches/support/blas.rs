use std::ffi::{c_char, c_int, CStr};

use crate::matmul::MatmulError;

const CBLAS_ROW_MAJOR: c_int = 101;
const CBLAS_NO_TRANS: c_int = 111;

#[link(name = "openblas")]
extern "C" {
    fn cblas_sgemm(
        layout: c_int,
        transpose_a: c_int,
        transpose_b: c_int,
        m: c_int,
        n: c_int,
        k: c_int,
        alpha: f32,
        a: *const f32,
        lda: c_int,
        b: *const f32,
        ldb: c_int,
        beta: f32,
        output: *mut f32,
        ldc: c_int,
    );
    fn openblas_get_config() -> *const c_char;
    fn openblas_get_corename() -> *const c_char;
    fn openblas_get_parallel() -> c_int;
    fn openblas_get_num_threads() -> c_int;
    fn openblas_set_num_threads(num_threads: c_int);
}

pub struct BlasInfo {
    pub config: String,
    pub core: String,
    pub parallel: &'static str,
    pub threads: c_int,
}

pub fn configure(threads: usize) -> BlasInfo {
    let threads = c_int::try_from(threads.max(1)).expect("BLAS thread count exceeds c_int");

    // SAFETY: These OpenBLAS process-global configuration functions accept no
    // borrowed pointers. The returned static strings are owned by OpenBLAS.
    unsafe {
        openblas_set_num_threads(threads);
        BlasInfo {
            config: c_string(openblas_get_config()),
            core: c_string(openblas_get_corename()),
            parallel: match openblas_get_parallel() {
                0 => "sequential",
                1 => "pthreads",
                2 => "openmp",
                _ => "unknown",
            },
            threads: openblas_get_num_threads(),
        }
    }
}

pub fn matmul_blas_f32(
    a: &[f32],
    b: &[f32],
    output: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) -> Result<(), MatmulError> {
    let a_length = m.checked_mul(k).ok_or(MatmulError::DimensionOverflow)?;
    let b_length = k.checked_mul(n).ok_or(MatmulError::DimensionOverflow)?;
    let output_length = m.checked_mul(n).ok_or(MatmulError::DimensionOverflow)?;

    if a.len() != a_length || b.len() != b_length || output.len() != output_length {
        return Err(MatmulError::LengthMismatch);
    }

    let m = c_int::try_from(m).map_err(|_| MatmulError::DimensionOverflow)?;
    let k = c_int::try_from(k).map_err(|_| MatmulError::DimensionOverflow)?;
    let n = c_int::try_from(n).map_err(|_| MatmulError::DimensionOverflow)?;

    if m == 0 || n == 0 || k == 0 {
        output.fill(0.0);
        return Ok(());
    }

    // SAFETY: Exact buffer lengths and CBLAS-compatible dimensions were
    // validated above. Row-major leading dimensions match the contiguous Rust
    // slices, and beta=0 makes output independent of its previous contents.
    unsafe {
        cblas_sgemm(
            CBLAS_ROW_MAJOR,
            CBLAS_NO_TRANS,
            CBLAS_NO_TRANS,
            m,
            n,
            k,
            1.0,
            a.as_ptr(),
            k,
            b.as_ptr(),
            n,
            0.0,
            output.as_mut_ptr(),
            n,
        );
    }

    Ok(())
}

unsafe fn c_string(pointer: *const c_char) -> String {
    if pointer.is_null() {
        return "unknown".to_owned();
    }

    // SAFETY: The caller passes a non-null static OpenBLAS string.
    unsafe { CStr::from_ptr(pointer) }
        .to_string_lossy()
        .into_owned()
}
