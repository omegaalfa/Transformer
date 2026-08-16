use std::ffi::{c_char, c_int, c_void};
use std::sync::OnceLock;

use super::matmul::MatmulError;

const CBLAS_ROW_MAJOR: c_int = 101;
const CBLAS_NO_TRANS: c_int = 111;
const RTLD_LAZY: c_int = 0x00001;
const RTLD_LOCAL: c_int = 0;

type Sgemm = unsafe extern "C" fn(
    c_int,
    c_int,
    c_int,
    c_int,
    c_int,
    c_int,
    f32,
    *const f32,
    c_int,
    *const f32,
    c_int,
    f32,
    *mut f32,
    c_int,
);

#[link(name = "dl")]
extern "C" {
    fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
}

static SGEMM: OnceLock<Option<Sgemm>> = OnceLock::new();

pub(super) fn is_available() -> bool {
    sgemm().is_some()
}

pub(super) fn try_matmul_blas_f32(
    a: &[f32],
    b: &[f32],
    output: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) -> Option<Result<(), MatmulError>> {
    let sgemm = sgemm()?;
    Some(call_sgemm(sgemm, a, b, output, m, k, n))
}

fn call_sgemm(
    sgemm: Sgemm,
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

    // SAFETY: Exact buffer sizes and CBLAS-compatible dimensions were checked.
    // The runtime Tensor is contiguous row-major, so no transpose or copy is
    // required. beta=0 makes output independent of its previous contents.
    unsafe {
        sgemm(
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

fn sgemm() -> Option<Sgemm> {
    *SGEMM.get_or_init(load_sgemm)
}

fn load_sgemm() -> Option<Sgemm> {
    const LIBRARIES: [&[u8]; 2] = [b"libopenblas.so.0\0", b"libopenblas.so\0"];
    const SYMBOL: &[u8] = b"cblas_sgemm\0";

    for library in LIBRARIES {
        // SAFETY: Names are statically NUL-terminated. A successful handle is
        // intentionally retained for process lifetime so the symbol stays valid.
        let handle = unsafe { dlopen(library.as_ptr().cast(), RTLD_LAZY | RTLD_LOCAL) };
        if handle.is_null() {
            continue;
        }

        // SAFETY: The loaded symbol is checked for null and OpenBLAS defines it
        // with the CBLAS SGEMM signature represented by `Sgemm`.
        let symbol = unsafe { dlsym(handle, SYMBOL.as_ptr().cast()) };
        if !symbol.is_null() {
            // SAFETY: Function pointers returned by dlsym are converted to the
            // exact C signature declared above.
            return Some(unsafe { std::mem::transmute::<*mut c_void, Sgemm>(symbol) });
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::matmul::matmul_f32;

    #[test]
    fn available_blas_matches_row_major_reference() {
        if !is_available() {
            return;
        }

        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = [7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let mut expected = [0.0; 4];
        let mut actual = [f32::NAN; 4];
        matmul_f32(&a, &b, &mut expected, 2, 3, 2).unwrap();

        assert_eq!(
            try_matmul_blas_f32(&a, &b, &mut actual, 2, 3, 2),
            Some(Ok(()))
        );
        assert_eq!(actual, expected);
    }
}
