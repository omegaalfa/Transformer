use std::ffi::{c_int, c_void};

pub const CUDA_BGE_MATH_FP32: c_int = 0;
pub const CUDA_BGE_MATH_TF32: c_int = 1;

extern "C" {
    fn cuda_bge_available_impl() -> c_int;
    fn cuda_bge_create_impl() -> *mut c_void;
    fn cuda_bge_set_parameter_impl(
        handle: *mut c_void,
        index: c_int,
        data: *const f32,
        count: usize,
    ) -> c_int;
    fn cuda_bge_finalize_impl(handle: *mut c_void) -> c_int;
    fn cuda_bge_set_math_mode_impl(handle: *mut c_void, mode: c_int) -> c_int;
    fn cuda_bge_set_graph_enabled_impl(handle: *mut c_void, enabled: c_int) -> c_int;
    fn cuda_bge_forward_impl(
        handle: *mut c_void,
        ids: *const i64,
        mask: *const u8,
        types: *const i64,
        batch: c_int,
        sequence: c_int,
        output: *mut f32,
    ) -> c_int;
    fn cuda_bge_profile_impl(
        handle: *mut c_void,
        ids: *const i64,
        mask: *const u8,
        types: *const i64,
        batch: c_int,
        sequence: c_int,
        output: *mut f32,
        timings: *mut f32,
        timing_capacity: c_int,
        timing_count: *mut c_int,
    ) -> c_int;
    fn cuda_bge_forward_detailed_impl(
        handle: *mut c_void,
        ids: *const i64,
        mask: *const u8,
        types: *const i64,
        batch: c_int,
        sequence: c_int,
        output: *mut f32,
        phases: *mut f32,
    ) -> c_int;
    fn cuda_bge_destroy_impl(handle: *mut c_void);
    fn cuda_bge_memory_info_impl(free_bytes: *mut usize, total_bytes: *mut usize) -> c_int;
    fn cuda_bge_diagnostics_impl(handle: *mut c_void, values: *mut u64, capacity: usize) -> c_int;
}

#[no_mangle]
pub extern "C" fn transformer_cuda_available() -> c_int {
    unsafe { cuda_bge_available_impl() }
}
#[no_mangle]
pub extern "C" fn transformer_cuda_bge_create() -> *mut c_void {
    unsafe { cuda_bge_create_impl() }
}
/// # Safety
/// `handle` must be a live CUDA BGE handle and `data` must address `count`
/// readable Float32 values for the duration of this call.
#[no_mangle]
pub unsafe extern "C" fn transformer_cuda_bge_set_parameter(
    handle: *mut c_void,
    index: c_int,
    data: *const f32,
    count: usize,
) -> c_int {
    unsafe { cuda_bge_set_parameter_impl(handle, index, data, count) }
}
/// # Safety
/// `handle` must be a live CUDA BGE handle created by this library.
#[no_mangle]
pub unsafe extern "C" fn transformer_cuda_bge_finalize(handle: *mut c_void) -> c_int {
    unsafe { cuda_bge_finalize_impl(handle) }
}
/// # Safety
/// `handle` must be a live CUDA BGE handle. Mode 0 selects pedantic FP32 and
/// mode 1 selects TF32 Tensor Core math.
#[no_mangle]
pub unsafe extern "C" fn transformer_cuda_bge_set_math_mode(
    handle: *mut c_void,
    mode: c_int,
) -> c_int {
    if mode != CUDA_BGE_MATH_FP32 && mode != CUDA_BGE_MATH_TF32 {
        return 1;
    }
    unsafe { cuda_bge_set_math_mode_impl(handle, mode) }
}
/// # Safety
/// `handle` must be live. This internal benchmark switch selects ordinary
/// stream submission (0) or the exact-shape device-resident CUDA Graph (1).
#[no_mangle]
pub unsafe extern "C" fn transformer_cuda_bge_set_graph_enabled(
    handle: *mut c_void,
    enabled: c_int,
) -> c_int {
    unsafe { cuda_bge_set_graph_enabled_impl(handle, enabled) }
}
/// # Safety
/// All pointers must be valid for the dimensions supplied. `output` must hold
/// at least `batch * 384` writable Float32 values.
#[no_mangle]
pub unsafe extern "C" fn transformer_cuda_bge_forward(
    handle: *mut c_void,
    ids: *const i64,
    mask: *const u8,
    types: *const i64,
    batch: c_int,
    sequence: c_int,
    output: *mut f32,
) -> c_int {
    unsafe { cuda_bge_forward_impl(handle, ids, mask, types, batch, sequence, output) }
}
/// # Safety
/// Input/output requirements are identical to `transformer_cuda_bge_forward`.
/// `timings` must hold 137 Float32 values and `timing_count` must be writable.
#[no_mangle]
pub unsafe extern "C" fn transformer_cuda_bge_profile(
    handle: *mut c_void,
    ids: *const i64,
    mask: *const u8,
    types: *const i64,
    batch: c_int,
    sequence: c_int,
    output: *mut f32,
    timings: *mut f32,
    timing_capacity: c_int,
    timing_count: *mut c_int,
) -> c_int {
    unsafe {
        cuda_bge_profile_impl(
            handle,
            ids,
            mask,
            types,
            batch,
            sequence,
            output,
            timings,
            timing_capacity,
            timing_count,
        )
    }
}
/// # Safety
/// Input/output requirements match `transformer_cuda_bge_forward`; `phases`
/// must hold three Float32 CUDA-event durations for H2D, device, and D2H.
#[no_mangle]
pub unsafe extern "C" fn transformer_cuda_bge_forward_detailed(
    handle: *mut c_void,
    ids: *const i64,
    mask: *const u8,
    types: *const i64,
    batch: c_int,
    sequence: c_int,
    output: *mut f32,
    phases: *mut f32,
) -> c_int {
    unsafe {
        cuda_bge_forward_detailed_impl(handle, ids, mask, types, batch, sequence, output, phases)
    }
}
/// # Safety
/// `handle` must be null or a live handle that has not previously been destroyed.
#[no_mangle]
pub unsafe extern "C" fn transformer_cuda_bge_destroy(handle: *mut c_void) {
    unsafe { cuda_bge_destroy_impl(handle) }
}

/// # Safety
/// Both pointers must address writable `usize` values.
#[no_mangle]
pub unsafe extern "C" fn transformer_cuda_memory_info(
    free_bytes: *mut usize,
    total_bytes: *mut usize,
) -> c_int {
    unsafe { cuda_bge_memory_info_impl(free_bytes, total_bytes) }
}

/// # Safety
/// `handle` must be live and `values` must address at least eighteen writable
/// `u64` entries. This diagnostic ABI is intended for internal benchmarks.
#[no_mangle]
pub unsafe extern "C" fn transformer_cuda_bge_diagnostics(
    handle: *mut c_void,
    values: *mut u64,
    capacity: usize,
) -> c_int {
    unsafe { cuda_bge_diagnostics_impl(handle, values, capacity) }
}
