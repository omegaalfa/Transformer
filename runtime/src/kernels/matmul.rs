#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatmulError {
    DimensionOverflow,
    LengthMismatch,
}

pub fn matmul_f32(
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

    for row in 0..m {
        for column in 0..n {
            let mut sum = 0.0;

            for inner in 0..k {
                sum += a[row * k + inner] * b[inner * n + column];
            }

            output[row * n + column] = sum;
        }
    }

    Ok(())
}

pub(crate) fn matmul_cache_friendly_f32(
    a: &[f32],
    b: &[f32],
    output: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) -> Result<(), MatmulError> {
    validate_lengths(a, b, output, m, k, n)?;
    output.fill(0.0);

    for row in 0..m {
        let a_row = &a[row * k..(row + 1) * k];
        let output_row = &mut output[row * n..(row + 1) * n];

        for (inner, &a_value) in a_row.iter().enumerate() {
            let b_row = &b[inner * n..(inner + 1) * n];
            for (destination, &b_value) in output_row.iter_mut().zip(b_row) {
                *destination += a_value * b_value;
            }
        }
    }

    Ok(())
}

pub(crate) fn matmul_tiled_f32(
    a: &[f32],
    b: &[f32],
    output: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) -> Result<(), MatmulError> {
    const M_TILE: usize = 32;
    const K_TILE: usize = 64;
    const N_TILE: usize = 128;

    validate_lengths(a, b, output, m, k, n)?;
    output.fill(0.0);

    for row_start in (0..m).step_by(M_TILE) {
        let row_end = (row_start + M_TILE).min(m);

        for inner_start in (0..k).step_by(K_TILE) {
            let inner_end = (inner_start + K_TILE).min(k);

            for column_start in (0..n).step_by(N_TILE) {
                let column_end = (column_start + N_TILE).min(n);

                for inner in inner_start..inner_end {
                    let b_row = &b[inner * n + column_start..inner * n + column_end];

                    for row in row_start..row_end {
                        let a_value = a[row * k + inner];
                        let output_row = &mut output[row * n + column_start..row * n + column_end];

                        for (destination, &b_value) in output_row.iter_mut().zip(b_row) {
                            *destination += a_value * b_value;
                        }
                    }
                }
            }
        }
    }

    Ok(())
}

fn validate_lengths(
    a: &[f32],
    b: &[f32],
    output: &[f32],
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

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{matmul_cache_friendly_f32, matmul_f32, matmul_tiled_f32, MatmulError};

    type Matmul = fn(&[f32], &[f32], &mut [f32], usize, usize, usize) -> Result<(), MatmulError>;

    const OPTIMIZED_MATMULS: [Matmul; 2] = [matmul_cache_friendly_f32, matmul_tiled_f32];

    #[test]
    fn multiplies_two_by_three_and_three_by_two() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = [7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let mut output = [0.0; 4];

        assert_eq!(matmul_f32(&a, &b, &mut output, 2, 3, 2), Ok(()));
        assert_eq!(output, [58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn multiplies_row_by_column() {
        let mut output = [0.0; 1];

        assert_eq!(
            matmul_f32(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], &mut output, 1, 3, 1),
            Ok(())
        );
        assert_eq!(output, [32.0]);
    }

    #[test]
    fn handles_zero_inner_dimension() {
        let mut output = [1.0; 4];

        assert_eq!(matmul_f32(&[], &[], &mut output, 2, 0, 2), Ok(()));
        assert_eq!(output, [0.0; 4]);
    }

    #[test]
    fn rejects_incorrect_buffer_lengths() {
        let mut output = [0.0; 4];

        assert_eq!(
            matmul_f32(&[1.0], &[1.0], &mut output, 2, 2, 2),
            Err(MatmulError::LengthMismatch)
        );
    }

    #[test]
    fn optimized_implementations_match_the_scalar_baseline() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = [7.0, 8.0, 9.0, 10.0, 11.0, 12.0];

        for matmul in OPTIMIZED_MATMULS {
            let mut output = [f32::NAN; 4];
            assert_eq!(matmul(&a, &b, &mut output, 2, 3, 2), Ok(()));
            assert_eq!(output, [58.0, 64.0, 139.0, 154.0]);
        }
    }

    #[test]
    fn optimized_implementations_overwrite_output_when_inner_dimension_is_zero() {
        for matmul in OPTIMIZED_MATMULS {
            let mut output = [1.0; 4];
            assert_eq!(matmul(&[], &[], &mut output, 2, 0, 2), Ok(()));
            assert_eq!(output, [0.0; 4]);
        }
    }

    #[test]
    fn optimized_implementations_preserve_length_validation() {
        for matmul in OPTIMIZED_MATMULS {
            let mut output = [0.0; 4];
            assert_eq!(
                matmul(&[1.0], &[1.0], &mut output, 2, 2, 2),
                Err(MatmulError::LengthMismatch)
            );
        }
    }

    #[test]
    fn optimized_implementations_handle_partial_edge_tiles() {
        let (m, k, n) = (33, 65, 129);
        let a: Vec<f32> = (0..m * k).map(|index| (index % 17) as f32).collect();
        let b: Vec<f32> = (0..k * n).map(|index| (index % 13) as f32).collect();
        let mut expected = vec![0.0; m * n];

        matmul_f32(&a, &b, &mut expected, m, k, n).unwrap();

        for matmul in OPTIMIZED_MATMULS {
            let mut actual = vec![f32::NAN; m * n];
            matmul(&a, &b, &mut actual, m, k, n).unwrap();
            assert_eq!(actual, expected);
        }
    }
}
