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

#[cfg(test)]
mod tests {
    use super::{matmul_f32, MatmulError};

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
}
