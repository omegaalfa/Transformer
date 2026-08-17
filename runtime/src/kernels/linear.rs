use super::matmul_dispatch::matmul_dispatch_f32;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinearError {
    InvalidDimensions,
    LengthMismatch,
    Matmul,
}

/// Projects the last dimension of a contiguous row-major Float32 tensor.
pub fn linear_last_dim_f32(
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    output: &mut [f32],
    input_features: usize,
    output_features: usize,
) -> Result<(), LinearError> {
    if input_features == 0 || output_features == 0 || input.len() % input_features != 0 {
        return Err(LinearError::InvalidDimensions);
    }
    let rows = input.len() / input_features;
    let weight_length = input_features
        .checked_mul(output_features)
        .ok_or(LinearError::InvalidDimensions)?;
    let output_length = rows
        .checked_mul(output_features)
        .ok_or(LinearError::InvalidDimensions)?;
    if weight.len() != weight_length
        || output.len() != output_length
        || bias.is_some_and(|values| values.len() != output_features)
    {
        return Err(LinearError::LengthMismatch);
    }

    matmul_dispatch_f32(input, weight, output, rows, input_features, output_features)
        .map_err(|_| LinearError::Matmul)?;
    if let Some(bias) = bias {
        for row in output.chunks_exact_mut(output_features) {
            for (value, bias) in row.iter_mut().zip(bias) {
                *value += bias;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn projects_rows_and_applies_unexpanded_bias() {
        let input = [1.0, 2.0, 3.0, 4.0];
        let weight = [1.0, -1.0, 2.0, 0.5, 3.0, 2.0];
        let bias = [0.25, -0.5, 1.0];
        let mut output = [0.0; 6];
        linear_last_dim_f32(&input, &weight, Some(&bias), &mut output, 2, 3).unwrap();
        assert_eq!(output, [2.25, 4.5, 7.0, 5.25, 8.5, 15.0]);
    }

    #[test]
    fn rejects_invalid_lengths_and_dimensions() {
        assert_eq!(
            linear_last_dim_f32(&[1.0], &[1.0], None, &mut [0.0], 0, 1),
            Err(LinearError::InvalidDimensions)
        );
        assert_eq!(
            linear_last_dim_f32(&[1.0, 2.0], &[1.0], None, &mut [0.0], 2, 1),
            Err(LinearError::LengthMismatch)
        );
    }
}
