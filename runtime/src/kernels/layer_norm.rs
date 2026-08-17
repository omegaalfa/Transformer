#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerNormError {
    InvalidDimension,
    InvalidEpsilon,
    InvalidLength,
    NonFiniteInput,
    NonFiniteParameter,
    NonFiniteResult,
}

pub fn layer_norm_f32(
    input: &[f32],
    gamma: &[f32],
    beta: &[f32],
    output: &mut [f32],
    d: usize,
    epsilon: f32,
) -> Result<(), LayerNormError> {
    if d == 0 {
        return Err(LayerNormError::InvalidDimension);
    }
    if !epsilon.is_finite() || epsilon <= 0.0 {
        return Err(LayerNormError::InvalidEpsilon);
    }
    if gamma.len() != d || beta.len() != d || input.len() != output.len() || input.len() % d != 0 {
        return Err(LayerNormError::InvalidLength);
    }
    if input.iter().any(|v| !v.is_finite()) {
        return Err(LayerNormError::NonFiniteInput);
    }
    if gamma.iter().chain(beta).any(|v| !v.is_finite()) {
        return Err(LayerNormError::NonFiniteParameter);
    }

    // Validate every derived value before mutating the caller's destination.
    for row in input.chunks_exact(d) {
        validate_row(row, gamma, beta, epsilon)?;
    }
    for (row, destination) in input.chunks_exact(d).zip(output.chunks_exact_mut(d)) {
        let (mean, inv_std) = statistics(row, epsilon)?;
        for i in 0..d {
            destination[i] =
                (gamma[i] as f64 * (row[i] as f64 - mean) * inv_std + beta[i] as f64) as f32;
        }
    }
    Ok(())
}

fn statistics(row: &[f32], epsilon: f32) -> Result<(f64, f64), LayerNormError> {
    let mut mean = 0.0_f64;
    let mut m2 = 0.0_f64;
    for (index, &value) in row.iter().enumerate() {
        let count = (index + 1) as f64;
        let delta = value as f64 - mean;
        mean += delta / count;
        m2 += delta * (value as f64 - mean);
        if !mean.is_finite() || !m2.is_finite() {
            return Err(LayerNormError::NonFiniteResult);
        }
    }
    let variance = m2 / row.len() as f64;
    let denominator = variance + epsilon as f64;
    let inv_std = 1.0 / denominator.sqrt();
    if !variance.is_finite() || variance < 0.0 || !inv_std.is_finite() {
        return Err(LayerNormError::NonFiniteResult);
    }
    Ok((mean, inv_std))
}

fn validate_row(
    row: &[f32],
    gamma: &[f32],
    beta: &[f32],
    epsilon: f32,
) -> Result<(), LayerNormError> {
    let (mean, inv_std) = statistics(row, epsilon)?;
    for i in 0..row.len() {
        let result = gamma[i] as f64 * (row[i] as f64 - mean) * inv_std + beta[i] as f64;
        if !result.is_finite() || !(result as f32).is_finite() {
            return Err(LayerNormError::NonFiniteResult);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizes_rows_with_population_variance_and_affine_parameters() {
        let input = [-2.0, 0.0, 2.0, 1.0, 1.0, 1.0];
        let gamma = [0.5, 2.0, -1.0];
        let beta = [1.0, -2.0, 3.0];
        let mut output = [0.0; 6];
        layer_norm_f32(&input, &gamma, &beta, &mut output, 3, 1e-5).unwrap();
        for (actual, expected) in output[3..].iter().zip(beta) {
            assert!((actual - expected).abs() < 1e-6);
        }
        assert!(output.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn supports_empty_outer_dimensions_and_d_one() {
        layer_norm_f32(&[], &[1.0, 2.0], &[0.0, 0.0], &mut [], 2, 1e-5).unwrap();
        let mut output = [0.0];
        layer_norm_f32(&[42.0], &[3.0], &[-4.0], &mut output, 1, 1e-5).unwrap();
        assert_eq!(output, [-4.0]);
    }

    #[test]
    fn rejects_all_invalid_inputs_without_writing() {
        type InvalidCase<'a> = (&'a [f32], &'a [f32], &'a [f32], usize, f32);
        let cases: &[InvalidCase<'_>] = &[
            (&[1.0, 2.0], &[1.0, 1.0], &[0.0, 0.0], 0, 1e-5),
            (&[1.0, 2.0], &[1.0, 1.0], &[0.0, 0.0], 2, 0.0),
            (&[f32::NAN, 2.0], &[1.0, 1.0], &[0.0, 0.0], 2, 1e-5),
            (&[1.0, 2.0], &[f32::INFINITY, 1.0], &[0.0, 0.0], 2, 1e-5),
        ];
        for &(input, gamma, beta, d, epsilon) in cases {
            let mut output = [9.0, 9.0];
            assert!(layer_norm_f32(input, gamma, beta, &mut output, d, epsilon).is_err());
            assert_eq!(output, [9.0, 9.0]);
        }
    }
}
