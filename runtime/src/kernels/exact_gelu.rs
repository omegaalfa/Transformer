#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExactGeluError {
    LengthMismatch,
    NonFiniteInput,
    NonFiniteResult,
}

pub fn exact_gelu_f32(input: &[f32], output: &mut [f32]) -> Result<(), ExactGeluError> {
    if input.len() != output.len() {
        return Err(ExactGeluError::LengthMismatch);
    }
    if input.iter().any(|value| !value.is_finite()) {
        return Err(ExactGeluError::NonFiniteInput);
    }

    for &value in input {
        exact_gelu_value(value as f64)?;
    }
    for (&value, destination) in input.iter().zip(output) {
        *destination = exact_gelu_value(value as f64)?;
    }
    Ok(())
}

fn exact_gelu_value(x: f64) -> Result<f32, ExactGeluError> {
    let result = 0.5 * x * (1.0 + libm::erf(x / std::f64::consts::SQRT_2));
    let output = result as f32;
    if !result.is_finite() || !output.is_finite() {
        return Err(ExactGeluError::NonFiniteResult);
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_erf_reference_for_finite_values() {
        let input = [-10.0, -1.0, -0.0001, -0.0, 0.0001, 1.0, 10.0];
        let mut output = [0.0; 7];
        exact_gelu_f32(&input, &mut output).unwrap();
        for (&actual, &value) in output.iter().zip(&input) {
            let x = value as f64;
            let expected = 0.5 * x * (1.0 + libm::erf(x / std::f64::consts::SQRT_2));
            assert!((actual as f64 - expected).abs() <= 1e-6 + 1e-6 * expected.abs());
        }
        assert!(output[3].is_sign_negative());
    }

    #[test]
    fn preserves_empty_and_rejects_non_finite_without_writing() {
        exact_gelu_f32(&[], &mut []).unwrap();
        for input in [[f32::NAN], [f32::INFINITY], [f32::NEG_INFINITY]] {
            let mut output = [99.0];
            assert_eq!(
                exact_gelu_f32(&input, &mut output),
                Err(ExactGeluError::NonFiniteInput)
            );
            assert_eq!(output, [99.0]);
        }
    }
}
