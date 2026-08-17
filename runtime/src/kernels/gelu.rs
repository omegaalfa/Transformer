#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GeluError {
    LengthMismatch,
    NonFiniteInput,
    NonFiniteResult,
}

pub fn gelu_f32(input: &[f32], output: &mut [f32]) -> Result<(), GeluError> {
    if input.len() != output.len() {
        return Err(GeluError::LengthMismatch);
    }
    if input.iter().any(|value| !value.is_finite()) {
        return Err(GeluError::NonFiniteInput);
    }

    // Validate every result before mutating the destination.
    for &value in input {
        gelu_value(value as f64)?;
    }
    for (&value, destination) in input.iter().zip(output) {
        *destination = gelu_value(value as f64)?;
    }
    Ok(())
}

fn gelu_value(x: f64) -> Result<f32, GeluError> {
    let coefficient = (2.0_f64 / std::f64::consts::PI).sqrt();
    let result = 0.5 * x * (1.0 + (coefficient * (x + 0.044_715 * x * x * x)).tanh());
    let output = result as f32;
    if !result.is_finite() || !output.is_finite() {
        return Err(GeluError::NonFiniteResult);
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn computes_tanh_gelu_for_negative_zero_and_positive_values() {
        let input = [-10.0, -1.0, -0.0001, 0.0, 0.0001, 1.0, 10.0];
        let mut output = [0.0; 7];
        gelu_f32(&input, &mut output).unwrap();
        for (&actual, &x) in output.iter().zip(&input) {
            let x = x as f64;
            let expected = 0.5
                * x
                * (1.0
                    + ((2.0 / std::f64::consts::PI).sqrt() * (x + 0.044_715 * x * x * x)).tanh());
            assert!((actual as f64 - expected).abs() <= 1e-6 + 1e-6 * expected.abs());
        }
    }

    #[test]
    fn accepts_empty_slices_and_preserves_input() {
        gelu_f32(&[], &mut []).unwrap();
        let input = [-3.0, 0.0, 3.0];
        let before = input;
        gelu_f32(&input, &mut [0.0; 3]).unwrap();
        assert_eq!(input, before);
    }

    #[test]
    fn rejects_invalid_values_and_lengths_without_writing() {
        for input in [[f32::NAN], [f32::INFINITY], [f32::NEG_INFINITY]] {
            let mut output = [99.0];
            assert_eq!(
                gelu_f32(&input, &mut output),
                Err(GeluError::NonFiniteInput)
            );
            assert_eq!(output, [99.0]);
        }
        let mut output = [99.0];
        assert_eq!(
            gelu_f32(&[1.0, 2.0], &mut output),
            Err(GeluError::LengthMismatch)
        );
        assert_eq!(output, [99.0]);
        assert_eq!(gelu_value(f64::MAX), Err(GeluError::NonFiniteResult));
    }
}
