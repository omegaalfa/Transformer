#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SoftmaxError {
    EmptyInput,
    LengthMismatch,
    NonFiniteInput,
    InvalidNormalization,
}

pub fn softmax_f32(input: &[f32], output: &mut [f32]) -> Result<(), SoftmaxError> {
    if input.is_empty() {
        return Err(SoftmaxError::EmptyInput);
    }

    if input.len() != output.len() {
        return Err(SoftmaxError::LengthMismatch);
    }

    if input.iter().any(|value| !value.is_finite()) {
        return Err(SoftmaxError::NonFiniteInput);
    }

    let mut maximum = input[0];

    for &value in &input[1..] {
        maximum = maximum.max(value);
    }

    let mut sum = 0.0;

    for (destination, &value) in output.iter_mut().zip(input) {
        let exponential = (value - maximum).exp();
        *destination = exponential;
        sum += exponential;
    }

    if !sum.is_finite() || sum <= 0.0 {
        return Err(SoftmaxError::InvalidNormalization);
    }

    for value in output {
        *value /= sum;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{softmax_f32, SoftmaxError};

    const TOLERANCE: f32 = 1.0e-5;
    const EXPECTED: [f32; 3] = [0.090_030_57, 0.244_728_48, 0.665_240_94];

    fn assert_approximately_equal(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());

        for (actual, expected) in actual.iter().zip(expected) {
            assert!((actual - expected).abs() <= TOLERANCE);
        }
    }

    #[test]
    fn computes_reference_vector() {
        let mut output = [0.0; 3];

        assert_eq!(softmax_f32(&[1.0, 2.0, 3.0], &mut output), Ok(()));
        assert_approximately_equal(&output, &EXPECTED);
        assert!((output.iter().sum::<f32>() - 1.0).abs() <= TOLERANCE);
    }

    #[test]
    fn remains_stable_for_large_positive_values() {
        let mut output = [0.0; 3];

        assert_eq!(softmax_f32(&[1000.0, 1001.0, 1002.0], &mut output), Ok(()));
        assert_approximately_equal(&output, &EXPECTED);
        assert!(output.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn distributes_equal_values_uniformly() {
        let mut output = [0.0; 3];

        assert_eq!(softmax_f32(&[0.0, 0.0, 0.0], &mut output), Ok(()));
        assert_approximately_equal(&output, &[1.0 / 3.0; 3]);
    }

    #[test]
    fn remains_finite_for_large_negative_values() {
        let mut output = [0.0; 3];

        assert_eq!(
            softmax_f32(&[-1000.0, -1001.0, -1002.0], &mut output),
            Ok(())
        );
        assert_approximately_equal(&output, &[0.665_240_94, 0.244_728_48, 0.090_030_57]);
        assert!(output.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn rejects_empty_input() {
        assert_eq!(softmax_f32(&[], &mut []), Err(SoftmaxError::EmptyInput));
    }

    #[test]
    fn rejects_non_finite_input() {
        let mut output = [0.0; 1];

        assert_eq!(
            softmax_f32(&[f32::INFINITY], &mut output),
            Err(SoftmaxError::NonFiniteInput)
        );
    }
}
