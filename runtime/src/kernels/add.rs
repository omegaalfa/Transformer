#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AddError {
    LengthMismatch,
}

pub fn add_f32(a: &[f32], b: &[f32], output: &mut [f32]) -> Result<(), AddError> {
    if a.len() != b.len() || a.len() != output.len() {
        return Err(AddError::LengthMismatch);
    }

    for index in 0..a.len() {
        output[index] = a[index] + b[index];
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{add_f32, AddError};

    #[test]
    fn adds_float32_slices() {
        let a = [1.5, -2.25, 0.5];
        let b = [2.5, 1.25, 4.0];
        let mut output = [0.0; 3];

        assert_eq!(add_f32(&a, &b, &mut output), Ok(()));
        assert_eq!(output, [4.0, -1.0, 4.5]);
    }

    #[test]
    fn accepts_empty_slices() {
        let mut output = [];

        assert_eq!(add_f32(&[], &[], &mut output), Ok(()));
    }

    #[test]
    fn rejects_different_lengths() {
        let mut output = [0.0; 1];

        assert_eq!(
            add_f32(&[1.0], &[2.0, 3.0], &mut output),
            Err(AddError::LengthMismatch)
        );
    }
}
