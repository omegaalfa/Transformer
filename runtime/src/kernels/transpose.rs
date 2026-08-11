#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransposeError {
    DimensionOverflow,
    LengthMismatch,
}

pub fn transpose_f32(
    input: &[f32],
    output: &mut [f32],
    rows: usize,
    columns: usize,
) -> Result<(), TransposeError> {
    let length = rows
        .checked_mul(columns)
        .ok_or(TransposeError::DimensionOverflow)?;

    if input.len() != length || output.len() != length {
        return Err(TransposeError::LengthMismatch);
    }

    for row in 0..rows {
        for column in 0..columns {
            let source_index = row * columns + column;
            let destination_index = column * rows + row;
            output[destination_index] = input[source_index];
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{transpose_f32, TransposeError};

    #[test]
    fn transposes_two_by_three_into_three_by_two() {
        let input = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut output = [0.0; 6];

        assert_eq!(transpose_f32(&input, &mut output, 2, 3), Ok(()));
        assert_eq!(output, [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn transposes_row_into_column() {
        let mut output = [0.0; 3];

        assert_eq!(transpose_f32(&[1.0, 2.0, 3.0], &mut output, 1, 3), Ok(()));
        assert_eq!(output, [1.0, 2.0, 3.0]);
    }

    #[test]
    fn transposes_square_matrix() {
        let mut output = [0.0; 4];

        assert_eq!(
            transpose_f32(&[1.0, 2.0, 3.0, 4.0], &mut output, 2, 2),
            Ok(())
        );
        assert_eq!(output, [1.0, 3.0, 2.0, 4.0]);
    }

    #[test]
    fn accepts_empty_shape() {
        assert_eq!(transpose_f32(&[], &mut [], 0, 3), Ok(()));
    }

    #[test]
    fn rejects_incorrect_buffer_length() {
        let mut output = [0.0; 6];

        assert_eq!(
            transpose_f32(&[1.0], &mut output, 2, 3),
            Err(TransposeError::LengthMismatch)
        );
    }
}
