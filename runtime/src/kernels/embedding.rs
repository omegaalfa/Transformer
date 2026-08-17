#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbeddingError {
    InvalidDimensions,
    ElementCountOverflow,
    WeightLengthMismatch,
    OutputLengthMismatch,
    TokenOutOfRange,
}

/// Copies selected rows of a contiguous row-major Float32 weight matrix.
///
/// Metadata and token IDs are validated before the first output write, so an
/// error leaves both `weight` and `output` unchanged.
pub fn embedding_f32(
    token_ids: &[i64],
    weight: &[f32],
    output: &mut [f32],
    vocab_size: usize,
    embedding_dim: usize,
) -> Result<(), EmbeddingError> {
    if vocab_size == 0 || embedding_dim == 0 {
        return Err(EmbeddingError::InvalidDimensions);
    }

    let expected_weight = vocab_size
        .checked_mul(embedding_dim)
        .ok_or(EmbeddingError::ElementCountOverflow)?;
    if weight.len() != expected_weight {
        return Err(EmbeddingError::WeightLengthMismatch);
    }

    let expected_output = token_ids
        .len()
        .checked_mul(embedding_dim)
        .ok_or(EmbeddingError::ElementCountOverflow)?;
    if output.len() != expected_output {
        return Err(EmbeddingError::OutputLengthMismatch);
    }

    for &token_id in token_ids {
        if token_id < 0 || usize::try_from(token_id).map_or(true, |id| id >= vocab_size) {
            return Err(EmbeddingError::TokenOutOfRange);
        }
    }

    for (output_row, &token_id) in output.chunks_exact_mut(embedding_dim).zip(token_ids) {
        let token_index = usize::try_from(token_id).map_err(|_| EmbeddingError::TokenOutOfRange)?;
        let row_start = token_index * embedding_dim;
        output_row.copy_from_slice(&weight[row_start..row_start + embedding_dim]);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const WEIGHT: [f32; 12] = [0.0, 0.1, 0.2, 1.0, 1.1, 1.2, 2.0, 2.1, 2.2, 3.0, 3.1, 3.2];

    #[test]
    fn copies_known_rows_in_token_order() {
        let mut output = [0.0; 9];
        embedding_f32(&[2, 0, 3], &WEIGHT, &mut output, 4, 3).unwrap();
        assert_eq!(output, [2.0, 2.1, 2.2, 0.0, 0.1, 0.2, 3.0, 3.1, 3.2]);
    }

    #[test]
    fn supports_repeated_tokens_and_multiple_batches_as_flat_rows() {
        let mut output = [0.0; 12];
        embedding_f32(&[1, 1, 3, 0], &WEIGHT, &mut output, 4, 3).unwrap();
        assert_eq!(
            output,
            [1.0, 1.1, 1.2, 1.0, 1.1, 1.2, 3.0, 3.1, 3.2, 0.0, 0.1, 0.2]
        );
    }

    #[test]
    fn accepts_empty_token_input() {
        assert_eq!(embedding_f32(&[], &WEIGHT, &mut [], 4, 3), Ok(()));
    }

    #[test]
    fn rejects_invalid_ids_before_writing() {
        for ids in [[0, -1], [0, 4]] {
            let mut output = [99.0; 6];
            assert_eq!(
                embedding_f32(&ids, &WEIGHT, &mut output, 4, 3),
                Err(EmbeddingError::TokenOutOfRange)
            );
            assert_eq!(output, [99.0; 6]);
        }
    }

    #[test]
    fn rejects_incorrect_lengths_and_dimension_overflow() {
        assert_eq!(
            embedding_f32(&[0], &WEIGHT, &mut [0.0; 2], 4, 3),
            Err(EmbeddingError::OutputLengthMismatch)
        );
        assert_eq!(
            embedding_f32(&[0], &WEIGHT[..11], &mut [0.0; 3], 4, 3),
            Err(EmbeddingError::WeightLengthMismatch)
        );
        assert_eq!(
            embedding_f32(&[], &[], &mut [], usize::MAX, 2),
            Err(EmbeddingError::ElementCountOverflow)
        );
    }

    #[test]
    fn preserves_weight() {
        let before = WEIGHT;
        embedding_f32(&[3, 1], &WEIGHT, &mut [0.0; 6], 4, 3).unwrap();
        assert_eq!(WEIGHT, before);
    }
}
