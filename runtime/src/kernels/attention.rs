use super::linear::linear_last_dim_f32;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttentionError {
    InvalidDimensions,
    LengthMismatch,
    InvalidMask,
    FullyMasked,
    NonFiniteInput,
    NonFiniteResult,
    Projection,
}

#[allow(clippy::too_many_arguments)]
pub fn multi_head_attention_f32(
    input: &[f32],
    q_weight: &[f32],
    k_weight: &[f32],
    v_weight: &[f32],
    out_weight: &[f32],
    mask: Option<&[u8]>,
    output: &mut [f32],
    batch: usize,
    sequence: usize,
    dimensions: usize,
    heads: usize,
) -> Result<(), AttentionError> {
    if dimensions == 0 || heads == 0 || dimensions % heads != 0 {
        return Err(AttentionError::InvalidDimensions);
    }
    let tokens = batch
        .checked_mul(sequence)
        .ok_or(AttentionError::InvalidDimensions)?;
    let values_length = tokens
        .checked_mul(dimensions)
        .ok_or(AttentionError::InvalidDimensions)?;
    let weight_length = dimensions
        .checked_mul(dimensions)
        .ok_or(AttentionError::InvalidDimensions)?;
    if input.len() != values_length
        || output.len() != values_length
        || [q_weight, k_weight, v_weight, out_weight]
            .iter()
            .any(|weight| weight.len() != weight_length)
    {
        return Err(AttentionError::LengthMismatch);
    }
    if let Some(mask) = mask {
        if mask.len() != tokens || mask.iter().any(|&value| value > 1) {
            return Err(AttentionError::InvalidMask);
        }
    }
    if input
        .iter()
        .chain(q_weight)
        .chain(k_weight)
        .chain(v_weight)
        .chain(out_weight)
        .any(|value| !value.is_finite())
    {
        return Err(AttentionError::NonFiniteInput);
    }
    if tokens == 0 {
        return Ok(());
    }
    if let Some(mask) = mask {
        for row in mask.chunks_exact(sequence) {
            if !row.iter().any(|&value| value == 1) {
                return Err(AttentionError::FullyMasked);
            }
        }
    }

    let head_dimensions = dimensions / heads;
    let scale = (1.0f64 / (head_dimensions as f64).sqrt()) as f32;
    if !scale.is_finite() || scale <= 0.0 {
        return Err(AttentionError::NonFiniteResult);
    }

    let mut q = vec![0.0; values_length];
    let mut k = vec![0.0; values_length];
    let mut v = vec![0.0; values_length];
    for (weight, destination) in [q_weight, k_weight, v_weight].into_iter().zip([
        q.as_mut_slice(),
        k.as_mut_slice(),
        v.as_mut_slice(),
    ]) {
        linear_last_dim_f32(input, weight, None, destination, dimensions, dimensions)
            .map_err(|_| AttentionError::Projection)?;
        if destination.iter().any(|value| !value.is_finite()) {
            return Err(AttentionError::NonFiniteResult);
        }
    }

    let score_length = batch
        .checked_mul(heads)
        .and_then(|value| value.checked_mul(sequence))
        .and_then(|value| value.checked_mul(sequence))
        .ok_or(AttentionError::InvalidDimensions)?;
    let mut probabilities = vec![0.0; score_length];
    for batch_index in 0..batch {
        for head in 0..heads {
            for query in 0..sequence {
                let row_start = ((batch_index * heads + head) * sequence + query) * sequence;
                let row = &mut probabilities[row_start..row_start + sequence];
                let mut maximum = f32::NEG_INFINITY;
                for key in 0..sequence {
                    if mask.is_some_and(|values| values[batch_index * sequence + key] == 0) {
                        continue;
                    }
                    let mut score = 0.0f32;
                    for inner in 0..head_dimensions {
                        let q_index = (batch_index * sequence + query) * dimensions
                            + head * head_dimensions
                            + inner;
                        let k_index = (batch_index * sequence + key) * dimensions
                            + head * head_dimensions
                            + inner;
                        score += q[q_index] * k[k_index];
                    }
                    score *= scale;
                    if !score.is_finite() {
                        return Err(AttentionError::NonFiniteResult);
                    }
                    row[key] = score;
                    maximum = maximum.max(score);
                }
                if !maximum.is_finite() {
                    return Err(AttentionError::FullyMasked);
                }
                let mut sum = 0.0f32;
                for key in 0..sequence {
                    if mask.is_some_and(|values| values[batch_index * sequence + key] == 0) {
                        row[key] = 0.0;
                    } else {
                        row[key] = (row[key] - maximum).exp();
                        sum += row[key];
                    }
                }
                if !sum.is_finite() || sum <= 0.0 {
                    return Err(AttentionError::NonFiniteResult);
                }
                for value in row {
                    *value /= sum;
                }
            }
        }
    }

    let mut merged = vec![0.0; values_length];
    for batch_index in 0..batch {
        for head in 0..heads {
            for query in 0..sequence {
                let probability_start =
                    ((batch_index * heads + head) * sequence + query) * sequence;
                for inner in 0..head_dimensions {
                    let mut value = 0.0f32;
                    for key in 0..sequence {
                        let v_index = (batch_index * sequence + key) * dimensions
                            + head * head_dimensions
                            + inner;
                        value += probabilities[probability_start + key] * v[v_index];
                    }
                    if !value.is_finite() {
                        return Err(AttentionError::NonFiniteResult);
                    }
                    merged[(batch_index * sequence + query) * dimensions
                        + head * head_dimensions
                        + inner] = value;
                }
            }
        }
    }

    let mut result = vec![0.0; values_length];
    linear_last_dim_f32(
        &merged,
        out_weight,
        None,
        &mut result,
        dimensions,
        dimensions,
    )
    .map_err(|_| AttentionError::Projection)?;
    if result.iter().any(|value| !value.is_finite()) {
        return Err(AttentionError::NonFiniteResult);
    }
    output.copy_from_slice(&result);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_projections_compute_attention() {
        let input = [1.0, 0.0, 0.0, 1.0];
        let identity = [1.0, 0.0, 0.0, 1.0];
        let mut output = [0.0; 4];
        multi_head_attention_f32(
            &input,
            &identity,
            &identity,
            &identity,
            &identity,
            None,
            &mut output,
            1,
            2,
            2,
            1,
        )
        .unwrap();
        assert!(output.iter().all(|value| value.is_finite()));
        assert!((output[0] - 0.669_761_54).abs() < 1e-6);
        assert!((output[1] - 0.330_238_46).abs() < 1e-6);
    }

    #[test]
    fn mask_excludes_keys_and_rejects_fully_masked_batch() {
        let input = [1.0, 2.0, 3.0, 4.0];
        let identity = [1.0, 0.0, 0.0, 1.0];
        let mut output = [0.0; 4];
        assert_eq!(
            multi_head_attention_f32(
                &input,
                &identity,
                &identity,
                &identity,
                &identity,
                Some(&[1, 0]),
                &mut output,
                1,
                2,
                2,
                1,
            ),
            Ok(())
        );
        assert_eq!(output, [1.0, 2.0, 1.0, 2.0]);
        assert_eq!(
            multi_head_attention_f32(
                &input,
                &identity,
                &identity,
                &identity,
                &identity,
                Some(&[0, 0]),
                &mut output,
                1,
                2,
                2,
                1,
            ),
            Err(AttentionError::FullyMasked)
        );
    }

    #[test]
    fn supports_empty_outer_shapes_and_rejects_invalid_contracts() {
        let identity = [1.0, 0.0, 0.0, 1.0];
        assert_eq!(
            multi_head_attention_f32(
                &[],
                &identity,
                &identity,
                &identity,
                &identity,
                Some(&[]),
                &mut [],
                0,
                3,
                2,
                1,
            ),
            Ok(())
        );
        assert_eq!(
            multi_head_attention_f32(
                &[],
                &identity,
                &identity,
                &identity,
                &identity,
                None,
                &mut [],
                1,
                0,
                2,
                1,
            ),
            Ok(())
        );
        assert_eq!(
            multi_head_attention_f32(&[], &[], &[], &[], &[], None, &mut [], 0, 0, 0, 0,),
            Err(AttentionError::InvalidDimensions)
        );
    }

    #[test]
    fn rejects_non_finite_without_writing_output() {
        let identity = [1.0, 0.0, 0.0, 1.0];
        let mut output = [7.0, 7.0];
        assert_eq!(
            multi_head_attention_f32(
                &[f32::NAN, 1.0],
                &identity,
                &identity,
                &identity,
                &identity,
                None,
                &mut output,
                1,
                1,
                2,
                1,
            ),
            Err(AttentionError::NonFiniteInput)
        );
        assert_eq!(output, [7.0, 7.0]);
    }
}
