use super::linear::linear_last_dim_f32;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BertAttentionError {
    InvalidDimensions,
    LengthMismatch,
    InvalidMask,
    FullyMasked,
    NonFiniteInput,
    NonFiniteResult,
    Projection,
}

#[allow(clippy::too_many_arguments)]
pub fn bert_self_attention_f32(
    input: &[f32],
    q_weight: &[f32],
    q_bias: &[f32],
    k_weight: &[f32],
    k_bias: &[f32],
    v_weight: &[f32],
    v_bias: &[f32],
    out_weight: &[f32],
    out_bias: &[f32],
    mask: Option<&[u8]>,
    output: &mut [f32],
    batch: usize,
    sequence: usize,
    dimensions: usize,
    heads: usize,
) -> Result<(), BertAttentionError> {
    if dimensions == 0 || heads == 0 || dimensions % heads != 0 {
        return Err(BertAttentionError::InvalidDimensions);
    }
    let tokens = batch
        .checked_mul(sequence)
        .ok_or(BertAttentionError::InvalidDimensions)?;
    let values_length = tokens
        .checked_mul(dimensions)
        .ok_or(BertAttentionError::InvalidDimensions)?;
    let weight_length = dimensions
        .checked_mul(dimensions)
        .ok_or(BertAttentionError::InvalidDimensions)?;
    let weights = [q_weight, k_weight, v_weight, out_weight];
    let biases = [q_bias, k_bias, v_bias, out_bias];
    if input.len() != values_length
        || output.len() != values_length
        || weights.iter().any(|weight| weight.len() != weight_length)
        || biases.iter().any(|bias| bias.len() != dimensions)
    {
        return Err(BertAttentionError::LengthMismatch);
    }
    if let Some(mask) = mask {
        if mask.len() != tokens || mask.iter().any(|&value| value > 1) {
            return Err(BertAttentionError::InvalidMask);
        }
    }
    if input
        .iter()
        .chain(weights.into_iter().flatten())
        .chain(biases.into_iter().flatten())
        .any(|value| !value.is_finite())
    {
        return Err(BertAttentionError::NonFiniteInput);
    }
    if tokens == 0 {
        return Ok(());
    }
    if let Some(mask) = mask {
        for row in mask.chunks_exact(sequence) {
            if !row.iter().any(|&value| value == 1) {
                return Err(BertAttentionError::FullyMasked);
            }
        }
    }

    let head_dimensions = dimensions / heads;
    let scale = (1.0f64 / (head_dimensions as f64).sqrt()) as f32;
    let mut q = vec![0.0; values_length];
    let mut k = vec![0.0; values_length];
    let mut v = vec![0.0; values_length];
    for ((weight, bias), destination) in
        [(q_weight, q_bias), (k_weight, k_bias), (v_weight, v_bias)]
            .into_iter()
            .zip([q.as_mut_slice(), k.as_mut_slice(), v.as_mut_slice()])
    {
        linear_last_dim_f32(
            input,
            weight,
            Some(bias),
            destination,
            dimensions,
            dimensions,
        )
        .map_err(|_| BertAttentionError::Projection)?;
    }

    let score_length = batch
        .checked_mul(heads)
        .and_then(|value| value.checked_mul(sequence))
        .and_then(|value| value.checked_mul(sequence))
        .ok_or(BertAttentionError::InvalidDimensions)?;
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
                        return Err(BertAttentionError::NonFiniteResult);
                    }
                    row[key] = score;
                    maximum = maximum.max(score);
                }
                if !maximum.is_finite() {
                    return Err(BertAttentionError::FullyMasked);
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
                    return Err(BertAttentionError::NonFiniteResult);
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
                        return Err(BertAttentionError::NonFiniteResult);
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
        Some(out_bias),
        &mut result,
        dimensions,
        dimensions,
    )
    .map_err(|_| BertAttentionError::Projection)?;
    if result.iter().any(|value| !value.is_finite()) {
        return Err(BertAttentionError::NonFiniteResult);
    }
    output.copy_from_slice(&result);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn applies_projection_biases_and_mask() {
        let input = [1.0, 2.0, 3.0, 4.0];
        let identity = [1.0, 0.0, 0.0, 1.0];
        let zero = [0.0, 0.0];
        let output_bias = [0.5, -0.5];
        let mut output = [0.0; 4];
        bert_self_attention_f32(
            &input,
            &identity,
            &zero,
            &identity,
            &zero,
            &identity,
            &zero,
            &identity,
            &output_bias,
            Some(&[1, 0]),
            &mut output,
            1,
            2,
            2,
            1,
        )
        .unwrap();
        assert_eq!(output, [1.5, 1.5, 1.5, 1.5]);
    }

    #[test]
    fn supports_empty_and_rejects_fully_masked_without_publication() {
        let identity = [1.0, 0.0, 0.0, 1.0];
        let zero = [0.0, 0.0];
        assert_eq!(
            bert_self_attention_f32(
                &[],
                &identity,
                &zero,
                &identity,
                &zero,
                &identity,
                &zero,
                &identity,
                &zero,
                Some(&[]),
                &mut [],
                0,
                3,
                2,
                1,
            ),
            Ok(())
        );
        let mut output = [7.0; 2];
        assert_eq!(
            bert_self_attention_f32(
                &[1.0, 2.0],
                &identity,
                &zero,
                &identity,
                &zero,
                &identity,
                &zero,
                &identity,
                &zero,
                Some(&[0]),
                &mut output,
                1,
                1,
                2,
                1,
            ),
            Err(BertAttentionError::FullyMasked)
        );
        assert_eq!(output, [7.0; 2]);
    }
}
