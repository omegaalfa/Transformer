use super::Shape;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OffsetError {
    RankMismatch {
        expected: usize,
        actual: usize,
    },
    IndexOutOfBounds {
        axis: usize,
        index: usize,
        dimension: usize,
    },
    OffsetOverflow,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Strides {
    values: Vec<usize>,
}

impl Strides {
    pub fn contiguous(shape: &Shape) -> Self {
        if shape.is_empty() {
            return Self {
                values: vec![0; shape.rank()],
            };
        }

        let mut values = vec![0; shape.rank()];
        let mut stride = 1;

        for axis in (0..shape.rank()).rev() {
            values[axis] = stride;
            stride *= shape.as_slice()[axis];
        }

        Self { values }
    }

    pub fn as_slice(&self) -> &[usize] {
        &self.values
    }

    pub fn offset(&self, shape: &Shape, indices: &[usize]) -> Result<usize, OffsetError> {
        if indices.len() != shape.rank() {
            return Err(OffsetError::RankMismatch {
                expected: shape.rank(),
                actual: indices.len(),
            });
        }

        let mut offset = 0usize;

        for (axis, ((&index, &dimension), &stride)) in indices
            .iter()
            .zip(shape.as_slice())
            .zip(&self.values)
            .enumerate()
        {
            if index >= dimension {
                return Err(OffsetError::IndexOutOfBounds {
                    axis,
                    index,
                    dimension,
                });
            }

            let axis_offset = index
                .checked_mul(stride)
                .ok_or(OffsetError::OffsetOverflow)?;
            offset = offset
                .checked_add(axis_offset)
                .ok_or(OffsetError::OffsetOverflow)?;
        }

        Ok(offset)
    }
}

#[cfg(test)]
mod tests {
    use super::{OffsetError, Strides};
    use crate::tensor::Shape;

    #[test]
    fn calculates_two_dimensional_row_major_strides() {
        let shape = Shape::new(vec![2, 3]).expect("valid shape");

        assert_eq!(Strides::contiguous(&shape).as_slice(), &[3, 1]);
    }

    #[test]
    fn calculates_three_dimensional_row_major_strides() {
        let shape = Shape::new(vec![2, 3, 4]).expect("valid shape");

        assert_eq!(Strides::contiguous(&shape).as_slice(), &[12, 4, 1]);
    }

    #[test]
    fn scalar_has_no_strides() {
        let shape = Shape::new(vec![]).expect("valid scalar shape");

        assert!(Strides::contiguous(&shape).as_slice().is_empty());
    }

    #[test]
    fn zero_sized_shapes_use_canonical_zero_strides() {
        let vector = Shape::new(vec![0]).expect("valid empty vector");
        let tensor = Shape::new(vec![2, 0, 3]).expect("valid empty tensor");

        assert_eq!(Strides::contiguous(&vector).as_slice(), &[0]);
        assert_eq!(Strides::contiguous(&tensor).as_slice(), &[0, 0, 0]);
    }

    #[test]
    fn calculates_offset() {
        let shape = Shape::new(vec![2, 3]).expect("valid shape");
        let strides = Strides::contiguous(&shape);

        assert_eq!(strides.offset(&shape, &[1, 2]), Ok(5));
    }

    #[test]
    fn scalar_offset_is_zero() {
        let shape = Shape::new(vec![]).expect("valid scalar shape");
        let strides = Strides::contiguous(&shape);

        assert_eq!(strides.offset(&shape, &[]), Ok(0));
    }

    #[test]
    fn rejects_out_of_bounds_index() {
        let shape = Shape::new(vec![2, 3]).expect("valid shape");
        let strides = Strides::contiguous(&shape);

        assert_eq!(
            strides.offset(&shape, &[2, 0]),
            Err(OffsetError::IndexOutOfBounds {
                axis: 0,
                index: 2,
                dimension: 2,
            })
        );
    }

    #[test]
    fn rejects_rank_mismatch() {
        let shape = Shape::new(vec![2, 3]).expect("valid shape");
        let strides = Strides::contiguous(&shape);

        assert_eq!(
            strides.offset(&shape, &[1]),
            Err(OffsetError::RankMismatch {
                expected: 2,
                actual: 1,
            })
        );
    }

    #[test]
    fn zero_sized_shape_has_no_valid_index() {
        let shape = Shape::new(vec![0]).expect("valid empty shape");
        let strides = Strides::contiguous(&shape);

        assert_eq!(
            strides.offset(&shape, &[0]),
            Err(OffsetError::IndexOutOfBounds {
                axis: 0,
                index: 0,
                dimension: 0,
            })
        );
    }
}
