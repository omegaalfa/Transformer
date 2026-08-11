#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShapeError {
    ElementCountOverflow,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Shape {
    dimensions: Vec<usize>,
    numel: usize,
}

impl Shape {
    pub fn new(dimensions: Vec<usize>) -> Result<Self, ShapeError> {
        let numel = if dimensions.contains(&0) {
            0
        } else {
            dimensions.iter().try_fold(1usize, |total, &dimension| {
                total
                    .checked_mul(dimension)
                    .ok_or(ShapeError::ElementCountOverflow)
            })?
        };

        Ok(Self { dimensions, numel })
    }

    pub fn rank(&self) -> usize {
        self.dimensions.len()
    }

    pub fn numel(&self) -> usize {
        self.numel
    }

    pub fn as_slice(&self) -> &[usize] {
        &self.dimensions
    }

    pub fn is_empty(&self) -> bool {
        self.numel == 0
    }
}

#[cfg(test)]
mod tests {
    use super::{Shape, ShapeError};

    #[test]
    fn calculates_rank_and_numel() {
        let shape = Shape::new(vec![2, 3]).expect("valid shape");

        assert_eq!(shape.rank(), 2);
        assert_eq!(shape.numel(), 6);
        assert_eq!(shape.as_slice(), &[2, 3]);
    }

    #[test]
    fn calculates_three_dimensional_numel() {
        let shape = Shape::new(vec![2, 3, 4]).expect("valid shape");

        assert_eq!(shape.rank(), 3);
        assert_eq!(shape.numel(), 24);
    }

    #[test]
    fn supports_rank_zero_scalar() {
        let shape = Shape::new(vec![]).expect("valid scalar shape");

        assert_eq!(shape.rank(), 0);
        assert_eq!(shape.numel(), 1);
        assert!(!shape.is_empty());
    }

    #[test]
    fn supports_zero_sized_dimensions() {
        let vector = Shape::new(vec![0]).expect("valid empty vector");
        let matrix = Shape::new(vec![2, 0, 3]).expect("valid empty matrix");

        assert_eq!(vector.numel(), 0);
        assert_eq!(matrix.numel(), 0);
        assert!(vector.is_empty());
        assert!(matrix.is_empty());
    }

    #[test]
    fn zero_dimension_short_circuits_irrelevant_products() {
        let shape = Shape::new(vec![usize::MAX, usize::MAX, 0]).expect("zero-sized shape");

        assert_eq!(shape.numel(), 0);
    }

    #[test]
    fn rejects_numel_overflow() {
        assert_eq!(
            Shape::new(vec![usize::MAX, 2]),
            Err(ShapeError::ElementCountOverflow)
        );
    }
}
