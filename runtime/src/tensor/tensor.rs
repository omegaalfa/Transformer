use super::{DType, Shape, Storage, Strides};

/// A contiguous, row-major Float32 tensor with exclusive ownership of its data.
#[derive(Debug, PartialEq)]
pub struct Tensor {
    storage: Storage,
    shape: Shape,
    strides: Strides,
    dtype: DType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorError {
    StorageLengthMismatch { expected: usize, actual: usize },
}

impl Tensor {
    pub fn new(storage: Storage, shape: Shape) -> Result<Self, TensorError> {
        let expected = shape.numel();
        let actual = storage.len();

        if actual != expected {
            return Err(TensorError::StorageLengthMismatch { expected, actual });
        }

        let strides = Strides::contiguous(&shape);

        Ok(Self {
            storage,
            shape,
            strides,
            dtype: DType::Float32,
        })
    }

    pub fn from_vec(data: Vec<f32>, shape: Shape) -> Result<Self, TensorError> {
        Self::new(Storage::from_vec(data), shape)
    }

    pub fn zeros(shape: Shape) -> Self {
        let storage = Storage::zeros(shape.numel());
        let strides = Strides::contiguous(&shape);

        Self {
            storage,
            shape,
            strides,
            dtype: DType::Float32,
        }
    }

    pub fn storage(&self) -> &Storage {
        &self.storage
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    pub fn strides(&self) -> &Strides {
        &self.strides
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn rank(&self) -> usize {
        self.shape.rank()
    }

    pub fn numel(&self) -> usize {
        self.shape.numel()
    }

    pub fn is_empty(&self) -> bool {
        self.storage.is_empty()
    }

    pub fn as_slice(&self) -> &[f32] {
        self.storage.as_slice()
    }

    pub fn into_storage(self) -> Storage {
        self.storage
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn composes_storage_shape_strides_and_dtype() {
        let shape = Shape::new(vec![2, 3]).unwrap();
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape).unwrap();

        assert_eq!(tensor.shape().as_slice(), &[2, 3]);
        assert_eq!(tensor.strides().as_slice(), &[3, 1]);
        assert_eq!(tensor.dtype(), DType::Float32);
        assert_eq!(tensor.rank(), 2);
        assert_eq!(tensor.numel(), 6);
        assert!(!tensor.is_empty());
        assert_eq!(tensor.as_slice(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn rejects_storage_shorter_than_numel() {
        let shape = Shape::new(vec![2, 3]).unwrap();

        assert_eq!(
            Tensor::from_vec(vec![1.0; 5], shape),
            Err(TensorError::StorageLengthMismatch {
                expected: 6,
                actual: 5,
            })
        );
    }

    #[test]
    fn rejects_storage_longer_than_numel() {
        let shape = Shape::new(vec![2, 3]).unwrap();

        assert_eq!(
            Tensor::from_vec(vec![1.0; 7], shape),
            Err(TensorError::StorageLengthMismatch {
                expected: 6,
                actual: 7,
            })
        );
    }

    #[test]
    fn supports_scalar_shape() {
        let shape = Shape::new(vec![]).unwrap();
        let tensor = Tensor::from_vec(vec![42.0], shape).unwrap();

        assert_eq!(tensor.rank(), 0);
        assert_eq!(tensor.numel(), 1);
        assert_eq!(tensor.strides().as_slice(), &[]);
        assert_eq!(tensor.as_slice(), &[42.0]);
    }

    #[test]
    fn supports_zero_sized_shape() {
        let shape = Shape::new(vec![2, 0, 3]).unwrap();
        let tensor = Tensor::from_vec(vec![], shape).unwrap();

        assert_eq!(tensor.numel(), 0);
        assert!(tensor.is_empty());
        assert_eq!(tensor.strides().as_slice(), &[0, 0, 0]);
    }

    #[test]
    fn zeros_allocates_exactly_numel_values() {
        let shape = Shape::new(vec![2, 2]).unwrap();
        let tensor = Tensor::zeros(shape);

        assert_eq!(tensor.as_slice(), &[0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn into_storage_transfers_exclusive_ownership() {
        let shape = Shape::new(vec![2]).unwrap();
        let tensor = Tensor::from_vec(vec![3.0, 4.0], shape).unwrap();

        assert_eq!(tensor.into_storage().into_vec(), vec![3.0, 4.0]);
    }
}
