use crate::tensor::Tensor;

/// Opaque C ABI representation of a native Tensor.
///
/// C consumers must only forward pointers to this type. Its fields and Rust
/// layout are private and must never be reproduced in a C header.
#[repr(C)]
#[derive(Debug, PartialEq)]
pub struct TransformerTensor {
    tensor: Tensor,
}

impl TransformerTensor {
    pub(super) fn new(tensor: Tensor) -> Self {
        Self { tensor }
    }

    pub(super) fn tensor(&self) -> &Tensor {
        &self.tensor
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Shape;

    #[test]
    fn wraps_tensor_without_exposing_or_copying_its_storage() {
        let shape = Shape::new(vec![2]).unwrap();
        let tensor = Tensor::from_vec(vec![3.0, 4.0], shape).unwrap();
        let handle = TransformerTensor::new(tensor);

        assert_eq!(handle.tensor().shape().as_slice(), &[2]);
        assert_eq!(handle.tensor().as_slice(), &[3.0, 4.0]);
    }

    #[test]
    fn owns_scalar_tensor_until_handle_is_dropped() {
        let shape = Shape::new(vec![]).unwrap();
        let tensor = Tensor::from_vec(vec![42.0], shape).unwrap();
        let handle = TransformerTensor::new(tensor);

        assert_eq!(handle.tensor().as_slice(), &[42.0]);
    }
}
