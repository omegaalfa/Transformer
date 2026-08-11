mod dtype;
mod shape;
mod storage;
mod strides;
#[path = "tensor.rs"]
mod value;

pub use dtype::DType;
pub use shape::{Shape, ShapeError};
pub use storage::Storage;
pub use strides::{OffsetError, Strides};
pub use value::{Tensor, TensorError};
