#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    Float32,
}

#[cfg(test)]
mod tests {
    use super::DType;

    #[test]
    fn starts_with_float32() {
        assert_eq!(DType::Float32, DType::Float32);
    }
}
