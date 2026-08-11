#[derive(Debug, PartialEq)]
pub struct Storage {
    data: Vec<f32>,
}

impl Storage {
    pub fn from_vec(data: Vec<f32>) -> Self {
        Self { data }
    }

    pub fn zeros(length: usize) -> Self {
        Self {
            data: vec![0.0; length],
        }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }

    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        &mut self.data
    }

    pub fn into_vec(self) -> Vec<f32> {
        self.data
    }
}

#[cfg(test)]
mod tests {
    use super::Storage;

    #[test]
    fn from_vec_preserves_owned_data() {
        let storage = Storage::from_vec(vec![1.0, 2.0, 3.0]);

        assert_eq!(storage.as_slice(), &[1.0, 2.0, 3.0]);
        assert_eq!(storage.len(), 3);
        assert!(!storage.is_empty());
    }

    #[test]
    fn zeros_allocates_requested_length() {
        let storage = Storage::zeros(4);

        assert_eq!(storage.as_slice(), &[0.0, 0.0, 0.0, 0.0]);
        assert_eq!(storage.len(), 4);
    }

    #[test]
    fn supports_empty_storage() {
        let from_vec = Storage::from_vec(vec![]);
        let zeros = Storage::zeros(0);

        assert!(from_vec.is_empty());
        assert!(zeros.is_empty());
        assert_eq!(from_vec.len(), 0);
        assert_eq!(zeros.len(), 0);
    }

    #[test]
    fn as_slice_exposes_read_only_data() {
        let storage = Storage::from_vec(vec![2.5, -1.0]);
        let data = storage.as_slice();

        assert_eq!(data[0], 2.5);
        assert_eq!(data[1], -1.0);
    }

    #[test]
    fn as_mut_slice_allows_controlled_mutation() {
        let mut storage = Storage::zeros(3);

        storage.as_mut_slice().copy_from_slice(&[1.0, 2.0, 3.0]);

        assert_eq!(storage.as_slice(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn into_vec_transfers_ownership() {
        let storage = Storage::from_vec(vec![4.0, 5.0]);
        let data = storage.into_vec();

        assert_eq!(data, vec![4.0, 5.0]);
    }
}
