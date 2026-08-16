use super::blas;
use super::matmul::{matmul_cache_friendly_f32, matmul_f32, matmul_tiled_f32, MatmulError};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MatmulBackend {
    Reference,
    CacheFriendly,
    Tiled,
    Blas,
}

/// Empirical single-thread CPU dispatch policy for the Transformer projection
/// shapes benchmarked on the reference host. Unknown shapes deliberately use
/// the scalar reference kernel until they have their own measured policy.
const BLAS_RULES: [BlasRule; 3] = [
    BlasRule::new(768, 768, 4),
    BlasRule::new(768, 3072, 2),
    BlasRule::new(3072, 768, 2),
];

#[derive(Debug, Clone, Copy)]
struct BlasRule {
    k: usize,
    n: usize,
    minimum_m: usize,
}

impl BlasRule {
    const fn new(k: usize, n: usize, minimum_m: usize) -> Self {
        Self { k, n, minimum_m }
    }
}

pub(crate) fn matmul_dispatch_f32(
    a: &[f32],
    b: &[f32],
    output: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) -> Result<(), MatmulError> {
    let blas_available = blas::is_available();
    match select_backend(m, k, n, blas_available) {
        MatmulBackend::Reference => matmul_f32(a, b, output, m, k, n),
        MatmulBackend::CacheFriendly => matmul_cache_friendly_f32(a, b, output, m, k, n),
        MatmulBackend::Tiled => matmul_tiled_f32(a, b, output, m, k, n),
        MatmulBackend::Blas => blas::try_matmul_blas_f32(a, b, output, m, k, n)
            .unwrap_or_else(|| matmul_tiled_f32(a, b, output, m, k, n)),
    }
}

pub(crate) fn select_backend(m: usize, k: usize, n: usize, blas_available: bool) -> MatmulBackend {
    if m == 1 {
        return MatmulBackend::CacheFriendly;
    }

    let Some(rule) = BLAS_RULES.iter().find(|rule| rule.k == k && rule.n == n) else {
        return MatmulBackend::Reference;
    };

    if m >= rule.minimum_m && blas_available {
        MatmulBackend::Blas
    } else {
        MatmulBackend::Tiled
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn policy_uses_cache_friendly_for_matrix_vector() {
        for (k, n) in [(768, 768), (768, 3072), (3072, 768), (17, 19)] {
            assert_eq!(select_backend(1, k, n, true), MatmulBackend::CacheFriendly);
        }
    }

    #[test]
    fn policy_uses_measured_blas_thresholds() {
        for (m, k, n) in [(4, 768, 768), (2, 768, 3072), (2, 3072, 768)] {
            assert_eq!(select_backend(m, k, n, true), MatmulBackend::Blas);
        }
        assert_eq!(select_backend(2, 768, 768, true), MatmulBackend::Tiled);
    }

    #[test]
    fn policy_falls_back_to_rust_when_blas_is_unavailable() {
        for (m, k, n) in [(4, 768, 768), (2, 768, 3072), (2, 3072, 768)] {
            assert_eq!(select_backend(m, k, n, false), MatmulBackend::Tiled);
        }
    }

    #[test]
    fn policy_keeps_unknown_shapes_on_reference_kernel() {
        assert_eq!(select_backend(8, 17, 19, true), MatmulBackend::Reference);
    }

    #[test]
    fn dispatcher_matches_reference_for_every_policy_branch() {
        for (m, k, n) in [
            (1, 768, 768),
            (2, 768, 768),
            (4, 768, 768),
            (2, 768, 3072),
            (2, 3072, 768),
            (3, 5, 7),
        ] {
            let a: Vec<f32> = (0..m * k).map(|index| (index % 17) as f32 * 0.01).collect();
            let b: Vec<f32> = (0..k * n).map(|index| (index % 13) as f32 * 0.02).collect();
            let mut expected = vec![0.0; m * n];
            let mut actual = vec![f32::NAN; m * n];
            matmul_f32(&a, &b, &mut expected, m, k, n).unwrap();
            matmul_dispatch_f32(&a, &b, &mut actual, m, k, n).unwrap();

            for (&expected, &actual) in expected.iter().zip(&actual) {
                let tolerance = 1.0e-4_f32.max(expected.abs() * 1.0e-4);
                assert!((expected - actual).abs() <= tolerance);
            }
        }
    }
}
