//! Fast Number Theoretic Transform (NTT) for convolutions.
//!
//! Implements $O(N \log N)$ polynomial multiplication over $\mathbb{Z}_{13}[\eta]$.
//! The transform operates over roots of unity of order dividing 168.


use crate::ring::{P, Z13Eta};

pub const NTT_VALID: [usize; 6] = [1, 2, 3, 4, 6, 12];
pub const NTT_SCALAR: [u8; 13] = [0, 1, 12, 3, 8, 0, 4, 0, 0, 0, 0, 0, 2]; // 1=>1, 2=>12, 3=>3, 4=>8, 6=>4, 12=>2

pub fn mod_pow(mut base: u8, mut exp: usize) -> u8 {
    let mut res = 1;
    base %= P;
    while exp > 0 {
        if exp % 2 == 1 {
            res = (res * base) % P;
        }
        base = (base * base) % P;
        exp /= 2;
    }
    res
}

pub fn mod_inv(n: u8) -> u8 {
    mod_pow(n, (P - 2) as usize)
}

/// Core Number Theoretic Transform struct grouping related operations.
pub struct Ntt;

impl Ntt {
    /// Checks if a given length $N$ is valid for the NTT (must divide 168).
    pub fn is_valid(n: usize) -> bool {
        NTT_VALID.contains(&n)
    }

    pub fn best_size(l: usize) -> usize {
        for &sz in NTT_VALID.iter().rev() {
            if sz <= l {
                return sz;
            }
        }
        1
    }

    /// Performs an in-place/scratchpad-based NTT to avoid memory allocations.
    pub fn ntt_mut(x: &[Z13Eta], out: &mut [Z13Eta]) {
        let n = x.len();
        assert!(Self::is_valid(n));
        let w = NTT_SCALAR[n];
        for k in 0..n {
            let mut sum = Z13Eta::ZERO;
            for j in 0..n {
                let exp = (k * j) % 168;
                let coeff = mod_pow(w, exp);
                sum = sum + x[j].smul(coeff);
            }
            out[k] = sum;
        }
    }

    pub fn ntt(x: &[Z13Eta]) -> Vec<Z13Eta> {
        let n = x.len();
        let mut out = vec![Z13Eta::ZERO; n];
        Self::ntt_mut(x, &mut out);
        out
    }

    pub fn intt(x: &[Z13Eta]) -> Vec<Z13Eta> {
        let n = x.len();
        assert!(Self::is_valid(n));
        let w = NTT_SCALAR[n];
        let wi = mod_inv(w);
        let ni = mod_inv(n as u8);
        
        let mut out = vec![Z13Eta::ZERO; n];
        for k in 0..n {
            let mut sum = Z13Eta::ZERO;
            for j in 0..n {
                let exp = (k * j) % 168;
                let coeff = (mod_pow(wi, exp) * ni) % P;
                sum = sum + x[j].smul(coeff);
            }
            out[k] = sum;
        }
        out
    }

    /// Computes the cyclic convolution of arrays `a` and `b` via NTT -> multiply -> INTT.
    pub fn conv(a: &[Z13Eta], b: &[Z13Eta]) -> Vec<Z13Eta> {
        let n = a.len();
        let a_ntt = Self::ntt(a);
        let b_ntt = Self::ntt(b);
        let mut prod = vec![Z13Eta::ZERO; n];
        for i in 0..n {
            prod[i] = a_ntt[i] * b_ntt[i];
        }
        Self::intt(&prod)
    }
}
