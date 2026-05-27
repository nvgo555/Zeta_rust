use ndarray::Array3;
use crate::constants::ETA_IPOW;
use crate::ring::ORD;
use crate::ring::Z13Eta;

pub struct HilbertEta;

impl HilbertEta {
    pub fn inner(u: &[Z13Eta], v: &[Z13Eta]) -> Z13Eta {
        let mut sum = Z13Eta::ZERO;
        let d = u.len();
        for i in 0..d {
            let w = ETA_IPOW[i % ORD as usize];
            sum = sum + u[i] * v[i] * w;
        }
        sum
    }

    pub fn norm_sq(u: &[Z13Eta]) -> Z13Eta {
        Self::inner(u, u)
    }

    pub fn layer_norm(x: &mut Array3<Z13Eta>) {
        let shape = x.shape();
        let (b, l, d) = (shape[0], shape[1], shape[2]);
        for i in 0..b {
            for j in 0..l {
                let mut row = Vec::with_capacity(d);
                for k in 0..d {
                    row.push(x[[i, j, k]]);
                }
                let ns = Self::norm_sq(&row);
                let inv_ns = ns.inv();
                for k in 0..d {
                    x[[i, j, k]] = x[[i, j, k]] * inv_ns;
                }
            }
        }
    }
}
