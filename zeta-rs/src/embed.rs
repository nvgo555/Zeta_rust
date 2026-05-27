use ndarray::{Array2, Array3};
use crate::constants::{ETA_IPOW, ETA_POW};
use crate::ring::{Z13Eta, ORD};

pub struct ZRingEmbed {
    w: Array2<Z13Eta>,
}

impl ZRingEmbed {
    pub fn new(v: usize, d: usize) -> Self {
        let mut w = Array2::from_elem((v, d), Z13Eta::ZERO);
        for i in 0..v {
            for j in 0..d {
                w[[i, j]] = ETA_POW[(i * d + j) % ORD as usize];
            }
        }
        Self { w }
    }

    pub fn forward(&self, t: &Array2<usize>) -> Array3<Z13Eta> {
        let shape = t.shape();
        let (b, l) = (shape[0], shape[1]);
        let d = self.w.shape()[1];
        let v = self.w.shape()[0];
        let mut out = Array3::from_elem((b, l, d), Z13Eta::ZERO);
        for i in 0..b {
            for j in 0..l {
                let token = t[[i, j]] % v;
                for k in 0..d {
                    out[[i, j, k]] = self.w[[token, k]];
                }
            }
        }
        out
    }
}

pub struct TatePE {
    d: usize,
}

impl TatePE {
    pub fn new(d: usize) -> Self {
        assert!(d % 2 == 0);
        Self { d }
    }

    pub fn forward(&self, x: &mut Array3<Z13Eta>) {
        let shape = x.shape();
        let (b, l, d) = (shape[0], shape[1], shape[2]);
        assert_eq!(d, self.d);
        
        for n in 0..l {
            for k in 0..(d / 2) {
                let nk = (n * k) % ORD as usize;
                let pe_pos = ETA_POW[nk];
                let pe_neg = ETA_IPOW[nk];
                for i in 0..b {
                    x[[i, n, 2 * k]] = x[[i, n, 2 * k]] + pe_pos;
                    x[[i, n, 2 * k + 1]] = x[[i, n, 2 * k + 1]] + pe_neg;
                }
            }
        }
    }
}
