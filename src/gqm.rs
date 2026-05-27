use ndarray::Array2;
use std::cmp::min;
use crate::constants::{ETA_POW, T3_POW};
use crate::ring::{Z13Eta, ORD};
use crate::ntt::Ntt;

pub struct ErrorCache {
    pub k: usize,
    pub rho: Array2<Z13Eta>,
    pub step: usize,
}

impl ErrorCache {
    pub fn new(k: usize) -> Self {
        Self {
            k,
            rho: Array2::from_elem((k, k), Z13Eta::ZERO),
            step: 0,
        }
    }
    
    pub fn absorb(&mut self, corr: &Array2<Z13Eta>) {
        let d = corr.shape()[0];
        let v = corr.shape()[1];
        let flat_len = d * v;
        let n = min(flat_len, self.k);
        let mut states = vec![Z13Eta::ZERO; self.k];
        
        let mut idx = 0;
        for i in 0..d {
            for j in 0..v {
                if idx < n {
                    states[idx] = corr[[i, j]];
                    idx += 1;
                }
            }
        }
        
        let mut new_rho = Array2::from_elem((self.k, self.k), Z13Eta::ZERO);
        for i in 0..self.k {
            for j in 0..self.k {
                new_rho[[i, j]] = states[i] * states[j];
            }
        }
        
        let mut has_old = false;
        for i in 0..self.k {
            for j in 0..self.k {
                if self.rho[[i, j]] != Z13Eta::ZERO {
                    has_old = true;
                    break;
                }
            }
        }
        
        if has_old {
            let m = T3_POW[1];
            let mi = T3_POW[ORD as usize - 1];
            let mut evolved = Array2::from_elem((self.k, self.k), Z13Eta::ZERO);
            for i in 0..self.k {
                for j in 0..self.k {
                    let old_val = self.rho[[i, j]];
                    let mut left = [0u8; 3];
                    for r in 0..3 {
                        let mut sum = 0;
                        for c in 0..3 {
                            sum += m[r][c] as u16 * old_val.0[c] as u16;
                        }
                        left[r] = (sum % crate::ring::P as u16) as u8;
                    }
                    let mut right = [0u8; 3];
                    for c in 0..3 {
                        let mut sum = 0;
                        for r in 0..3 {
                            sum += left[r] as u16 * mi[r][c] as u16;
                        }
                        right[c] = (sum % crate::ring::P as u16) as u8;
                    }
                    evolved[[i, j]] = Z13Eta(right);
                }
            }
            for i in 0..self.k {
                for j in 0..self.k {
                    self.rho[[i, j]] = evolved[[i, j]] + new_rho[[i, j]];
                }
            }
        } else {
            self.rho = new_rho;
        }
        self.step += 1;
    }

    pub fn spectral_correction(&self, d: usize, v: usize) -> Array2<Z13Eta> {
        let mut all_zero = true;
        for i in 0..self.k {
            for j in 0..self.k {
                if self.rho[[i, j]] != Z13Eta::ZERO {
                    all_zero = false;
                    break;
                }
            }
        }
        if all_zero {
            return Array2::from_elem((d, v), Z13Eta::ZERO);
        }
        
        let mut diag = vec![Z13Eta::ZERO; self.k];
        for i in 0..self.k {
            diag[i] = self.rho[[i, i]];
        }
        
        let n = Ntt::best_size(self.k);
        let spec = Ntt::ntt(&diag[..n]);
        
        let total = d * v;
        let mut full = vec![Z13Eta::ZERO; total];
        for i in 0..total {
            full[i] = spec[i % n];
        }
        
        let mut corr = Array2::from_elem((d, v), Z13Eta::ZERO);
        let mut idx = 0;
        let phase = ETA_POW[self.step % ORD as usize];
        
        for i in 0..d {
            for j in 0..v {
                corr[[i, j]] = full[idx] * phase;
                idx += 1;
            }
        }
        corr
    }
}
