use ndarray::{Array3, s};
use crate::constants::{val_13, ETA_IPOW};
use crate::ring::{Z13Eta, ORD};
use crate::spectral::{t3n, SylvesterProjectors};

pub struct PAdicKernel;

impl PAdicKernel {
    pub fn n_levels(l: usize) -> usize {
        let mut h = 1;
        let mut n = 0;
        while h < l {
            h *= 13;
            n += 1;
        }
        n
    }

    pub fn build_tree(x: &Array3<Z13Eta>) -> Vec<Array3<Z13Eta>> {
        let shape = x.shape();
        let (b, l, d) = (shape[0], shape[1], shape[2]);
        let h_max = Self::n_levels(l);
        
        let mut levels = vec![x.clone()];
        let mut current = x.clone();
        
        for h in 0..h_max {
            let lc = current.shape()[1];
            if lc < 13 { break; }
            let lpad = ((lc + 12) / 13) * 13;
            
            let mut padded = Array3::from_elem((b, lpad, d), Z13Eta::ZERO);
            padded.slice_mut(s![.., ..lc, ..]).assign(&current);
            
            let nb = lpad / 13;
            let mut grouped = Array3::from_elem((b, nb, d), Z13Eta::ZERO);
            
            for i in 0..b {
                for j in 0..nb {
                    for k in 0..d {
                        let mut sum = Z13Eta::ZERO;
                        for m in 0..13 {
                            sum = sum + padded[[i, j * 13 + m, k]];
                        }
                        grouped[[i, j, k]] = sum;
                    }
                }
            }
            
            let mut next_level = Array3::from_elem((b, nb, d), Z13Eta::ZERO);
            let mut n_step = 1;
            for _ in 0..=h { n_step = (n_step * 13) % ORD as usize; }
            
            for i in 0..b {
                for j in 0..nb {
                    for k in 0..d {
                        let evolved = t3n(&grouped[[i, j, k]], n_step);
                        next_level[[i, j, k]] = SylvesterProjectors::p1(&evolved);
                    }
                }
            }
            levels.push(next_level.clone());
            current = next_level;
        }
        levels
    }

    pub fn tree_attend(levels: &[Array3<Z13Eta>], x: &Array3<Z13Eta>) -> Array3<Z13Eta> {
        if levels.len() == 1 {
            return x.clone();
        }
        let shape = x.shape();
        let (b, l, d) = (shape[0], shape[1], shape[2]);
        let mut out = x.clone();
        
        for (h, sh) in levels.iter().enumerate().skip(1) {
            let mut factor = 1;
            for _ in 0..h { factor *= 13; }
            
            let nb = sh.shape()[1];
            let mut n_inv = 1;
            for _ in 0..h { n_inv = (n_inv * 13) % ORD as usize; }
            n_inv = (ORD as usize - n_inv) % ORD as usize;
            
            let w = ETA_IPOW[h % ORD as usize];
            
            for i in 0..b {
                for j in 0..l {
                    let coarse_j = j / factor;
                    if coarse_j < nb {
                        let val = sh[[i, coarse_j, d - 1]]; // wait, we map for each d
                        for k in 0..d {
                            let up = t3n(&sh[[i, coarse_j, k]], n_inv);
                            let up_w = up * w;
                            let up_sub = SylvesterProjectors::p23(&up_w);
                            out[[i, j, k]] = out[[i, j, k]] + up_sub;
                        }
                    }
                }
            }
        }
        out
    }

    pub fn apply(x: &Array3<Z13Eta>) -> Array3<Z13Eta> {
        let levels = Self::build_tree(x);
        Self::tree_attend(&levels, x)
    }

    pub fn strong_triangle_inequality(i: usize, j: usize, k: usize) -> bool {
        let vi = val_13(i.abs_diff(j));
        let vj = val_13(j.abs_diff(k));
        let vk = val_13(i.abs_diff(k));
        vk >= std::cmp::min(vi, vj)
    }
}
