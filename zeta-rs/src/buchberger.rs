use ndarray::{Array2, Array3};
use crate::constants::ETA_POW;
use crate::ring::{P, Z13Eta, ORD};

pub const V: usize = 256;

pub struct BuchbergerEngine;

impl BuchbergerEngine {
    pub fn error_ideal(pred: &Array2<usize>, target: &Array2<usize>) -> Array3<Z13Eta> {
        let shape = pred.shape();
        let (b, l) = (shape[0], shape[1]);
        let mut e = Array3::from_elem((b, l, 3), Z13Eta::ZERO);
        
        for i in 0..b {
            for j in 0..l {
                if pred[[i, j]] != target[[i, j]] {
                    let mut diff = (target[[i, j]] as i32 - pred[[i, j]] as i32) % P as i32;
                    if diff < 0 { diff += P as i32; }
                    e[[i, j, 0]] = Z13Eta([diff as u8, 0, 0]);
                }
            }
        }
        e
    }

    pub fn nullstellensatz_correction(
        x: &Array3<Z13Eta>, // (B, L, D)
        pred: &Array2<usize>, // (B, L)
        target: &Array2<usize>, // (B, L)
        step: usize,
        scale: Option<Z13Eta>,
    ) -> Array2<Z13Eta> { // (D, V)
        let shape = x.shape();
        let (b, l, d) = (shape[0], shape[1], shape[2]);
        
        let e = Self::error_ideal(pred, target);
        let orbit = ETA_POW[step % ORD as usize];
        
        let mut delta = Array2::from_elem((d, V), Z13Eta::ZERO);
        
        for i in 0..b {
            for j in 0..l {
                let e_bl = e[[i, j, 0]]; // Array3 shape in python is B, L, 3, but our Array3 is Z13Eta directly
                if e_bl == Z13Eta::ZERO { continue; }
                
                let t_id = target[[i, j]];
                let p_id = pred[[i, j]];
                
                for k in 0..d {
                    let g = x[[i, j, k]] * orbit;
                    let contrib = g * e_bl;
                    
                    delta[[k, t_id]] = delta[[k, t_id]] + contrib;
                    delta[[k, p_id]] = delta[[k, p_id]] - contrib;
                }
            }
        }
        
        if let Some(s) = scale {
            for k in 0..d {
                for v in 0..V {
                    delta[[k, v]] = delta[[k, v]] * s;
                }
            }
        }
        
        let phase = ETA_POW[(step * 7) % ORD as usize];
        for k in 0..d {
            for v in 0..V {
                delta[[k, v]] = delta[[k, v]] * phase;
            }
        }
        
        delta
    }

    pub fn groebner_reduce(errors: &Array3<Z13Eta>) -> Z13Eta {
        let shape = errors.shape();
        let (b, l, _) = (shape[0], shape[1], shape[2]);
        for i in 0..b {
            for j in 0..l {
                if errors[[i, j, 0]] != Z13Eta::ZERO {
                    return Z13Eta::ONE;
                }
            }
        }
        Z13Eta::ZERO
    }
}
