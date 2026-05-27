//! Zeta Neural Architecture.
//!
//! Connects the embeddings, attention layers, and Buchberger learning engine into 
//! a cohesive state machine operating entirely within $\mathbb{Z}_{13}[\eta]$.

use ndarray::{Array2, Array3, Zip};
use ndarray::parallel::prelude::*;
use std::cmp::min;
use crate::constants::{ETA_POW};
use crate::ring::{Z13Eta, ORD};
use crate::embed::{ZRingEmbed, TatePE};
use crate::attention::SpectralAttention;
use crate::gqm::ErrorCache;
use crate::spectral::t3n;
use crate::witt::WittVector;
use crate::buchberger::BuchbergerEngine;

/// The main Zeta AI Model.
pub struct ZetaModel {
    /// Vocabulary size.
    pub v: usize,
    /// Embedding dimension.
    pub d: usize,
    /// Number of attention layers.
    pub n: usize,
    /// Current execution step.
    pub step: usize,
    /// Current $p$-adic precision level (up to 4).
    pub prec: usize,
    embed: ZRingEmbed,
    pe: TatePE,
    layers: Vec<SpectralAttention>,
    /// The classification head weights in $\mathbb{Z}_{13}[\eta]$.
    pub head: Array2<Z13Eta>,
    /// The multi-precision Witt vector state of the classification head.
    pub witt_head: Array3<Z13Eta>, // (D, V, PREC)
    pub error_cache: Option<ErrorCache>,
    pub last_x: Option<Array3<Z13Eta>>,
    pub s3_orbit: i32,
}

impl ZetaModel {
    /// Constructs a new ZetaModel with the given vocabulary, dimension, and depth.
    pub fn new(v: usize, d: usize, n: usize) -> Self {
        let embed = ZRingEmbed::new(v, d);
        let pe = TatePE::new(d);
        let mut layers = Vec::new();
        for _ in 0..n {
            layers.push(SpectralAttention::new(d));
        }
        
        let mut head = Array2::from_elem((d, v), Z13Eta::ZERO);
        for i in 0..d {
            for j in 0..v {
                head[[i, j]] = ETA_POW[(i * v + j) % ORD as usize];
            }
        }
        
        let mut witt_head = Array3::from_elem((d, v, 4), Z13Eta::ZERO);
        for i in 0..d {
            for j in 0..v {
                let w = WittVector::from_ring(head[[i, j]], 4);
                for p in 0..4 {
                    witt_head[[i, j, p]] = w.data[p];
                }
            }
        }
        
        Self {
            v, d, n, step: 0, prec: 1,
            embed, pe, layers, head, witt_head,
            error_cache: None, last_x: None, s3_orbit: -1,
        }
    }

    pub fn sync_head(&mut self) {
        for i in 0..self.d {
            for j in 0..self.v {
                let mut combo = Z13Eta::ZERO;
                for p in 0..self.prec {
                    combo = combo + self.witt_head[[i, j, p]].smul((p + 1) as u8);
                }
                self.head[[i, j]] = combo;
            }
        }
    }

    pub fn lift(&mut self) {
        if self.prec < 4 {
            self.prec += 1;
            self.sync_head();
        }
    }

    /// Executes the forward pass of the model, transforming token IDs into $\mathbb{Z}_{13}$ logits.
    ///
    /// The forward pass maps the tensor via $T_3^n$, applies spectral attention, and computes traces.
    pub fn forward(&mut self, tokens: &Array2<usize>, is_training: bool) -> Array3<u8> {
        let shape = tokens.shape();
        let (b, l) = (shape[0], shape[1]);
        
        let mut x = self.embed.forward(tokens);
        self.pe.forward(&mut x);
        
        for i in 0..self.n {
            let n_pow = (i + 1) % ORD as usize;
            x.par_map_inplace(|v| *v = t3n(v, n_pow));
            self.layers[i].step = self.step;
            x = self.layers[i].forward(&x, is_training);
        }
        
        self.sync_head();
        self.last_x = Some(x.clone());
        
        let mut sc = Array3::from_elem((b, l, self.v), 0u8);
        for i in 0..b {
            for j in 0..l {
                for v_idx in 0..self.v {
                    let mut sum = 0u16;
                    for k in 0..self.d {
                        sum += (x[[i, j, k]] * self.head[[k, v_idx]]).trace() as u16;
                    }
                    sc[[i, j, v_idx]] = (sum % crate::ring::P as u16) as u8;
                }
            }
        }
        
        self.step += 1;
        sc
    }

    /// Performs a single training step using the Buchberger-Nullstellensatz algorithm.
    /// Returns a tuple of (hallucinations, delta_norm).
    pub fn train_step(&mut self, tokens: &Array2<usize>, targets: &Array2<usize>) -> (usize, usize) {
        let shape = tokens.shape();
        let (b, l) = (shape[0], shape[1]);
        
        let sc = self.forward(tokens, true);
        
        let mut preds = Array2::from_elem((b, l), 0usize);
        let mut hall = 0;
        
        for i in 0..b {
            for j in 0..l {
                let mut max_val = 0;
                let mut max_idx = 0;
                for v_idx in 0..self.v {
                    if sc[[i, j, v_idx]] > max_val {
                        max_val = sc[[i, j, v_idx]];
                        max_idx = v_idx;
                    }
                }
                preds[[i, j]] = max_idx;
                if max_idx != targets[[i, j]] {
                    hall += 1;
                }
            }
        }
        
        let mut delta_norm = 0;
        if hall > 0 {
            if let Some(last_x) = &self.last_x {
                let mut delta = BuchbergerEngine::nullstellensatz_correction(last_x, &preds, targets, self.step, None);
                for i in 0..self.d {
                    for j in 0..self.v {
                        delta_norm += delta[[i, j]].norm() as usize;
                    }
                }
                if delta_norm > 6 {
                    for i in 0..self.d {
                        for j in 0..self.v {
                            delta[[i, j]] = delta[[i, j]].smul(7);
                        }
                    }
                }
                
                for i in 0..self.d {
                    for j in 0..self.v {
                        let dw = WittVector::from_ring(delta[[i, j]], 4);
                        let mut aw_data = [Z13Eta::ZERO; 4];
                        for p in 0..self.prec {
                            aw_data[p] = self.witt_head[[i, j, p]];
                        }
                        let aw = WittVector { prec: self.prec, data: aw_data };
                        let sum_w = WittVector::wadd(&aw, &dw);
                        for p in 0..self.prec {
                            self.witt_head[[i, j, p]] = sum_w.data[p];
                        }
                    }
                }
                self.sync_head();
                
                if self.error_cache.is_none() {
                    self.error_cache = Some(ErrorCache::new(min(self.d * self.v, 64)));
                }
                
                if let Some(ec) = &mut self.error_cache {
                    ec.absorb(&delta);
                    let mut spec_corr = ec.spectral_correction(self.d, self.v);
                    for i in 0..self.d {
                        for j in 0..self.v {
                            spec_corr[[i, j]] = spec_corr[[i, j]].smul(2);
                        }
                    }
                    
                    for i in 0..self.d {
                        for j in 0..self.v {
                            let sw = WittVector::from_ring(spec_corr[[i, j]], 4);
                            let mut aw_data = [Z13Eta::ZERO; 4];
                            for p in 0..self.prec {
                                aw_data[p] = self.witt_head[[i, j, p]];
                            }
                            let aw = WittVector { prec: self.prec, data: aw_data };
                            let sum_w = WittVector::wadd(&aw, &sw);
                            for p in 0..self.prec {
                                self.witt_head[[i, j, p]] = sum_w.data[p];
                            }
                        }
                    }
                    self.sync_head();
                }
            }
        }
        
        (hall, delta_norm)
    }
}
