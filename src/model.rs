//! Zeta Neural Architecture.
//!
//! Connects the embeddings, attention layers, and Buchberger learning engine into 
//! a cohesive state machine operating entirely within $\mathbb{Z}_{13}[\eta]$.

use ndarray::{Array2, Array3, Axis, Zip};
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
        let head = &self.head;
        let v = self.v;
        let d = self.d;

        Zip::from(sc.lanes_mut(Axis(2)))
            .and(x.lanes(Axis(2)))
            .par_for_each(|mut sc_row, x_row| {
                for v_idx in 0..v {
                    let mut c0 = 0u32;
                    let mut c1 = 0u32;
                    let mut c2 = 0u32;
                    for k in 0..d {
                        let a = x_row[k].0;
                        let b = head[[k, v_idx]].0;
                        
                        let a0 = a[0] as u32; let a1 = a[1] as u32; let a2 = a[2] as u32;
                        let b0 = b[0] as u32; let b1 = b[1] as u32; let b2 = b[2] as u32;
                        
                        let a1_b2_a2_b1 = a1 * b2 + a2 * b1;
                        let a2_b2 = a2 * b2;
                        
                        c0 += a0 * b0 + a1_b2_a2_b1 + a2_b2;
                        c1 += a0 * b1 + a1 * b0 + a1_b2_a2_b1 + 2 * a2_b2;
                        c2 += a0 * b2 + a1 * b1 + a2 * b0 + a1_b2_a2_b1 + 2 * a2_b2;
                    }
                    let c0 = (c0 % 13) as u8;
                    let c1 = (c1 % 13) as u8;
                    let c2 = (c2 % 13) as u8;
                    
                    let sum = (3 * c0 as u16 + c1 as u16 + 3 * c2 as u16) % 13;
                    sc_row[v_idx] = sum as u8;
                }
            });
        
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
