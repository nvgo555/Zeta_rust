use ndarray::{Array2, Array3};
use std::cmp::min;
use crate::constants::{ETA_POW, ORD};
use crate::ring::Z13Eta;
use crate::embed::{ZRingEmbed, TatePE};
use crate::attention::SpectralAttention;
use crate::gqm::ErrorCache;
use crate::spectral::t3n;
use crate::witt::WittVector;
use crate::buchberger::BuchbergerEngine;

pub struct ZetaModel {
    pub v: usize,
    pub d: usize,
    pub n: usize,
    pub step: usize,
    pub prec: usize,
    embed: ZRingEmbed,
    pe: TatePE,
    layers: Vec<SpectralAttention>,
    pub head: Array2<Z13Eta>,
    pub witt_head: Array3<Z13Eta>, // (D, V, PREC)
    pub error_cache: Option<ErrorCache>,
    pub last_x: Option<Array3<Z13Eta>>,
    pub s3_orbit: i32,
}

impl ZetaModel {
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
        
        let witt_head = WittVector::from_ring(&head, 4); // max prec = 4
        
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

    pub fn forward(&mut self, tokens: &Array2<usize>, is_training: bool) -> Array3<u8> {
        let shape = tokens.shape();
        let (b, l) = (shape[0], shape[1]);
        
        let mut x = self.embed.forward(tokens);
        self.pe.forward(&mut x);
        
        for i in 0..self.n {
            x = t3n(&x, (i + 1) % ORD as usize);
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
                let mut delta = BuchbergerEngine::nullstellensatz_correction(last_x, &preds, targets, self.step);
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
                
                let delta_witt = WittVector::from_ring(&delta, 4);
                let mut active = Array3::from_elem((self.d, self.v, self.prec), Z13Eta::ZERO);
                for i in 0..self.d {
                    for j in 0..self.v {
                        for p in 0..self.prec {
                            active[[i, j, p]] = self.witt_head[[i, j, p]];
                        }
                    }
                }
                
                let mut delta_active = Array3::from_elem((self.d, self.v, self.prec), Z13Eta::ZERO);
                for i in 0..self.d {
                    for j in 0..self.v {
                        for p in 0..self.prec {
                            delta_active[[i, j, p]] = delta_witt[[i, j, p]];
                        }
                    }
                }
                
                let new_active = WittVector::wadd(&active, &delta_active);
                for i in 0..self.d {
                    for j in 0..self.v {
                        for p in 0..self.prec {
                            self.witt_head[[i, j, p]] = new_active[[i, j, p]];
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
                    
                    let spec_witt = WittVector::from_ring(&spec_corr, 4);
                    let mut active2 = Array3::from_elem((self.d, self.v, self.prec), Z13Eta::ZERO);
                    for i in 0..self.d {
                        for j in 0..self.v {
                            for p in 0..self.prec {
                                active2[[i, j, p]] = self.witt_head[[i, j, p]];
                            }
                        }
                    }
                    
                    let mut spec_active = Array3::from_elem((self.d, self.v, self.prec), Z13Eta::ZERO);
                    for i in 0..self.d {
                        for j in 0..self.v {
                            for p in 0..self.prec {
                                spec_active[[i, j, p]] = spec_witt[[i, j, p]];
                            }
                        }
                    }
                    
                    let new_active2 = WittVector::wadd(&active2, &spec_active);
                    for i in 0..self.d {
                        for j in 0..self.v {
                            for p in 0..self.prec {
                                self.witt_head[[i, j, p]] = new_active2[[i, j, p]];
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
