use ndarray::{Array3, Array2, concatenate, Axis};
use crate::ring::Z13Eta;
use crate::spectral::SylvesterProjectors;
use crate::kernel::PAdicKernel;
use crate::hilbert::HilbertEta;
use crate::linear::{ring_lin, laplacian, w_seed, alg_dropout};

pub struct SpectralAttention {
    pub step: usize,
    wq_dom: Array2<Z13Eta>,
    wk_dom: Array2<Z13Eta>,
    wv_dom: Array2<Z13Eta>,
    wq_sub: Array2<Z13Eta>,
    wk_sub: Array2<Z13Eta>,
    wv_sub: Array2<Z13Eta>,
    wo: Array2<Z13Eta>,
}

impl SpectralAttention {
    pub fn new(d: usize) -> Self {
        assert!(d % 6 == 0);
        let hd = d / 3;
        Self {
            step: 0,
            wq_dom: w_seed(d, hd),
            wk_dom: w_seed(d, hd),
            wv_dom: w_seed(d, hd),
            wq_sub: w_seed(d, hd),
            wk_sub: w_seed(d, hd),
            wv_sub: w_seed(d, hd),
            wo: w_seed(2 * hd, d),
        }
    }

    pub fn forward(&self, x: &Array3<Z13Eta>, is_training: bool) -> Array3<Z13Eta> {
        let shape = x.shape();
        let (b, l, d) = (shape[0], shape[1], shape[2]);
        let hd = d / 3;

        let mut x_p1 = Array3::from_elem((b, l, d), Z13Eta::ZERO);
        let mut x_p23 = Array3::from_elem((b, l, d), Z13Eta::ZERO);
        
        for i in 0..b {
            for j in 0..l {
                for k in 0..d {
                    let (p1, p23) = SylvesterProjectors::split(&x[[i, j, k]]);
                    x_p1[[i, j, k]] = p1;
                    x_p23[[i, j, k]] = p23;
                }
            }
        }
        
        let x_p1_k = PAdicKernel::apply(&x_p1);
        let x_p23_k = PAdicKernel::apply(&x_p23);
        
        let q1 = ring_lin(&x_p1_k, &self.wq_dom);
        let k1 = ring_lin(&x_p1_k, &self.wk_dom);
        let v1 = ring_lin(&x_p1_k, &self.wv_dom);
        
        let q2 = ring_lin(&x_p23_k, &self.wq_sub);
        let k2 = ring_lin(&x_p23_k, &self.wk_sub);
        let v2 = ring_lin(&x_p23_k, &self.wv_sub);
        
        let mut o1 = Array3::from_elem((b, l, hd), Z13Eta::ZERO);
        let mut o2 = Array3::from_elem((b, l, hd), Z13Eta::ZERO);
        
        for i in 0..b {
            for j in 0..l {
                for k in 0..hd {
                    let s1 = (q1[[i, j, k]] * k1[[i, j, k]]).trace();
                    let s2 = (q2[[i, j, k]] * k2[[i, j, k]]).trace();
                    
                    o1[[i, j, k]] = v1[[i, j, k]].smul(s1);
                    o2[[i, j, k]] = v2[[i, j, k]].smul(s2);
                }
            }
        }
        
        let combined = concatenate(Axis(2), &[o1.view(), o2.view()]).unwrap();
        let out_proj = ring_lin(&combined, &self.wo);
        
        let mut out = Array3::from_elem((b, l, d), Z13Eta::ZERO);
        for i in 0..b {
            for j in 0..l {
                for k in 0..d {
                    out[[i, j, k]] = x[[i, j, k]] + out_proj[[i, j, k]];
                }
            }
        }
        
        let mut out = laplacian(&out);
        HilbertEta::layer_norm(&mut out);
        
        alg_dropout(&mut out, 3, self.step, is_training); // DROP = 3
        
        out
    }
}
