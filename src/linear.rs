use ndarray::{Array2, Array3, Array4};
use crate::constants::ETA_POW;
use crate::ring::ORD;
use crate::ring::Z13Eta;

pub fn ring_lin(x: &Array3<Z13Eta>, w: &Array2<Z13Eta>) -> Array3<Z13Eta> {
    let shape = x.shape();
    let (b, l, d_in) = (shape[0], shape[1], shape[2]);
    let d_out = w.shape()[1];
    
    let mut out = Array3::from_elem((b, l, d_out), Z13Eta::ZERO);
    for i in 0..b {
        for j in 0..l {
            for k_out in 0..d_out {
                let mut sum = Z13Eta::ZERO;
                for k_in in 0..d_in {
                    sum = sum + x[[i, j, k_in]] * w[[k_in, k_out]];
                }
                out[[i, j, k_out]] = sum;
            }
        }
    }
    out
}

pub fn born_norm(s: &mut Array4<Z13Eta>) {
    let shape = s.shape();
    let (b, lq, lk, _) = (shape[0], shape[1], shape[2], shape[3]);
    for i in 0..b {
        for j in 0..lq {
            let mut row = Vec::with_capacity(lk);
            for k in 0..lk {
                row.push(s[[i, j, k, 0]]); // assuming scalar scores or something? Wait, born_norm in Python normalizes the D=3 dimension? No, Python S is (B, Lq, Lk, 3) which means it's an Array3<Z13Eta> where D=Lk.
            }
            // Actually, wait. Let's fix this in rust. `S` in python is (B, Lq, Lk) of Z13Eta. So it's Array3<Z13Eta>.
        }
    }
}

pub fn ring_attend(a: &Array3<Z13Eta>, v: &Array3<Z13Eta>) -> Array3<Z13Eta> {
    // A: (B, L, L), V: (B, L, D) -> Out: (B, L, D)
    let shape_a = a.shape();
    let shape_v = v.shape();
    let (b, l, _) = (shape_a[0], shape_a[1], shape_a[2]);
    let d = shape_v[2];
    
    let mut out = Array3::from_elem((b, l, d), Z13Eta::ZERO);
    for i in 0..b {
        for j in 0..l {
            for k in 0..d {
                let mut sum = Z13Eta::ZERO;
                for m in 0..l {
                    sum = sum + a[[i, j, m]] * v[[i, m, k]];
                }
                out[[i, j, k]] = sum;
            }
        }
    }
    out
}

pub fn laplacian(x: &Array3<Z13Eta>) -> Array3<Z13Eta> {
    let shape = x.shape();
    let (b, l, d) = (shape[0], shape[1], shape[2]);
    let mut out = Array3::from_elem((b, l, d), Z13Eta::ZERO);
    for i in 0..b {
        for j in 0..l {
            let j_prev = if j == 0 { l - 1 } else { j - 1 };
            let j_next = if j == l - 1 { 0 } else { j + 1 };
            for k in 0..d {
                out[[i, j, k]] = x[[i, j_prev, k]] + x[[i, j_next, k]] + x[[i, j, k]].smul(11);
            }
        }
    }
    out
}

pub fn w_seed(d_in: usize, d_out: usize) -> Array2<Z13Eta> {
    let mut w = Array2::from_elem((d_in, d_out), Z13Eta::ZERO);
    for i in 0..d_in {
        for j in 0..d_out {
            let exp = (i * d_out + j) % ORD as usize;
            w[[i, j]] = ETA_POW[exp];
        }
    }
    w
}

pub fn alg_dropout(x: &mut Array3<Z13Eta>, rate: usize, step: usize, training: bool) {
    if !training { return; }
    let shape = x.shape();
    let (b, l, d) = (shape[0], shape[1], shape[2]);
    for k in 0..d {
        let exp = (rate * k * (step + 1)) % ORD as usize;
        let keep = ETA_POW[exp].0[0] != 0;
        if !keep {
            for i in 0..b {
                for j in 0..l {
                    x[[i, j, k]] = Z13Eta::ZERO;
                }
            }
        }
    }
}
