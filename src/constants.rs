use lazy_static::lazy_static;
use ndarray::Array3;
use crate::ring::{Z13Eta, F169, P, ORD};

lazy_static! {
    pub static ref ETA_POW: Vec<Z13Eta> = {
        let mut ep = vec![Z13Eta::ONE];
        for _ in 0..ORD {
            let last = *ep.last().unwrap();
            ep.push(last * Z13Eta([0, 1, 0]));
        }
        ep
    };

    pub static ref ETA_IPOW: Vec<Z13Eta> = {
        let mut eip = Vec::with_capacity((ORD + 1) as usize);
        for k in 0..=(ORD as usize) {
            eip.push(ETA_POW[(ORD as usize - k % ORD as usize) % ORD as usize]);
        }
        eip
    };

    pub static ref T3_ROWS: [[u8; 3]; 3] = [[0,0,1], [1,0,1], [0,1,1]];

    pub static ref T3_POW: [[[u8; 3]; 3]; 168] = {
        let mut tp = [[[0; 3]; 3]; 168];
        tp[0] = [[1,0,0], [0,1,0], [0,0,1]];
        for k in 1..168 {
            let mut next = [[0; 3]; 3];
            for i in 0..3 {
                for j in 0..3 {
                    let mut sum = 0;
                    for r in 0..3 {
                        sum += tp[k-1][i][r] * T3_ROWS[r][j];
                    }
                    next[i][j] = sum % P;
                }
            }
            tp[k] = next;
        }
        tp
    };

    pub static ref INV_TBL: Array3<Z13Eta> = {
        let mut invs = Array3::from_elem((13, 13, 13), Z13Eta::ZERO);
        for a0 in 0..13 {
            for a1 in 0..13 {
                for a2 in 0..13 {
                    let a = Z13Eta([a0, a1, a2]);
                    let phi1 = a.phi1();
                    let phi2 = a.phi2();
                    let is_unit = phi1 != 0 && (phi2.0[0] != 0 || phi2.0[1] != 0);
                    if !is_unit {
                        invs[[a0 as usize, a1 as usize, a2 as usize]] = Z13Eta::ZERO;
                    } else {
                        let mut r = Z13Eta::ONE;
                        let mut b = a;
                        let mut n = 167;
                        while n > 0 {
                            if n & 1 != 0 {
                                r = r * b;
                            }
                            b = b * b;
                            n >>= 1;
                        }
                        invs[[a0 as usize, a1 as usize, a2 as usize]] = r;
                    }
                }
            }
        }
        invs
    };
    
    pub static ref VAL_LUT: Vec<u8> = {
        let val_max = 13_usize.pow(5);
        let mut val = vec![0; val_max];
        for d in 0..val_max {
            let mut tmp = d;
            let mut v = 0;
            while tmp > 0 && tmp % 13 == 0 {
                v += 1;
                tmp /= 13;
            }
            val[d] = v;
        }
        val
    };

    pub static ref S3_MATS: [[[u8; 3]; 3]; 6] = [
        [[1,0,0],[0,1,0],[0,0,1]],  // e
        [[1,0,0],[0,0,1],[0,1,0]],  // (12)
        [[0,1,0],[1,0,0],[0,0,1]],  // (01)
        [[0,0,1],[0,1,0],[1,0,0]],  // (02)
        [[0,1,0],[0,0,1],[1,0,0]],  // (012)
        [[0,0,1],[1,0,0],[0,1,0]],  // (021)
    ];

    pub static ref S3_MUL: [[usize; 6]; 6] = {
        let mut mul = [[0; 6]; 6];
        for g in 0..6 {
            for h in 0..6 {
                let mut prod = [[0; 3]; 3];
                for i in 0..3 {
                    for j in 0..3 {
                        let mut sum = 0;
                        for r in 0..3 {
                            sum += S3_MATS[g][i][r] * S3_MATS[h][r][j];
                        }
                        prod[i][j] = sum % P;
                    }
                }
                for k in 0..6 {
                    if S3_MATS[k] == prod {
                        mul[g][h] = k;
                        break;
                    }
                }
            }
        }
        mul
    };
}

pub const P1_MAT: [[u8; 3]; 3] = [[1,7,10],[3,8,4],[7,10,5]];
pub const P23_MAT: [[u8; 3]; 3] = [[0,6,3],[10,6,9],[6,3,9]];

pub fn val_13(d: usize) -> u8 {
    if d < 13_usize.pow(5) {
        VAL_LUT[d]
    } else {
        let mut tmp = d;
        let mut v = 0;
        while tmp > 0 && tmp % 13 == 0 {
            v += 1;
            tmp /= 13;
        }
        v
    }
}
