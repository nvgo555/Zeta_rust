use crate::constants::{P1_MAT, P23_MAT, S3_MATS, S3_MUL, T3_POW};
use crate::ring::{P, Z13Eta, ORD};

pub fn proj(x: &Z13Eta, m: &[[u8; 3]; 3]) -> Z13Eta {
    let x0 = x.0[0]; let x1 = x.0[1]; let x2 = x.0[2];
    Z13Eta([
        (x0 * m[0][0] + x1 * m[0][1] + x2 * m[0][2]) % P,
        (x0 * m[1][0] + x1 * m[1][1] + x2 * m[1][2]) % P,
        (x0 * m[2][0] + x1 * m[2][1] + x2 * m[2][2]) % P,
    ])
}

pub fn t3n(x: &Z13Eta, n: usize) -> Z13Eta {
    proj(x, &T3_POW[n % ORD as usize])
}

pub fn t3n_s3(x: &Z13Eta, n: usize, g: usize) -> Z13Eta {
    S3Galois::apply(&t3n(x, n), g)
}

pub struct SylvesterProjectors;

impl SylvesterProjectors {
    pub fn p1(x: &Z13Eta) -> Z13Eta {
        proj(x, &P1_MAT)
    }

    pub fn p23(x: &Z13Eta) -> Z13Eta {
        proj(x, &P23_MAT)
    }

    pub fn split(x: &Z13Eta) -> (Z13Eta, Z13Eta) {
        let p1x = proj(x, &P1_MAT);
        let p23x = *x - p1x;
        (p1x, p23x)
    }
}

pub struct SpectralDecomposition;

impl SpectralDecomposition {
    pub fn evolve(x: &Z13Eta, n: usize) -> Z13Eta {
        let p1x = SylvesterProjectors::p1(x);
        let mut p1_mult = 1;
        for _ in 0..n { p1_mult = (p1_mult * 7) % P; }
        
        let term1 = p1x.smul(p1_mult);
        let p23x = *x - p1x;
        let term2 = t3n(&p23x, n);
        
        term1 + term2
    }
}

pub struct S3Galois;

impl S3Galois {
    pub fn apply(x: &Z13Eta, g: usize) -> Z13Eta {
        proj(x, &S3_MATS[g % 6])
    }

    pub fn compose_indices(g: usize, h: usize) -> usize {
        S3_MUL[g % 6][h % 6]
    }

    pub fn orbit(x: &Z13Eta) -> [Z13Eta; 6] {
        [
            proj(x, &S3_MATS[0]),
            proj(x, &S3_MATS[1]),
            proj(x, &S3_MATS[2]),
            proj(x, &S3_MATS[3]),
            proj(x, &S3_MATS[4]),
            proj(x, &S3_MATS[5]),
        ]
    }

    pub fn casimir(x: &Z13Eta) -> Z13Eta {
        let o = Self::orbit(x);
        let mut sum = Z13Eta::ZERO;
        for item in &o {
            sum = sum + *item;
        }
        sum
    }
}
