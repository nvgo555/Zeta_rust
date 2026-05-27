use crate::ring::{Z13Eta, P};
use std::cmp::min;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WittVector {
    pub prec: usize,
    pub data: Vec<Z13Eta>,
}

impl WittVector {
    pub fn from_ring(a: Z13Eta, prec: usize) -> Self {
        let mut data = vec![Z13Eta::ZERO; prec];
        if prec > 0 {
            data[0] = a;
        }
        WittVector { prec, data }
    }

    pub fn to_ring(&self) -> Z13Eta {
        if self.prec > 0 {
            self.data[0]
        } else {
            Z13Eta::ZERO
        }
    }

    pub fn wadd(x: &WittVector, y: &WittVector) -> WittVector {
        let prec = min(x.prec, y.prec);
        let mut out = vec![Z13Eta::ZERO; prec];
        let mut carry = [0u8; 3];
        
        for i in 0..prec {
            let mut raw = [0u8; 3];
            let mut next_carry = [0u8; 3];
            for j in 0..3 {
                let sum = x.data[i].0[j] as u16 + y.data[i].0[j] as u16 + carry[j] as u16;
                raw[j] = (sum % P as u16) as u8;
                next_carry[j] = (sum / P as u16) as u8;
            }
            out[i] = Z13Eta(raw);
            carry = next_carry;
        }
        
        WittVector { prec, data: out }
    }

    pub fn wneg(x: &WittVector) -> WittVector {
        let mut out = vec![Z13Eta::ZERO; x.prec];
        for i in 0..x.prec {
            out[i] = -x.data[i];
        }
        WittVector { prec: x.prec, data: out }
    }

    pub fn wsub(x: &WittVector, y: &WittVector) -> WittVector {
        Self::wadd(x, &Self::wneg(y))
    }

    pub fn wmul(x: &WittVector, y: &WittVector) -> WittVector {
        let prec = min(x.prec, y.prec);
        let mut l = vec![[0u16; 3]; 2 * prec];
        
        for i in 0..prec {
            for j in 0..prec {
                let a = x.data[i].0;
                let b = y.data[j].0;
                let c0 = a[0] as u16 * b[0] as u16;
                let c1 = a[0] as u16 * b[1] as u16 + a[1] as u16 * b[0] as u16;
                let c2 = a[0] as u16 * b[2] as u16 + a[1] as u16 * b[1] as u16 + a[2] as u16 * b[0] as u16;
                let c3 = a[1] as u16 * b[2] as u16 + a[2] as u16 * b[1] as u16;
                let c4 = a[2] as u16 * b[2] as u16;
                
                let exact = [
                    c0 + c3 + c4,
                    c1 + c3 + 2 * c4,
                    c2 + c3 + 2 * c4,
                ];
                
                let target_idx = i + j;
                if target_idx < 2 * prec {
                    l[target_idx][0] += exact[0];
                    l[target_idx][1] += exact[1];
                    l[target_idx][2] += exact[2];
                }
            }
        }
        
        let mut out = vec![Z13Eta::ZERO; prec];
        let mut carry = [0u16; 3];
        
        for i in 0..prec {
            let mut raw = [0u8; 3];
            let mut next_carry = [0u16; 3];
            for j in 0..3 {
                let sum = l[i][j] + carry[j];
                raw[j] = (sum % P as u16) as u8;
                next_carry[j] = sum / P as u16;
            }
            out[i] = Z13Eta(raw);
            carry = next_carry;
        }
        
        WittVector { prec, data: out }
    }
}
