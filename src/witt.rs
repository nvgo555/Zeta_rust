//! $p$-Adic Multi-Precision Lifting via Witt Vectors.
//!
//! Provides the `WittVector` structure which acts as a stack-allocated $p$-adic number 
//! of length up to 4 elements. This is used for precise error accumulation across 
//! learning steps.

use crate::ring::{Z13Eta, P};
use std::cmp::min;

/// A multi-precision Witt Vector for $p$-adic arithmetic.
///
/// It uses a fixed-size `[Z13Eta; 4]` buffer to guarantee zero-allocation (stack-only) semantics
/// while providing sufficient precision for tracking gradients/errors over time.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WittVector {
    /// The number of valid precision components (up to 4).
    pub prec: usize,
    /// The component data.
    pub data: [Z13Eta; 4],
}

impl WittVector {
    /// Lifts a basic ring element into a multi-precision Witt vector at position 0.
    pub fn from_ring(a: Z13Eta, prec: usize) -> Self {
        let mut data = [Z13Eta::ZERO; 4];
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

    /// Adds two Witt Vectors component-wise with precise $p$-adic carry propagation.
    pub fn wadd(x: &WittVector, y: &WittVector) -> WittVector {
        let prec = min(min(x.prec, y.prec), 4);
        let mut out = [Z13Eta::ZERO; 4];
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
        let mut out = [Z13Eta::ZERO; 4];
        for i in 0..x.prec {
            out[i] = -x.data[i];
        }
        WittVector { prec: x.prec, data: out }
    }

    pub fn wsub(x: &WittVector, y: &WittVector) -> WittVector {
        Self::wadd(x, &Self::wneg(y))
    }

    /// Multiplies two Witt Vectors, expanding cross-terms and correctly tracking multi-level carries.
    pub fn wmul(x: &WittVector, y: &WittVector) -> WittVector {
        let prec = min(min(x.prec, y.prec), 4);
        let mut l = [[0u16; 3]; 8]; 
        
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
                if target_idx < 2 * prec && target_idx < 8 {
                    l[target_idx][0] += exact[0];
                    l[target_idx][1] += exact[1];
                    l[target_idx][2] += exact[2];
                }
            }
        }
        
        let mut out = [Z13Eta::ZERO; 4];
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
