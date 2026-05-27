//! Core mathematical definitions for the finite cubic ring $\mathbb{Z}_{13}[\eta]$.
//!
//! This ring is defined as $\mathbb{Z}_{13}\[X\] / (X^3 - X^2 - X - 1)$. It contains 2197 elements.
//! All neural activations, weights, and states in the Zeta engine are elements of this ring.

use std::ops::{Add, Sub, Mul, Neg};
use crate::constants::INV_TBL;

pub const P: u8 = 13;
pub const ORD: u16 = 168;

/// An element in the finite ring $\mathbb{Z}_{13}[\eta]$.
///
/// Internally stored as an array of 3 coefficients in $\mathbb{Z}_{13}$, representing $a_0 + a_1\eta + a_2\eta^2$.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct Z13Eta(pub [u8; 3]);

impl Z13Eta {
    /// The additive identity $0$.
    pub const ZERO: Self = Z13Eta([0, 0, 0]);
    
    /// The multiplicative identity $1$.
    pub const ONE: Self = Z13Eta([1, 0, 0]);

    /// Computes the multiplicative inverse using a precomputed 13x13x13 lookup table.
    #[inline(always)]
    pub fn inv(&self) -> Self {
        INV_TBL[[self.0[0] as usize, self.0[1] as usize, self.0[2] as usize]]
    }

    /// Multiplies the ring element by a scalar $k \in \mathbb{Z}_{13}$.
    #[inline(always)]
    pub fn smul(&self, k: u8) -> Self {
        let k_mod = k % P;
        Z13Eta([
            (self.0[0] * k_mod) % P,
            (self.0[1] * k_mod) % P,
            (self.0[2] * k_mod) % P,
        ])
    }

    /// Computes the field trace $\mathrm{Tr}_{\mathbb{Z}_{13}[\eta] / \mathbb{Z}_{13}}(x)$.
    /// Used for projecting ring elements down to the base field for attention scoring.
    #[inline(always)]
    pub fn trace(&self) -> u8 {
        (3 * self.0[0] + self.0[1] + 3 * self.0[2]) % P
    }

    /// Computes the algebraic norm $\mathrm{N}(x)$ via the determinant of the multiplication matrix.
    #[inline(always)]
    pub fn norm(&self) -> u8 {
        let a0 = self.0[0]; let a1 = self.0[1]; let a2 = self.0[2];
        let r0 = [a0, a2, (a1 + a2) % P];
        let r1 = [a1, (a0 + a2) % P, (a1 + 2 * a2) % P];
        let r2 = [a2, (a1 + a2) % P, (a0 + a1 + 2 * a2) % P];
        
        let det = r0[0] as i32 * (r1[1] as i32 * r2[2] as i32 - r1[2] as i32 * r2[1] as i32)
                - r0[1] as i32 * (r1[0] as i32 * r2[2] as i32 - r1[2] as i32 * r2[0] as i32)
                + r0[2] as i32 * (r1[0] as i32 * r2[1] as i32 - r1[1] as i32 * r2[0] as i32);
        
        ((det % P as i32 + P as i32) % P as i32) as u8
    }

    #[inline(always)]
    pub fn phi1(&self) -> u8 {
        (self.0[0] + 7 * self.0[1] + 10 * self.0[2]) % P
    }

    #[inline(always)]
    pub fn phi2(&self) -> F169 {
        F169([
            (self.0[0] + 11 * self.0[2]) % P,
            (self.0[1] + 7 * self.0[2]) % P,
        ])
    }
}

impl Add for Z13Eta {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Z13Eta([
            (self.0[0] + rhs.0[0]) % P,
            (self.0[1] + rhs.0[1]) % P,
            (self.0[2] + rhs.0[2]) % P,
        ])
    }
}

impl Sub for Z13Eta {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Z13Eta([
            (self.0[0] + P - rhs.0[0]) % P,
            (self.0[1] + P - rhs.0[1]) % P,
            (self.0[2] + P - rhs.0[2]) % P,
        ])
    }
}

impl Mul for Z13Eta {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        let a0 = self.0[0] as u16; let a1 = self.0[1] as u16; let a2 = self.0[2] as u16;
        let b0 = rhs.0[0] as u16; let b1 = rhs.0[1] as u16; let b2 = rhs.0[2] as u16;
        
        let c0 = a0 * b0;
        let c1 = a0 * b1 + a1 * b0;
        let c2 = a0 * b2 + a1 * b1 + a2 * b0;
        let c3 = a1 * b2 + a2 * b1;
        let c4 = a2 * b2;
        
        Z13Eta([
            ((c0 + c3 + c4) % P as u16) as u8,
            ((c1 + c3 + 2 * c4) % P as u16) as u8,
            ((c2 + c3 + 2 * c4) % P as u16) as u8,
        ])
    }
}

impl Neg for Z13Eta {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        Z13Eta([
            (P - self.0[0]) % P,
            (P - self.0[1]) % P,
            (P - self.0[2]) % P,
        ])
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct F169(pub [u8; 2]);

impl Mul for F169 {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        let a0 = self.0[0] as u16; let a1 = self.0[1] as u16;
        let b0 = rhs.0[0] as u16; let b1 = rhs.0[1] as u16;
        let c0 = (a0 * b0 + 11 * a1 * b1) % P as u16;
        let c1 = (a0 * b1 + a1 * b0 + 7 * a1 * b1) % P as u16;
        F169([c0 as u8, c1 as u8])
    }
}
