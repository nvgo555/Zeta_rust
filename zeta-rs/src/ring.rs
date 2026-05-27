use std::ops::{Add, Sub, Mul, Neg};
use crate::constants::INV_TBL;

pub const P: u8 = 13;
pub const ORD: u16 = 168;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct Z13Eta(pub [u8; 3]);

impl Z13Eta {
    pub const ZERO: Self = Z13Eta([0, 0, 0]);
    pub const ONE: Self = Z13Eta([1, 0, 0]);

    pub fn inv(&self) -> Self {
        INV_TBL[[self.0[0] as usize, self.0[1] as usize, self.0[2] as usize]]
    }

    pub fn smul(&self, k: u8) -> Self {
        let k_mod = k % P;
        Z13Eta([
            (self.0[0] * k_mod) % P,
            (self.0[1] * k_mod) % P,
            (self.0[2] * k_mod) % P,
        ])
    }

    pub fn trace(&self) -> u8 {
        (3 * self.0[0] + self.0[1] + 3 * self.0[2]) % P
    }

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

    pub fn phi1(&self) -> u8 {
        (self.0[0] + 7 * self.0[1] + 10 * self.0[2]) % P
    }

    pub fn phi2(&self) -> F169 {
        F169([
            (self.0[0] + 11 * self.0[2]) % P,
            (self.0[1] + 7 * self.0[2]) % P,
        ])
    }
}

impl Add for Z13Eta {
    type Output = Self;
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
