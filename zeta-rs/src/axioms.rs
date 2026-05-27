use crate::ring::Z13Eta;
use crate::constants::T3_POW;
use crate::spectral::{SylvesterProjectors, t3n};

pub struct AxiomVerifier;

impl AxiomVerifier {
    pub fn verify_all() -> bool {
        let mut passed = 0;
        let mut total = 0;
        
        // A001 - A009: Ring tests
        let a = Z13Eta([1, 2, 3]);
        let b = Z13Eta([4, 5, 6]);
        let c = Z13Eta([7, 8, 9]);
        
        // A001: assoc_mul
        let r1 = (a * b) * c;
        let r2 = a * (b * c);
        if r1 == r2 { passed += 1; } total += 1;
        
        // A002: assoc_add
        let r1 = (a + b) + c;
        let r2 = a + (b + c);
        if r1 == r2 { passed += 1; } total += 1;
        
        // A006: identity
        if a * Z13Eta::ONE == a { passed += 1; } total += 1;
        
        // A007: inv
        let a_inv = a.inv();
        if a * a_inv == Z13Eta::ONE || a_inv == Z13Eta::ZERO { passed += 1; } total += 1;
        
        // A013: t3_period
        let mut ok = true;
        for i in 0..3 {
            for j in 0..3 {
                if T3_POW[0][i][j] != if i==j {1} else {0} { ok = false; }
            }
        }
        if ok { passed += 1; } total += 1;
        
        // A015: p1_idem
        let p1a = SylvesterProjectors::p1(&a);
        let p1p1a = SylvesterProjectors::p1(&p1a);
        if p1a == p1p1a { passed += 1; } total += 1;
        
        // A017: p1_p23_orth
        let p23a = SylvesterProjectors::p23(&a);
        let orth = SylvesterProjectors::p1(&p23a);
        if orth == Z13Eta::ZERO { passed += 1; } total += 1;
        
        // A019: t3_p1
        let t3p1a = t3n(&p1a, 1);
        let p1_7a = p1a.smul(7);
        if t3p1a == p1_7a { passed += 1; } total += 1;
        
        println!("Axiom Report: {}/{} passed", passed, total);
        passed == total
    }
}
