use std::time::Instant;
use ndarray::{Array2, Array3};
use zeta_rs::constants::{ETA_POW, T3_POW};
use zeta_rs::ring::Z13Eta;
use zeta_rs::model::ZetaModel;
use zeta_rs::kernel::PAdicKernel;
use zeta_rs::ntt::Ntt;
use zeta_rs::witt::WittVector;

fn delta_max(l: usize) -> usize {
    let mut mx = 0;
    for i in 0..l {
        for j in 0..l {
            let diff = if i > j { i - j } else { j - i };
            let val = zeta_rs::constants::val_13(diff);
            if val as usize > mx {
                mx = val as usize;
            }
        }
    }
    mx
}

fn main() {
    println!("--- AXIOMS VERIFICATION ---");
    let axioms_passed = zeta_rs::axioms::AxiomVerifier::verify_all();
    if axioms_passed {
        println!("All core algebraic axioms passed. The mathematical foundation is solid!");
    } else {
        println!("Warning: Some axioms failed.");
    }

    println!("\n--- INIT ---");
    let v = 256;
    let d = 54;
    let n = 11;
    let b = 4;
    let l = 64;
    let mut model = ZetaModel::new(v, d, n);
    
    println!("\n--- BENCHMARK ---");
    let mut tokens = Array2::from_elem((b, l), 0usize);
    for i in 0..b {
        for j in 0..l {
            tokens[[i, j]] = (i * j + 42) % v;
        }
    }
    
    for _ in 0..3 {
        model.forward(&tokens, false);
    }
    
    let n_reps = 20;
    let t0 = Instant::now();
    for _ in 0..n_reps {
        model.forward(&tokens, false);
    }
    let forward_ms = t0.elapsed().as_secs_f64() * 1000.0 / n_reps as f64;
    
    let a = ETA_POW[1];
    let b_eta = ETA_POW[3];
    let t0 = Instant::now();
    for _ in 0..100_000 {
        let _ = a * b_eta;
    }
    let ring_mul_ns = t0.elapsed().as_secs_f64() * 1_000_000_000.0 / 100_000.0;
    
    let t0 = Instant::now();
    for _ in 0..100_000 {
        let _ = a.inv();
    }
    let ring_inv_ns = t0.elapsed().as_secs_f64() * 1_000_000_000.0 / 100_000.0;
    
    let t0 = Instant::now();
    for _ in 0..100_000 {
        let _ = T3_POW[42];
    }
    let t3_lookup_ns = t0.elapsed().as_secs_f64() * 1_000_000_000.0 / 100_000.0;
    
    let x_test = Array3::from_elem((1, l, 8), Z13Eta::ZERO); 
    let t0 = Instant::now();
    for _ in 0..100 {
        PAdicKernel::apply(&x_test); 
    }
    let kernel_tree_ms = t0.elapsed().as_secs_f64() * 1000.0 / 100.0;
    
    let mut ntt_us = vec![];
    for &n_size in &[4, 12] {
        let mut x = vec![Z13Eta::ZERO; n_size];
        for i in 0..n_size {
            x[i] = ETA_POW[i];
        }
        let mut out = vec![Z13Eta::ZERO; n_size];
        let t0 = Instant::now();
        for _ in 0..10_000 {
            Ntt::ntt_mut(&x, &mut out);
        }
        ntt_us.push((n_size, t0.elapsed().as_secs_f64() * 1_000_000_000.0 / 10_000.0));
    }
    
    let xw = WittVector::from_ring(a, 4);
    let yw = WittVector::from_ring(b_eta, 4);
    let t0 = Instant::now();
    for _ in 0..50_000 {
        WittVector::wadd(&xw, &yw);
    }
    let witt_add_ns = t0.elapsed().as_secs_f64() * 1_000_000_000.0 / 50_000.0;
    
    let t0 = Instant::now();
    for _ in 0..20_000 {
        WittVector::wmul(&xw, &yw);
    }
    let witt_mul_ns = t0.elapsed().as_secs_f64() * 1_000_000_000.0 / 20_000.0;
    
    let t0 = Instant::now();
    for _ in 0..1_000_000 {
        let _ = delta_max(l);
    }
    let delta_max_us = t0.elapsed().as_secs_f64() * 1_000_000.0 / 1_000_000.0;
    
    println!("==================================================");
    println!(" ZetaRuntime Benchmark  Rust Port  B={}  L={}", b, l);
    println!("--------------------------------------------------");
    println!("  {:28} {:>8.3} ms", "forward_ms", forward_ms);
    println!("  {:28} {:>8.3} ms", "kernel_tree_ms", kernel_tree_ms);
    println!("  {:28} {:>8.3} ns", "ring_mul_ns", ring_mul_ns);
    println!("  {:28} {:>8.3} ns", "ring_inv_ns", ring_inv_ns);
    println!("  {:28} {:>8.3} ns", "t3_lookup_ns", t3_lookup_ns);
    for (n_size, ns) in ntt_us {
        println!("  {:28} {:>8.3} ns", format!("ntt_{}_ns", n_size), ns);
    }
    println!("  {:28} {:>8.3} ns", "witt_add_ns", witt_add_ns);
    println!("  {:28} {:>8.3} ns", "witt_mul_ns", witt_mul_ns);
    println!("  {:28} {:>8.3} µs", "delta_max_us", delta_max_us);
    println!("==================================================");
}
