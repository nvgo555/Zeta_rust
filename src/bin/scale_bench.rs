use std::time::Instant;
use ndarray::{Array2, Array3};
use zeta_rs::model::ZetaModel;
use zeta_rs::ring::Z13Eta;

fn main() {
    println!("==================================================");
    println!("            ZETA-RS DETAIL SCALE BENCHMARK        ");
    println!("==================================================");

    // --- BENCHMARK 1: Depth (N) scaling ---
    // Here we test depth with sequence length L=64, V=256, D=54
    println!("\n--- 1. Depth (N) Scaling Benchmark (B=4, L=64, V=256, D=54) ---");
    let depths = vec![1, 11, 50, 100, 500];
    for n in depths {
        let t0 = Instant::now();
        let mut model = ZetaModel::new(256, 54, n);
        let init_ms = t0.elapsed().as_secs_f64() * 1000.0;
        
        let tokens = Array2::from_elem((4, 64), 42usize);
        
        // Warmup
        model.forward(&tokens, false);
        
        let reps = 5;
        let t1 = Instant::now();
        for _ in 0..reps {
            model.forward(&tokens, false);
        }
        let forward_ms = t1.elapsed().as_secs_f64() * 1000.0 / reps as f64;
        
        // Weight Memory calculation
        let hd = 54 / 3;
        let layer_mem = n * (6 * 54 * hd + 2 * hd * 54) * 3;
        let embed_mem = 256 * 54 * 3;
        let head_mem = 54 * 256 * 3;
        let witt_head_mem = 54 * 256 * 4 * 3;
        let total_bytes = layer_mem + embed_mem + head_mem + witt_head_mem;
        let total_kb = total_bytes as f64 / 1024.0;
        
        println!(
            "N = {:3} | Init: {:8.2} ms | Forward: {:8.2} ms | Weight Mem: {:8.2} KB",
            n, init_ms, forward_ms, total_kb
        );
    }

    // --- BENCHMARK 2: Sequence Length (L) scaling ---
    // Here we fix B=1 (inference scenario) and B=4 (training/batch scenario)
    // with N=1 layer, D=54, V=256.
    println!("\n--- 2. Sequence Length (L) Scaling Benchmark (N=1, V=256, D=54) ---");
    let seq_lengths = vec![64, 256, 1024, 4096, 16384];
    for &l in &seq_lengths {
        for &b in &[1, 4] {
            let mut model = ZetaModel::new(256, 54, 1);
            let tokens = Array2::from_elem((b, l), 42usize);
            
            // Limit iterations to prevent extremely long test runs
            let reps = if l > 20000 { 1 } else { 3 };
            
            // Warmup
            model.forward(&tokens, false);
            
            let t0 = Instant::now();
            for _ in 0..reps {
                model.forward(&tokens, false);
            }
            let forward_ms = t0.elapsed().as_secs_f64() * 1000.0 / reps as f64;
            
            // Calculate output logits tensor size in memory
            let logits_mem_kb = (b * l * 256) as f64 / 1024.0;
            
            println!(
                "B = {} | L = {:6} | Forward: {:8.2} ms | Output Logits: {:8.2} KB",
                b, l, forward_ms, logits_mem_kb
            );
        }
    }

    // --- BENCHMARK 3: Vocab Size (V) scaling ---
    // Fix B=4, L=64, N=1, D=54, scale V up to 100k
    println!("\n--- 3. Vocabulary Size (V) Scaling Benchmark (B=4, L=64, N=1, D=54) ---");
    let vocabs = vec![256, 1024, 8192,32000];
    for v in vocabs {
        let t0 = Instant::now();
        let mut model = ZetaModel::new(v, 54, 1);
        let init_ms = t0.elapsed().as_secs_f64() * 1000.0;
        
        let tokens = Array2::from_elem((4, 64), 42 % v);
        
        // Warmup
        model.forward(&tokens, false);
        
        let reps = 3;
        let t1 = Instant::now();
        for _ in 0..reps {
            model.forward(&tokens, false);
        }
        let forward_ms = t1.elapsed().as_secs_f64() * 1000.0 / reps as f64;
        
        // Memory calculation
        let hd = 54 / 3;
        let layer_mem = 1 * (6 * 54 * hd + 2 * hd * 54) * 3;
        let embed_mem = v * 54 * 3;
        let head_mem = 54 * v * 3;
        let witt_head_mem = 54 * v * 4 * 3;
        let total_bytes = layer_mem + embed_mem + head_mem + witt_head_mem;
        let total_mb = total_bytes as f64 / (1024.0 * 1024.0);
        
        println!(
            "V = {:6} | Init: {:8.2} ms | Forward: {:8.2} ms | Weight Mem: {:8.2} MB",
            v, init_ms, forward_ms, total_mb
        );
    }
}
