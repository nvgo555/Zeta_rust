use zeta_rs::axioms::AxiomVerifier;

fn main() {
    println!("Zeta p-adic AI (Rust Port)");
    println!("==========================");
    let success = AxiomVerifier::verify_all();
    if success {
        println!("All core algebraic axioms passed. The mathematical foundation is solid!");
    } else {
        println!("Warning: Some axioms failed.");
    }
}
