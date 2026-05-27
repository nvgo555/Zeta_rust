//! # Zeta p-Adic Integer AI Engine
//!
//! This crate implements the Zeta AI engine entirely within the finite cubic ring $\mathbb{Z}_{13}[\eta] / (\eta^3 - \eta^2 - \eta - 1)$.
//! It completely bypasses floating-point arithmetic, gradient descent, and real-valued weights, 
//! instead relying on strict ultrametric topology and algebraic geometry for state evolution and learning.
//!
//! ## Core Components
//! * `ring`: The foundational mathematics of $\mathbb{Z}_{13}[\eta]$.
//! * `witt`: $p$-adic multi-precision lifting via fixed-size stack-allocated Witt Vectors.
//! * `ntt`: Fast $O(N \log N)$ convolution using Number Theoretic Transforms over order-168 roots of unity.
//! * `buchberger`: Non-gradient learning using the Nullstellensatz over polynomial ideals.
//! * `model`: The top-level `ZetaModel` neural architecture.

pub mod constants;
pub mod ring;
pub mod spectral;
pub mod ntt;
pub mod witt;
pub mod kernel;
pub mod hilbert;
pub mod linear;
pub mod buchberger;
pub mod axioms;
pub mod embed;
pub mod attention;
pub mod gqm;
pub mod model;
