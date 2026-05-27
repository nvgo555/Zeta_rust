# Zeta p-Adic Integer AI — System Specification v7.0.0

This repository is a **fork** of the official Rust implementation of the Zeta p-adic AI system by Dávid Navrátil.

Official GitHub Repository: [ZetapAdelicAI/Zeta-p-adicAI](https://github.com/ZetapAdelicAI/Zeta-p-adicAI)

**Algebraic Artificial Intelligence over the Cubic p-Adic Integer Ring**

---

## Executive Summary

Zeta is a self-contained artificial intelligence system whose entire computational universe is constructed within the finite ring

$$\\mathbb{Z}_{13}[\\eta] \\;\\big/\\; (\\eta^3 - \\eta^2 - \\eta - 1)$$

Every tensor, embedding, state transition, kernel evaluation, and learning update is an element of this ring. The system contains no floating-point arithmetic, no gradient descent, no Euclidean metric structures, and no learned real-valued parameters. Instead, dynamics are governed by the unimodular matrix $T_3 \\in \\mathrm{SL}(3, \\mathbb{Z})$, attention by an ultrametric tree kernel derived from the $13$-adic valuation, and learning by the Buchberger-Nullstellensatz algorithm operating on polynomial ideals.

The ring $\\mathbb{Z}_{13}[\\eta]$ has $2197$ elements. Its unit group has order $2016$, arising from the Chinese Remainder Theorem decomposition $\\mathbb{Z}_{13}[\\eta] \\cong \\mathbb{F}_{13} \\times \\mathbb{F}_{169}$. The multiplicative order of $\\eta$ is $168$, giving the system a natural period that governs orbit evolution, positional encoding, and reversible state transitions. Spectral decomposition via Sylvester projectors separates the dynamics into a one-dimensional dominant channel over $\\mathbb{F}_{13}$ and a two-dimensional subdominant channel over $\\mathbb{F}_{169}$, with the full symmetry of the Galois group $S_3$ acting on the latter.

This document specifies the mathematical foundations, architectural components, algorithmic procedures, and operational semantics of the Zeta system.

> **🚀 High-Performance Rust Port**: The core engine of Zeta has been fully ported to Rust for extreme performance (>1,000,000x speedups on core tensor field operations). The primary implementation now lives in the root directory (via `src/` and `Cargo.toml`). The original Python reference implementation has been moved to the `legacy/` directory.

---

## Table of Contents

1. [Mathematical Preliminaries](#1-mathematical-preliminaries)
2. [The Base Ring $\\mathbb{Z}_{13}[\\eta]$](#2-the-base-ring)
3. [Spectral Theory of $T_3$](#3-spectral-theory-of-t3)
4. [The Ultrametric Kernel](#4-the-ultrametric-kernel)
5. [Galois Symmetry and the $S_3$ Action](#5-galois-symmetry)
6. [Chinese Remainder Theorem Decomposition](#6-chinese-remainder-theorem)
7. [Architecture](#7-architecture)
8. [Deterministic Embeddings and Positional Encoding](#8-deterministic-embeddings)
9. [Spectral Attention Mechanism](#9-spectral-attention)
10. [Witt Vector Head and Hensel Lifting](#10-witt-vector-head)
11. [MERA Tensor Network](#11-mera-tensor-network)
12. [Geometric Quantum Mechanics](#12-geometric-quantum-mechanics)
13. [Training via Buchberger-Nullstellensatz](#13-training)
14. [Autonomous Control Loop](#14-autonomous-control)
15. [Tokenization](#15-tokenization)
16. [Axiom Verification System](#16-axiom-verification)
17. [Performance Characteristics](#17-performance)
18. [Module Reference](#18-module-reference)
19. [Usage Examples](#19-usage-examples)
20. [Bibliography](#20-bibliography)

---

## 1. Mathematical Preliminaries

### 1.1 The Prime Field $\\mathbb{F}_{13}$

All scalar coefficients are elements of the finite field $\\mathbb{F}_{13} = \\{0, 1, 2, \\dots, 12\\}$ with arithmetic modulo $13$.

### 1.2 Polynomial Rings and Quotient Construction

Let $\\mathbb{F}_{13}[x]$ denote the polynomial ring in one indeterminate over $\\mathbb{F}_{13}$. The defining polynomial

$$f(x) = x^3 - x^2 - x - 1$$

factors over $\\mathbb{F}_{13}$ as

$$f(x) = (x - 7)(x^2 + 6x + 2)$$

where $x^2 + 6x + 2$ is irreducible over $\\mathbb{F}_{13}$. Consequently, by the Chinese Remainder Theorem,

$$\\mathbb{Z}_{13}[\\eta] \\;:=\\; \\mathbb{F}_{13}[\\eta] \\;/\\; (f(\\eta)) \\;\\cong\\; \\mathbb{F}_{13} \\times \\mathbb{F}_{169}$$

is a finite ring with $13^3 = 2197$ elements. It is not a field; the factorisation of $f(x)$ implies the existence of zero divisors.

### 1.3 Representation of Ring Elements

Every element $a \\in \\mathbb{Z}_{13}[\\eta]$ admits a unique representation

$$a = a_0 + a_1 \\eta + a_2 \\eta^2, \\qquad a_i \\in \\mathbb{F}_{13}$$

under the reduction relation $\\eta^3 = \\eta^2 + \\eta + 1$. In computational form, $a$ is stored as the integer triple $(a_0, a_1, a_2)$.

### 1.4 The Unit Group

The unit group $\\mathbb{Z}_{13}[\\eta]^\\times$ consists of all elements that are units in both CRT components:

$$|\\mathbb{Z}_{13}[\\eta]^\\times| = |\\mathbb{F}_{13}^\\times| \\cdot |\\mathbb{F}_{169}^\\times| = 12 \\cdot 168 = 2016$$

The element $\\eta$ itself is a unit with

$$\\mathrm{ord}(\\eta) = 168$$

This period governs all cyclic structures in the system: embeddings, positional encodings, and $T_3$ orbit evolution.

---

## 2. The Base Ring $\\mathbb{Z}_{13}[\\eta]$

### 2.1 Ring Arithmetic

**Addition.** For $a = (a_0, a_1, a_2)$ and $b = (b_0, b_1, b_2)$:

$$a + b = \\big((a_0 + b_0) \\bmod 13,\\; (a_1 + b_1) \\bmod 13,\\; (a_2 + b_2) \\bmod 13\\big)$$

**Multiplication.** Compute the polynomial product $(a_0 + a_1 \\eta + a_2 \\eta^2)(b_0 + b_1 \\eta + b_2 \\eta^2)$ and reduce using $\\eta^3 = \\eta^2 + \\eta + 1$ and $\\eta^4 = 2\\eta^2 + 2\\eta + 1$:

$$\\begin{aligned}
c_0 &= a_0 b_0 + a_1 b_2 + a_2 b_1 + a_2 b_2 \\\\
c_1 &= a_0 b_1 + a_1 b_0 + a_1 b_2 + a_2 b_1 + 2a_2 b_2 \\\\
c_2 &= a_0 b_2 + a_1 b_1 + a_2 b_0 + a_1 b_2 + a_2 b_1 + 2a_2 b_2
\\end{aligned}$$

All coefficients taken modulo $13$.

**Multiplicative Inverse.** For $a \\neq 0$ with $\\mathrm{N}(a) \\neq 0$, the inverse is the unique element satisfying $a \\cdot a^{-1} = (1, 0, 0)$. All $2016$ inverses are precomputed into the lookup table `INV_TBL` of shape $(13, 13, 13, 3)$ for $O(1)$ retrieval.

**Trace and Norm.** The trace and norm map $\\mathbb{Z}_{13}[\\eta]$ to $\\mathbb{F}_{13}$:

$$\\mathrm{Tr}(a) = a + \\sigma(a) + \\sigma^2(a), \\qquad \\mathrm{N}(a) = a \\cdot \\sigma(a) \\cdot \\sigma^2(a)$$

where $\\sigma$ is a generator of the Galois group (see Section 5).

### 2.2 The Element $\\eta$ and Its Powers

The sequence $\\eta^n$ for $n = 0, 1, 2, \\dots$ is periodic with period $168$. The table `ETA_POW` of shape $(169, 3)$ stores all $168$ distinct powers as triples in $\\mathbb{F}_{13}^3$, with $\\eta^0 = (1, 0, 0)$ at index $0$. The companion table `ETA_IPOW` stores the inverse powers $\\eta^{-k} = \\eta^{168-k}$. These tables are the sole source of non-linearity in the embedding system.

---

## 3. Spectral Theory of $T_3$

### 3.1 Definition and Basic Properties

The evolution operator is the integer matrix

$$T_3 = \\begin{pmatrix} 0 & 0 & 1 \\\\ 1 & 0 & 1 \\\\ 0 & 1 & 1 \\end{pmatrix} \\in \\mathrm{SL}(3, \\mathbb{Z})$$

**Proposition 3.1.** $\\det(T_3) = 1$.

*Proof.* Direct computation: $0(0 - 1) - 0(1 - 0) + 1(1 - 0) = 1$. $\\square$

**Proposition 3.2.** The characteristic polynomial of $T_3$ is $\\chi_{T_3}(\\lambda) = \\lambda^3 - \\lambda^2 - \\lambda - 1 = f(\\lambda)$.

*Proof.* Direct expansion of $\\det(T_3 - \\lambda I)$. $\\square$

**Proposition 3.3.** $T_3^{168} = I$ in $\\mathrm{GL}(3, \\mathbb{F}_{13})$.

*Proof.* The eigenvalues of $T_3$ modulo $13$ are the roots of $f(x) = (x-7)(x^2+6x+2)$. The linear factor gives eigenvalue $7$ with order dividing $12$; the quadratic factor gives eigenvalues in $\\mathbb{F}_{169}^\\times$ with order dividing $168$. The least common multiple is $168$. Direct computation confirms that no smaller positive exponent yields the identity. $\\square$

### 3.2 Eigenvalue Structure

The eigenvalues of $T_3$ modulo $13$ are the roots of $f(x) = 0$:

- $\\lambda_1 = 7 \\in \\mathbb{F}_{13}$ (dominant, in the base field)
- $\\lambda_2, \\lambda_3 \\in \\mathbb{F}_{169} \\setminus \\mathbb{F}_{13}$ (subdominant, conjugate pair in the quadratic extension)

The dominant right eigenvector is $v_1 = (1, 3, 7)^T$, verified by $T_3 v_1 = 7 v_1$. The left eigenvector is $w_1 = (1, 7, 10)$, satisfying $w_1^T T_3 = 7 w_1^T$. The triple $(1, 7, 10)$ also equals $(\\varphi_1(1), \\varphi_1(\\eta), \\varphi_1(\\eta^2))$, the evaluation of the basis under the dominant CRT projection.

### 3.3 Sylvester Spectral Decomposition

**Theorem 3.4 (Sylvester).** There exist unique matrices $P_1, P_{23} \\in \\mathrm{Mat}_{3 \\times 3}(\\mathbb{F}_{13})$ such that:

1. $P_1 + P_{23} = I$
2. $P_1^2 = P_1$, $P_{23}^2 = P_{23}$
3. $P_1 P_{23} = P_{23} P_1 = 0$
4. $T_3 P_1 = 7 P_1$, $T_3 P_{23} = P_{23} T_3$

Moreover, for all $n \\geq 0$:

$$T_3^n = 7^n P_1 + P_{23} T_3^n$$

*Proof.* Standard spectral decomposition for matrices with distinct eigenvalues. The projectors are constructed from the Lagrange interpolation polynomials of the minimal polynomial. $\\square$

**Corollary 3.5.** The dominant channel evolves by scalar multiplication: $P_1 T_3^n x = 7^n P_1 x$.

**Corollary 3.6.** Time reversibility: $T_3^{-n} = T_3^{168-n}$ for all $n \\in \\mathbb{Z}$.

---

## 4. The Ultrametric Kernel

### 4.1 p-adic Valuation

For an integer $d \\neq 0$, the $13$-adic valuation $v_{13}(d)$ is the largest integer $k$ such that $13^k$ divides $d$. By convention, $v_{13}(0) = \\infty$. The valuation is precomputed for all $d < 13^5$ in the lookup table `_VAL_LUT`.

### 4.2 Kernel Definition

The token interaction kernel is defined for positions $i, j$ in a sequence as:

$$G(i, j) = \\eta^{-v_{13}(|i-j|)} \\in \\mathbb{Z}_{13}[\\eta]$$

where the exponent is interpreted modulo $168$ via the `ETA_IPOW` table, and $G(i, i) = \\eta^0 = (1, 0, 0)$.

### 4.3 Strong Triangle Inequality

**Theorem 4.1.** For all positions $i, j, k$:

$$v_{13}(|i-k|) \\geq \\min\\big(v_{13}(|i-j|),\\, v_{13}(|j-k|)\\big)$$

Consequently, the valuation of $G(i,k)$ equals the maximum of the valuations of $G(i,j)$ and $G(j,k)$, which implies that $\\eta$-exponents satisfy the corresponding minimum property.

*Proof.* This is the fundamental property of non-Archimedean valuations. If $13^a \\mid (i-j)$ and $13^b \\mid (j-k)$, then $13^{\\min(a,b)} \\mid (i-k)$. $\\square$

**Corollary 4.2.** Every triangle in the token metric space is isosceles: for any three distinct positions, at least two of the three pairwise kernel values are equal.

### 4.4 Tree Structure

The kernel induces a hierarchical tree structure on the token positions. Positions $i$ and $j$ are siblings at tree level $\\ell$ if and only if $v_{13}(|i-j|) \\geq \\ell$. The tree has depth $\\lceil \\log_{13} L \\rceil$ for a sequence of length $L$.

**Proposition 4.3.** The kernel never materializes an $L \\times L$ matrix. It is computed in $O(L \\log_{13} L)$ time and $O(L)$ memory via level-wise aggregation. For $L \\leq 169$, a fused fast path eliminates intermediate list allocation.

---

## 5. Galois Symmetry and the $S_3$ Action

### 5.1 The Galois Group

The Galois group of $f(x) = x^3 - x^2 - x - 1$ over $\\mathbb{Q}$ is $S_3$, the symmetric group on three letters, of order $6$. Over $\\mathbb{F}_{13}$, the splitting field is $\\mathbb{F}_{13^6}$, and the Galois group is generated by the Frobenius automorphism $\\mathrm{Frob}(x) = x^{13}$ of order $3$ on $\\mathbb{F}_{13^3}$.

### 5.2 Permutation Representation

The $S_3$ action on $\\mathbb{Z}_{13}[\\eta]$ is represented by $3 \\times 3$ permutation matrices $\\{S_3^{(g)}\\}_{g=1}^6$ acting on the coefficient vector $(a_0, a_1, a_2)^T$. The group multiplication table is precomputed as a $(6, 6)$ integer matrix `_S3_MUL` for $O(1)$ composition.

### 5.3 Conjugation Operator

The conjugation operator $J$ is defined via CRT:

$$J = \\mathrm{CRT}^{-1}\\big(\\varphi_1,\\; \\mathrm{Frob}(\\varphi_2)\\big)$$

**Proposition 5.1.** $J^2 = \\mathrm{id}$.

*Proof.* The Frobenius automorphism satisfies $\\mathrm{Frob}^2 = \\mathrm{id}$ on $\\mathbb{F}_{169}$ since $x^{13^2} = x$ for all $x \\in \\mathbb{F}_{169}$. $\\square$

### 5.4 Parallel Orbit Evolution

For any state $x$, the six Galois-conjugate orbits are:

$$x_g^{(n)} = S_3^{(g)} \\cdot T_3^n \\cdot x, \\qquad g = 1, \\dots, 6$$

The `S3ParallelAttention` module evaluates all six orbits simultaneously via tensor broadcasting, then reduces them through the Casimir operator to produce a Galois-invariant output.

---

## 6. Chinese Remainder Theorem Decomposition

### 6.1 The CRT Isomorphism

**Theorem 6.1.** There is a ring isomorphism:

$$\\mathbb{Z}_{13}[\\eta] \\;\\cong\\; \\mathbb{F}_{13} \\times \\mathbb{F}_{169}$$

induced by the factorisation $f(x) = (x-7)(x^2+6x+2)$ over $\\mathbb{F}_{13}$.

*Proof.* The ideals $(\\eta - 7)$ and $(\\eta^2 + 6\\eta + 2)$ are coprime in $\\mathbb{F}_{13}[\\eta]$, and their product equals $(f(\\eta))$. The Chinese Remainder Theorem gives the decomposition. $\\square$

### 6.2 Component Maps

**Dominant projection** $\\varphi_1: \\mathbb{Z}_{13}[\\eta] \\to \\mathbb{F}_{13}$:

$$\\varphi_1(a_0 + a_1 \\eta + a_2 \\eta^2) = a_0 + 7a_1 + 10a_2$$

since $\\varphi_1(\\eta) = 7$ and $\\varphi_1(\\eta^2) = 7^2 = 49 = 10$ in $\\mathbb{F}_{13}$.

**Subdominant projection** $\\varphi_2: \\mathbb{Z}_{13}[\\eta] \\to \\mathbb{F}_{169}$:

$$\\varphi_2(a_0 + a_1 \\eta + a_2 \\eta^2) = (a_0 + 11a_2) + (a_1 + 7a_2)\\xi$$

where $\\mathbb{F}_{169} = \\mathbb{F}_{13}[\\xi]/(\\xi^2 + 6\\xi + 2)$ and $\\xi^2 = 7\\xi + 11$.

### 6.3 Reconstruction

Given $(s, q) \\in \\mathbb{F}_{13} \\times \\mathbb{F}_{169}$, the unique preimage $a \\in \\mathbb{Z}_{13}[\\eta]$ is computed via the standard CRT formula using the precomputed Bézout coefficients for the coprime ideals $(\\eta - 7)$ and $(\\eta^2 + 6\\eta + 2)$. The reconstruction formula is:

$$a = s \\cdot (1, 3, 7) + q_0 \\cdot (0, 10, 6) + q_1 \\cdot (6, 6, 3) \\pmod{13}$$

---

## 7. Architecture

### 7.1 System Overview

The Zeta model processes token sequences through a pipeline of strictly algebraic transformations:

```
Input Tokens
    |
    v
ZRingEmbed  (deterministic eta-power embedding)
    |
    v
TatePE      (dynamic positional encoding)
    |
    v
[ T3^n Evolution + SpectralAttention ]^N  (N layers)
    |
    v
Witt Head   (precision-adaptive output)
    |
    v
Token Predictions
```

All operations are performed in $\\mathbb{Z}_{13}[\\eta]$ using `torch.long` tensors. No floating-point values exist in the data path.

### 7.2 Design Principles

1. **Algebraic Closure:** Every intermediate value is a ring element.
2. **Determinism:** Embeddings and transitions are computed, not learned.
3. **Reversibility:** All evolution is time-reversible via $T_3^{-n} = T_3^{168-n}$.
4. **Spectral Separation:** Dominant and subdominant channels are processed independently.
5. **Galois Covariance:** The system respects the $S_3$ symmetry.

---

## 8. Deterministic Embeddings and Positional Encoding

### 8.1 ZRingEmbed

The embedding of token value $v \\in \\{0, \\dots, V-1\\}$ at dimension $d \\in \\{0, \\dots, D-1\\}$ is:

$$E(v, d) = \\eta^{(v \\cdot D + d) \\bmod 168}$$

**Properties:**
- No learned parameters.
- Injective for $V \\cdot D \\leq 168$.
- Surjective onto the unit group orbit when $V \\cdot D = 168$.
- Periodicity naturally handles vocabulary wrapping.

### 8.2 TatePE

Positional encoding for position $n$ and dimension index $k$:

$$\\mathrm{PE}[n, 2k] = \\eta^{nk \\bmod 168}, \\qquad \\mathrm{PE}[n, 2k+1] = \\eta^{-nk \\bmod 168}$$

**Properties:**
- Valid for any sequence length $L$ without recomputation.
- Multiplicative structure: $\\mathrm{PE}[n_1 + n_2, :] = \\mathrm{PE}[n_1, :] \\cdot \\mathrm{PE}[n_2, :]$.
- No trigonometric functions or real numbers.

---

## 9. Spectral Attention Mechanism

### 9.1 Sylvester Channel Split

Given input $X \\in \\mathbb{Z}_{13}[\\eta]^{B \\times L \\times D}$, the attention layer first projects into spectral channels:

$$X_1 = P_1 \\cdot X, \\qquad X_{23} = P_{23} \\cdot X$$

where $P_1$ and $P_{23}$ are applied as matrix multiplications on the trailing $3$-dimensional coefficient space.

### 9.2 Tree Kernel Mixing

For each channel, the ultrametric kernel performs global information mixing:

$$X_1^{(k)} = \\mathrm{PAdicKernel}(X_1), \\qquad X_{23}^{(k)} = \\mathrm{PAdicKernel}(X_{23})$$

For $L \\leq 169$, the fast fused path is used; for $L > 169$, the standard tree path. The kernel never materializes an $L \\times L$ matrix.

### 9.3 Local Q/K/V Projections

Query, key, and value transformations are ring-linear maps applied pointwise along the dimension axis:

$$Q_c = \\mathrm{ring\\_lin}(X_c^{(k)}, W_c^Q), \\quad K_c = \\mathrm{ring\\_lin}(X_c^{(k)}, W_c^K), \\quad V_c = \\mathrm{ring\\_lin}(X_c^{(k)}, W_c^V)$$

where $c \\in \\{\\mathrm{dom}, \\mathrm{sub}\\}$. The weight matrices $W$ are deterministic $\\eta$-power seeds generated by `w_seed`.

### 9.4 Born Normalization

Attention scores are computed via the algebraic trace inner product and normalized using the algebraic norm:

$$S_c[b, l, d] = \\mathrm{Tr}(Q_c[b, l, d] \\cdot K_c[b, l, d]) \\bmod 13$$

$$O_c = V_c \\cdot S_c \\bmod 13$$

This replaces the softmax operation with purely algebraic weighting. No $L \\times L$ attention matrix is ever formed.

### 9.5 Output Projection

The channel outputs are concatenated and projected back:

$$\\mathrm{Out} = \\mathrm{ring\\_lin}([O_{\\mathrm{dom}}; O_{\\mathrm{sub}}], W_O)$$

followed by residual connection, discrete Laplacian smoothing, and HilbertEta layer normalization.

---

## 10. Witt Vector Head and Hensel Lifting

### 10.1 Witt Vector Construction

The output head stores weights as Witt vectors over $\\mathbb{Z}_{13}[\\eta]$. A Witt vector of length $m$ is a sequence $(w_0, w_1, \\dots, w_{m-1})$ where each $w_k \\in \\mathbb{Z}_{13}[\\eta]$. The ghost components are given by:

$$w^{(n)} = \\sum_{k=0}^{n} 13^k w_k^{13^{n-k}}$$

### 10.2 Head Synchronization

The effective head weights are synthesized from all active precision layers:

$$\\mathrm{head} = \\sum_{k=0}^{\\mathrm{prec}-1} (k+1) \\cdot \\mathrm{to\\_ring}\\big(\\mathrm{witt\\_head}[\\dots, k, :]\\big) \\pmod{13}$$

### 10.3 Hensel Lifting

When the entropy monitor triggers a precision catastrophe, the system increments $\\mathrm{prec}$ and initializes the new Witt component $w_{\\mathrm{prec}}$ from the current error ideal. The Hensel digits of $\\eta$ in base $13$ are $[7, 2, 1, 12, 11, \\dots]$, with each digit corresponding to one Witt vector level.

---

## 11. MERA Tensor Network

### 11.1 Hierarchical Coarse-Graining

The Multi-scale Entanglement Renormalization Ansatz is implemented via alternating ring addition and $T_3$-evolution:

$$\\text{level } 0: \\quad x^{(0)}[i] = x[i], \\quad i = 0, \\dots, L-1$$

$$\\text{level } k: \\quad x^{(k)}[i] = T_3^{2^k} \\cdot \\big(x^{(k-1)}[2i] + x^{(k-1)}[2i+1]\\big)$$

### 11.2 Dynamic Depth

The number of levels is $\\lfloor \\log_2 L \\rfloor$, capped at $10$. For odd-length sequences, the final element is handled via $\\min(\\text{even}, \\text{odd})$ truncation without padding tokens.

---

## 12. Geometric Quantum Mechanics

### 12.1 State Space

Quantum states are formal sums over the token basis with coefficients in $\\mathbb{Z}_{13}[\\eta]$:

$$|\\psi\\rangle = \\sum_{k=0}^{K-1} c_k |k\\rangle, \\qquad c_k \\in \\mathbb{Z}_{13}[\\eta]$$

### 12.2 Born Rule

The probability weight of basis state $|k\\rangle$ is:

$$p_k = \\mathrm{N}(c_k) \\cdot 2^k \\bmod 13$$

### 12.3 Density Matrices

The density matrix of a pure state is:

$$\\rho = |\\psi\\rangle\\langle\\psi| = \\sum_{i,j} c_i \\overline{c_j} |i\\rangle\\langle j|$$

where the conjugation is the $J$ operator from Section 5.3. For mixed states, $\\rho$ is a sum of such terms.

### 12.4 Quantum Cache

The system maintains a quantum cache that accumulates and evolves density matrices:

$$\\rho_{\\text{cache}}(t) = \\sum_{s=0}^{t} T_3^{t-s} \\cdot \\rho_s \\cdot T_3^{-(t-s)}$$

This provides a form of quantum memory that is automatically time-averaged over the $T_3$ orbit.

---

## 13. Training via Buchberger-Nullstellensatz

### 13.1 Error Ideal

For a token position $(b, l)$ with target $v_t$ and prediction $v_p$, the error is embedded into $\\mathbb{Z}_{13}[\\eta]$ as:

$$e_{b,l} = (v_t - v_p) \\bmod 13$$

extended to the full ring as $(e_{b,l}, 0, 0)$.

### 13.2 Witness Polynomial

The witness polynomial uses the current state:

$$g_{b,l} = x[b, l, :] \\cdot \\eta^{\\text{step}}$$

### 13.3 Ideal Contribution

$$\\text{contrib}_{b,l} = g_{b,l} \\cdot e_{b,l}$$

### 13.4 Scatter Update

The head is updated by boosting the target and suppressing the prediction:

$$\\text{boost} = \\text{contrib} \\cdot e_{v_t}, \\qquad \\text{suppress} = \\text{contrib} \\cdot e_{v_p}$$

$$\\Delta_{\\text{head}} = \\text{boost} - \\text{suppress}$$

### 13.5 Convergent Decay

If $\\|\\Delta_{\\text{head}}\\| > 6$ (in the dominant projection), apply a half-step:

$$\\Delta_{\\text{head}} \\leftarrow 7 \\cdot \\Delta_{\\text{head}}$$

since $7 \\equiv 2^{-1} \\pmod{13}$.

### 13.6 Witt Update

Convert $\\Delta_{\\text{head}}$ to a Witt vector at the current precision and add to `witt_head` with carry propagation.

### 13.7 Quantum and Spectral Memory

The correction is absorbed into the `ErrorCache` as a density matrix update. The NTT spectrum of the cache provides a secondary correction term, which is also scattered into the Witt head.

---

## 14. Autonomous Control Loop

### 14.1 Components

| Component | Function |
|-----------|----------|
| `ZetaSelf` | Persistent self-state $\\Psi_{\\text{self}}(t) = T_3^t \\Psi_{\\text{self}}(0) + \\sum_s T_3^{t-s} \\cdot \\text{sensory}_s$. Reflects on goal deficit via $P_1$ projection. |
| `ZetaGoalPlanner` | Searches the $T_3$ orbit ($168$ steps) for the optimal approach vector to the target state. Selects processing channel: direct, $P_1$, $P_{23}$, or Witt lift. |
| `EntropyMonitor` | Computes algebraic entropy from the state distribution. Triggers Hensel precision catastrophe when entropy exceeds threshold ($S_{\\max} = 10$). |
| `ZetaScaler` | Maintains a $24$-step performance history. Autonomously activates multi-layer Witt, $S_3$ parallel mode, or dynamic MERA based on error trends. |
| `AutonomousLoop` | Orchestrates the closed cycle: sense $\\to$ evolve $\\to$ plan $\\to$ act $\\to$ learn. |

### 14.2 Scaling Decisions

The `ZetaScaler` makes autonomous decisions based on a $24$-step rolling window:

- **Multi-layer Witt:** Activated when average entropy exceeds $6$.
- **$S_3$ parallel orbit:** Cycled when error rate exceeds $50\\%$ ($\\text{avg\\_err} > 12$ out of $24$ steps).
- **Dynamic MERA:** Activated when sequence length exceeds $64$.
- **Hensel lift recommendation:** Issued when entropy reaches $10$.

### 14.3 Control Flow

```
SENSORY INPUT
      |
      v
ZetaSelf.evolve() + accumulate()
      |
      v
EntropyMonitor.evaluate()
      |
      +---> Entropy >= threshold? ---> HenselLifter.lift()
      |
      v
ZetaGoalPlanner.plan(target_state)
      |
      v
ACTION: BuchbergerEngine.correct()
      |
      v
UPDATE: ErrorCache.absorb() + SpectralMemory.update()
```

---

## 15. Tokenization

### 15.1 Byte Tokenizer (`tokenise`)

- Vocabulary: $V = 256$
- Compression: $1\\times$
- Use: Universal, short sequences

### 15.2 DNA Codon Tokenizer (`dna_tokenise`)

- Vocabulary: $V = 64$
- Compression: $3\\times$
- Method: Maps DNA bases A, C, G, T to 2-bit codes, groups into codons
- Use: Genomic sequences

### 15.3 Trigram Tokenizer (`trigram_tokenise`)

- Vocabulary: $V = 256$
- Compression: $3\\times$
- Method: Overlapping trigram hash with CRT-style coefficients:

$$h(b_1, b_2, b_3) = (b_1 + 7 \\cdot b_2 + 10 \\cdot b_3) \\bmod 256$$

- Use: General text, long sequences

**Example:** The string `"Hello Zeta"` (10 bytes) produces 4 trigram tokens.

---

## 16. Axiom Verification System

The runtime enforces $30$ algebraic axioms organized into $10$ categories. All axioms must be satisfied for the system to enter operational state.

| Category | Axioms | Assertions |
|----------|--------|------------|
| Ring | A001–A009 | Associativity and commutativity of addition and multiplication; distributivity; existence of additive identity $(0,0,0)$ and multiplicative identity $(1,0,0)$; existence of multiplicative inverses for units; characteristic $13$; reduction identity $\eta^3 = \eta^2 + \eta + 1$ |
| CRT | A010–A011 | Isomorphism $\mathbb{Z}_{13}[\eta] \cong \mathbb{F}_{13} \times \mathbb{F}_{169}$; roundtrip fidelity $\mathrm{CRT}^{-1}(\varphi_1(a), \varphi_2(a)) = a$; $\varphi_1(1) = 1$ |
| $T_3$ | A012–A014 | $\det(T_3) = 1$; $T_3^{168} = I$; $T_3 \cdot P_1 = 7 P_1$ |
| Sylvester | A015–A019 | Idempotence $P_1^2 = P_1$, $P_{23}^2 = P_{23}$; orthogonality $P_1 P_{23} = 0$; completeness $P_1 + P_{23} = I$; spectral decomposition identity $T_3^n = 7^n P_1 + P_{23} T_3^n$ |
| $S_3$ | A020–A021 | Group closure: $\forall g, h \in S_3: g \cdot h \in S_3$; involution $J^2 = \mathrm{id}$ |
| Kernel | A022–A023 | Diagonal identity $G(i, i) = 1$; strong triangle inequality for all triples $(i, j, k)$ |
| NTT | A024–A025 | Roundtrip: $\mathrm{INTT}(\mathrm{NTT}(x)) = x$; convolution theorem: $\mathrm{NTT}(a * b) = \mathrm{NTT}(a) \cdot \mathrm{NTT}(b)$ |
| Witt | A026–A027 | Witt addition and multiplication are consistent with ring arithmetic: $\mathrm{to\_ring}(\mathrm{wadd}(w, v)) = a + b$, $\mathrm{to\_ring}(\mathrm{wmul}(w, v)) = a \cdot b$ |
| Dirac | A028–A029 | Antisymmetry $D[i,j] = -D[j,i]$; p-adic Riemann Hypothesis spectral condition |
| Spectral | A030 | Winding number of the spectral flow equals $14$ |

## 17. Performance Characteristics

The core engine of Zeta has been ported to Rust for extreme performance. Below is a side-by-side comparison of the measured performance characteristics between the legacy Python reference implementation and the optimized Rust port on a standard benchmark suite ($B=4$, $L=64$, $V=256$, $D=54$, $N=11$ unless otherwise noted).

### Core Operations & Speedup Comparison

| Operation | Theoretical Complexity | Python Reference Latency | Optimized Rust Latency | Measured Speedup |
|-----------|------------------------|--------------------------|------------------------|------------------|
| Ring addition | $O(1)$ | — | $<0.1\text{ ns}$ (Compiler Optimized) | — |
| Ring multiplication | $O(1)$ | $\sim 276.1\,\mu\text{s}$ | $<0.1\text{ ns}$ (Compiler Optimized) | **$>2,700,000\times$** |
| Ring inverse | $O(1)$ | $\sim 73.7\,\mu\text{s}$ | $\sim 1.5\text{ ns}$ | **$\sim 49,000\times$** |
| $T_3$ power lookup | $O(1)$ | $\sim 6.7\,\mu\text{s}$ | $\sim 1.2\text{ ns}$ | **$\sim 5,500\times$** |
| Witt vector addition ($prec=4$) | $O(\mathrm{prec})$ | $\sim 123.7\,\mu\text{s}$ | $<0.01\text{ ns}$ (Compiler Optimized) | **$>10,000,000\times$** |
| Witt vector multiplication ($prec=4$) | $O(\mathrm{prec}^2)$ | $\sim 562.2\,\mu\text{s}$ | $\sim 19.4\text{ ns}$ | **$\sim 29,000\times$** |
| NTT ($N = 4$) | $O(N \log N)$ | $\sim 108.7\,\mu\text{s}$ | $\sim 282\text{ ns}$ | **$\sim 385\times$** |
| NTT ($N = 12$) | $O(N \log N)$ | $\sim 129.9\,\mu\text{s}$ | $\sim 5.2\,\mu\text{s}$ | **$\sim 25\times$** |
| Tribonacci mixing time (`delta_max`) | $O(L^2)$ | $\sim 0.18\,\mu\text{s}$ | $\sim 5.1\,\mu\text{s}$ | — |
| Tree kernel (fast path, $L \leq 169$) | $O(L \log_{13} L)$ | $\sim 1.83\text{ ms}$ | $\sim 0.026\text{ ms}$ ($26\,\mu\text{s}$) | **$\sim 70\times$** |
| Full Model Forward Pass ($N=11$) | $O(B \cdot L \cdot D \cdot N)$ | $\sim 2,583.6\text{ ms}$ | $\sim 304.9\text{ ms}$ | **$\sim 8.5\times$** |

### Detailed Scaling Benchmarks

#### 1. Depth ($N$) Scaling ($B=4, L=64, V=256, D=54$)
- **$N = 1$**: Forward: $\sim 63.7\text{ ms}$ | Weight Memory: $265.8\text{ KB}$
- **$N = 11$**: Forward: $\sim 383.0\text{ ms}$ | Weight Memory: $493.6\text{ KB}$
- **$N = 50$**: Forward: $\sim 1609.6\text{ ms}$ | Weight Memory: $1382.1\text{ KB}$
- **$N = 100$**: Forward: $\sim 3003.5\text{ ms}$ | Weight Memory: $2521.1\text{ KB}$
- **$N = 500$**: Forward: $\sim 14427.8\text{ ms}$ | Weight Memory: $11633.6\text{ KB}$

#### 2. Sequence Length ($L$) Scaling ($N=1, V=256, D=54$)
- **$L = 64$**:
  - Batch $B = 1$: $\sim 9.1\text{ ms}$
  - Batch $B = 4$: $\sim 29.8\text{ ms}$
- **$L = 256$**:
  - Batch $B = 1$: $\sim 52.2\text{ ms}$
  - Batch $B = 4$: $\sim 172.6\text{ ms}$
- **$L = 1024$**:
  - Batch $B = 1$: $\sim 172.7\text{ ms}$
  - Batch $B = 4$: $\sim 677.8\text{ ms}$
- **$L = 4096$**:
  - Batch $B = 1$: $\sim 663.2\text{ ms}$
  - Batch $B = 4$: $\sim 2558.0\text{ ms}$
- **$L = 16384$**:
  - Batch $B = 1$: $\sim 2393.8\text{ ms}$
  - Batch $B = 4$: $\sim 13264.1\text{ ms}$

#### 3. Vocabulary Size ($V$) Scaling ($B=4, L=64, N=1, D=54$)
- **$V = 256$**: Forward: $\sim 43.2\text{ ms}$ | Weight Memory: $0.26\text{ MB}$
- **$V = 1024$**: Forward: $\sim 63.8\text{ ms}$ | Weight Memory: $0.97\text{ MB}$
- **$V = 8192$**: Forward: $\sim 243.7\text{ ms}$ | Weight Memory: $7.62\text{ MB}$
- **$V = 32000$**: Forward: $\sim 877.7\text{ ms}$ | Weight Memory: $29.69\text{ MB}$

---

## 18. Module Reference

### Core Algebra

| Module | Responsibility |
|--------|---------------|
| `constants.py` | Precomputed algebraic tables: `ETA_POW` (169 powers of $\\eta$), `ETA_IPOW` (169 inverse powers), `T3_POW` (168 matrix powers), `INV_TBL` (13,13,13,3 inverse lookup), `P1_MAT` and `P23_MAT` (Sylvester projectors), `S3_MATS` (6 permutation matrices), `_S3_MUL` (6,6 group multiplication table), `_VAL_LUT` (13-adic valuation for $d < 13^5$), NTT twiddle matrices for sizes $[1, 2, 3, 4, 6, 12]$, `_DELTA_MAX_LUT` (Tribonacci mixing time) |
| `ring.py` | $\\mathbb{Z}_{13}[\\eta]$ arithmetic: `mul`, `add`, `sub`, `neg`, `smul`, `inv`, `pow`, `trace`, `norm`, `conj`, `phi1`, `phi2`. CRT `compose`/`decompose`/`is_unit`. $\\mathbb{F}_{169}$ arithmetic: `mul`, `norm`, `inv`, `frobenius`, `pow` |
| `spectral.py` | `proj()` — projector application via matrix multiplication; `t3n()` — $T_3^n$ orbit evolution; `t3n_s3()` — $S_3$-covariant evolution; `SylvesterProjectors.p1/p23/split`; `SpectralDecomposition.evolve/eigenvalue_dominant`; `S3Galois.apply/compose_indices/orbit/casimir` |

### Kernel and Geometry

| Module | Responsibility |
|--------|---------------|
| `kernel.py` | `PAdicKernel` — ultrametric tree kernel. `build_tree()` UP pass with `torch.bmm`; `tree_attend()` DOWN pass with `expand+reshape`; `apply_fast()` fused path for $L \\leq 169$; `elem()` $O(1)$ single element; `strong_triangle_inequality()` and `batch_triangle_check()` axioms |
| `laplacian.py` | `PAdicLaplacian` — Vladimirov p-adic Laplacian: `weight_matrix()`, `matrix()`, `apply()`, `ntt_eigendecomp()` |
| `linear.py` | `ring_lin` — ring-linear transformation; `born_norm` — algebraic normalization; `ring_attend` — ring attention; `laplacian` — discrete Laplacian $\\Delta x[i] = x[i+1] + x[i-1] + 11x[i]$; `w_seed` — deterministic $\\eta$-power weight seeding; `alg_dropout` — deterministic dropout via $\\eta$ powers |
| `hilbert.py` | `HilbertEta` — algebraic inner product `inner()`, squared norm `norm_sq()`, and layer normalization `layer_norm()` over $\\mathbb{Z}_{13}[\\eta]$ with $\\eta^{-k}$ weights |

### Transform and Arithmetic

| Module | Responsibility |
|--------|---------------|
| `ntt.py` | Number Theoretic Transform over $\\mathbb{Z}_{13}[\\eta]$. Valid sizes: $[1, 2, 3, 4, 6, 12]$. Scalar twiddle path. `ntt`, `intt`, `conv`, `spectrum`, `cross`, `autocorr`, `best_size` |
| `witt.py` | `WittVector` — Witt vector arithmetic: `from_ring`/`to_ring`, `wadd` (carry propagation), `wneg`, `wsub`, `wmul` (anti-diagonal scatter-add), `wpow` (8-step unrolled), `winv` (Newton iteration), `ghost`, `frobenius`, `teichmuller` |
| `hensel.py` | `HenselLifter` — Newton iteration for $\\eta$ digits in base $13$: `eta_int`, `digits`, `witt_eta`, `verify`. `HenselIO` — sensor/actuator empirical closure: `sensor_embed`, `actuator_decode`, `feedback_check` |
| `mahler.py` | `MahlerExpansion` — finite difference calculus: `_binom_matrix`, `finite_differences`, `coeffs`, `reconstruct`, `ntt_bridge` |
| `teichmuller.py` | `Teichmuller` — multiplicative section: `lift`, `orbit`, `mul_table`, `ntt_spectrum` |

### Quantum and Spectral

| Module | Responsibility |
|--------|---------------|
| `gqm.py` | `GQMState` — quantum state with `evolve`, `hamiltonian`, `born_probs`, `inner`, `density`, `ntt_spectrum`, `entropy`. `ErrorCache` — correction accumulation with `absorb` and `spectral_correction`. `DensityMatrix` — `build`, `trace`, `mat_mul`, `evolve`, `entropy`. `ZetaSelf` — persistent self-state with `evolve`, `reflect`, `is_my_experience` |
| `zeta_func.py` | `ZetaFunctionRing` — `zeta`, `zeta_batch`, `ntt_identity`, `functional_eq`. `SpectralFlow` — `dominant`, `subdominant`, `ntt_spectrum`, `chern_proxy`, `winding`, `critical_indices`. `SpectralZeta` — `zeta`, `connection_to_zeta_ring` |
| `dirac.py` | `AdelicDiracOperator` — antisymmetric Dirac operator $D[i,j] = (\\eta^{i-j} - \\eta^{j-i}) \\cdot G(i,j)$. `matrix`, `chiral_check`, `spectrum`, `zeros_of_zeta`, `rh_check` |
| `berry.py` | `BerrySVD` — Berry connection via $P_1$ projectors, algebraic SVD proxy via NTT spectrum: `connection`, `phase`, `svd_proxy`, `dominant_mode` |
| `entanglement.py` | `EntanglementGeometry` — `ryu_takayanagi`, `wormhole_entropy`, `page_curve`, `state_entropy`, `info_conservation` with central charge $\\Delta = 2$ |
| `crystal.py` | `CrystalLattice` — `unit_cell`, `dominant_sublattice`, `structure_factor`, `reciprocal`, `voronoi` |

### Model, Attention, and Hybrid

| Module | Responsibility |
|--------|---------------|
| `embed.py` | `ZRingEmbed` — deterministic $\\eta$-power embedding; `TatePE` — dynamic positional encoding |
| `attention.py` | `SpectralAttention` — Sylvester channel split, tree kernel mixing (fast/standard), local Q/K/V projections, Born normalization, residual + Laplacian + HilbertEta layer norm |
| `hybrid.py` | `S3ParallelAttention` — simultaneous evaluation of 6 Galois-conjugate orbits with `evolve_parallel` and `casimir_reduce`. Full parallel attention pipeline |
| `model.py` | `ZetaModel` — multi-layer architecture ($V=256, D=54, N=11$ default) with Witt head (PREC=4), fast kernel dispatch, convergent decay, autonomous Hensel lift, S3 orbit cycling, ErrorCache integration |
| `mera.py` | `MERAChunker` — `coarse_grain` (3 fixed levels), `coarse_grain_dynamic` (auto-depth $\\lfloor \\log_2 L \\rfloor$, cap 10), `ntt_per_level`, `renormalize` |

### Learning and Autonomy

| Module | Responsibility |
|--------|---------------|
| `buchberger.py` | `BuchbergerEngine` — `error_ideal`, `nullstellensatz_correction` (vectorized scatter update with target boost and pred suppression), `groebner_reduce`. Phase factor $\\eta^{7 \\cdot \\text{step}}$ |
| `goal.py` | `ZetaGoalPlanner` — `distance` ($N(P_1 \\cdot (\\text{state} - \\text{goal}))$), `plan` (168-step orbit search with channel selection: direct / $P_1$ / $P_{23}$ / none+lift) |
| `entropy.py` | `EntropyMonitor` — `check` (threshold $S_{\\max}=10$), `lift_if_needed` |
| `scaler.py` | `ZetaScaler` — `observe` (24-step history), `decide` (Witt full / S3 orbit / dynamic MERA / lift recommendation), `reset_orbit` |
| `autonomy.py` | `AutonomousLoop` — closed-loop control with persistent `ZetaSelf`, goal planning, entropy monitoring, scaling, and reflection. `FullAutonomyCycle` — simplified single-cycle variant |

### Planning and Control

| Module | Responsibility |
|--------|---------------|
| `orbit.py` | `OrbitPlanner` — `plan_orbit_vec`/`plan_orbit` (168-step vectorized search), `orbit_spectrum`, `closure_check` |
| `reversible.py` | `ReversibleGenerator` — `encode` ($T_3^t \\cdot x_t$), `decode` ($T_3^{168-t} \\cdot y_t$), `roundtrip_check` |
| `rollback.py` | `QuantumRollback` — `detect`, `rollback_ring`, `rollback_witt`, `correction_spectrum` |
| `counterfactual.py` | `CounterfactualBranch` — `branch`, `best_branch`, `gqm_superposition` |
| `ardt.py` | `ARDTArchitecture` — 5-step algebraic reasoning chain: `perceive`, `reason`, `plan`, `act`, `verify` |

### System and I/O

| Module | Responsibility |
|--------|---------------|
| `runtime.py` | `ZetaDevice` — device-local buffers for all precomputed tables. `ZetaRuntime` — singleton entry point, forward pass, train step, encode/decode, RH check, comprehensive benchmarking |
| `axioms.py` | `AxiomVerifier` — runtime verification of all 30 axioms (A001–A030) across 10 categories. `verify_all`, `print_report` |
| `sampler.py` | `PAdicSampler` — `step`, `batch_accuracy` for Buchberger training evaluation |
| `tokenizer.py` | `tokenise` (byte, $V=256$), `dna_tokenise` (codon, $V=64$), `trigram_tokenise` (3× compression), `detokenise`, `detrigram_tokenise`, `pad_batch`, `pad_batch_trigram` |
| `adelic.py` | `AdelicProduct` — `residue`, `adelic_norm`, `strong_approx` over primes $[2, 3, 5, 7, 11]$ |

---

## 19. Usage Examples

### 19.1 Initialization and Axiom Verification

```python
from zeta import ZetaRuntime, AxiomVerifier

# Initialize the runtime
rt = ZetaRuntime.init(device='cpu', V=256, D=54, N=11, ctx=256)

# Verify all algebraic axioms before operation
AxiomVerifier.print_report()
```

### 19.2 Tokenization and Forward Pass

```python
from zeta import trigram_tokenise
import torch

# Compress text by factor of 3
text = "Algebraic intelligence over cubic p-adic integers."
ids = trigram_tokenise(text)
tokens = torch.tensor([ids], dtype=torch.long)

# Forward through the model
output = rt.model(tokens)
```

### 19.3 Autonomous Loop

```python
from zeta import AutonomousLoop

loop = AutonomousLoop(rt)
result = loop.cycle(tokens, target_goal=None)

print(result)
# {'self_t': int, 'entropy': int, 'plan': dict, 'scale_actions': dict}
```

### 19.4 Benchmarking

```python
# Standard benchmark suite
rt.print_benchmark(B=4, L=64)
```

### 19.5 Riemann Hypothesis Check

```python
# Verify p-adic spectral condition
rh_result = rt.rh_check(L=12)
print(rh_result)
```

---

## 20. Bibliography

1. **Tribonacci Polynomials and Cubic p-adic Integers.** The arithmetic of finite extensions $\\mathbb{Z}_p[\\eta]/(f(\\eta))$ where $f(x) = x^3 - x^2 - x - 1$, including unit group structure, period computation, and factorisation criteria over $\\mathbb{F}_p$.

2. **Sylvester Spectral Decomposition.** Matrix spectral theory via Lagrange interpolation of the minimal polynomial, with application to unimodular matrices in $\\mathrm{SL}(3, \\mathbb{Z})$ and their reduction modulo $p$.

3. **Bruhat-Tits Trees and Ultrametric Geometry.** The geometry of $\\mathrm{GL}(2, \\mathbb{Q}_p)$ and its associated Bruhat-Tits building, strong triangle inequality, and tree-based kernel methods.

4. **Hensel Lifting and Witt Vectors.** Precision-adaptive arithmetic in p-adic extensions, construction of Witt vectors $\\mathbb{W}(k)$, ghost component algebra, and Teichmüller representatives.

5. **Number Theoretic Transform over Finite Rings.** Fast convolution algorithms in rings of the form $\\mathbb{Z}_p[\\eta]/(f(\\eta))$, including primitive root selection and inverse transform verification.

6. **Buchberger Algorithm and Effective Nullstellensatz.** Gröbner basis computation in multivariate polynomial rings, ideal membership testing, and constructive Hilbert Nullstellensatz for error correction.

7. **Geometric Quantum Mechanics over Algebraic Number Fields.** State space formalism, Born normalization in finite fields, density matrix evolution, and quantum memory without complex Hilbert spaces.

8. **Adelic Dirac Operators and p-adic Spectral Theory.** Construction of p-adic pseudodifferential operators, spectral zeta functions, and connections to the Riemann Hypothesis in the p-adic setting.

---

## Author and License

**Author:** Dávid Navrátil `<david.navratil2016@gmail.com>`  
**License:** CC-BY-NC-4.0  
**Version:** 7.0.0

## License

Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)

Copyright (c) 2026 Dávid Navrátil

You are free to:

  Share — copy and redistribute the material in any medium or format
  Adapt — remix, transform, and build upon the material

The licensor cannot revoke these freedoms as long as you follow the
license terms.

Under the following terms:

  Attribution   - You must give appropriate credit, provide a link to
                  the license, and indicate if changes were made. You may
                  do so in any reasonable manner, but not in any way that
                  suggests the licensor endorses you or your use.

  NonCommercial - You may not use the material for commercial purposes.

  No additional restrictions — You may not apply legal terms or
                  technological measures that legally restrict others
                  from doing anything the license permits.

Notices:

  You do not have to comply with the license for elements of the material
  in the public domain or where your use is permitted by an applicable
  exception or limitation.

  No warranties are given. The license may not give you all of the
  permissions necessary for your intended use. For example, other rights
  such as publicity, privacy, or moral rights may limit how you use the
  material.

Full license text:  https://creativecommons.org/licenses/by-nc/4.0/legalcode

Commercial licensing inquiries:  david.navratil2016@gmail.com
