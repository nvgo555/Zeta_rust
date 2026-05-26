# Zeta p-Adic Integer AI — Module Reference Document v7.0.0

**Complete Algebraic Specification of All 39 System Modules**

---

## Document Purpose

This document provides a rigorous, line-by-line description of every module in the Zeta p-Adic Integer AI system. All descriptions are derived directly from the source code files; no external assumptions, no hallucinated functionality, and no ad hoc additions are included. Each module entry contains: (1) the exact mathematical role, (2) all public classes and functions with their signatures, (3) tensor shapes and data types, (4) dependencies on other Zeta modules, and (5) the precise connection to the overall pipeline.

---

## Table of Modules

### Tier 1: Foundation (Constants, Ring, Spectral)
| Module | File | Role |
|--------|------|------|
| constants | `constants.py` | Immutable precomputed algebraic tables |
| ring | `ring.py` | All arithmetic in Z_13[eta] and F_169 |
| spectral | `spectral.py` | T3 orbit, Sylvester projectors, S3 Galois |

### Tier 2: Geometry and Kernel
| Module | File | Role |
|--------|------|------|
| kernel | `kernel.py` | Ultrametric Bruhat-Tits tree kernel |
| laplacian | `laplacian.py` | Vladimirov p-adic Laplacian |
| linear | `linear.py` | Shared ring-linear operations |
| hilbert | `hilbert.py` | Hilbert-eta inner product and layer norm |

### Tier 3: Transforms and Arithmetic
| Module | File | Role |
|--------|------|------|
| ntt | `ntt.py` | Number Theoretic Transform over Z_13[eta] |
| witt | `witt.py` | Witt vector arithmetic |
| hensel | `hensel.py` | Hensel lifting and empirical I/O |
| mahler | `mahler.py` | Mahler expansion and finite differences |
| teichmuller | `teichmuller.py` | Teichmüller lifts |

### Tier 4: Quantum and Spectral Physics
| Module | File | Role |
|--------|------|------|
| gqm | `gqm.py` | Geometric Quantum Mechanics states and memory |
| zeta_func | `zeta_func.py` | Spectral zeta functions and eigenvalue flows |
| dirac | `dirac.py` | Adelic Dirac operator and p-adic RH |
| berry | `berry.py` | Berry connection and algebraic SVD proxy |
| entanglement | `entanglement.py` | Holographic entropy formulas |
| crystal | `crystal.py` | Crystal lattice structure factor |

### Tier 5: Model, Attention, Hybrid
| Module | File | Role |
|--------|------|------|
| embed | `embed.py` | Deterministic token and positional embeddings |
| attention | `attention.py` | Spectral attention with tree kernel |
| hybrid | `hybrid.py` | S3 parallel conjugate orbit attention |
| model | `model.py` | Complete Zeta language model |
| mera | `mera.py` | MERA tensor network coarse-graining |

### Tier 6: Learning and Autonomy
| Module | File | Role |
|--------|------|------|
| buchberger | `buchberger.py` | Buchberger-Nullstellensatz training engine |
| goal | `goal.py` | T3 orbit goal planning |
| entropy | `entropy.py` | Entropy monitoring and Hensel trigger |
| scaler | `scaler.py` | Autonomous scaling decisions |
| autonomy | `autonomy.py` | Closed-loop autonomous control |

### Tier 7: Planning and Control
| Module | File | Role |
|--------|------|------|
| orbit | `orbit.py` | T3 orbit trajectory planner |
| reversible | `reversible.py` | Time-reversible encode/decode |
| rollback | `rollback.py` | Quantum error detection and correction |
| counterfactual | `counterfactual.py` | Parallel trajectory exploration |
| ardt | `ardt.py` | 5-step algebraic reasoning chain |

### Tier 8: System and I/O
| Module | File | Role |
|--------|------|------|
| runtime | `runtime.py` | Device manager and system runtime |
| axioms | `axioms.py` | Runtime axiom verification (A001–A030) |
| sampler | `sampler.py` | Buchberger batch accuracy estimation |
| tokenizer | `tokenizer.py` | Byte, DNA codon, and trigram tokenizers |
| adelic | `adelic.py` | Adelic norm and strong approximation |
| __init__ | `__init__.py` | Package exports and version |

---


## Tier 1: Foundation

---

### 1.1 `constants.py` — Global Algebraic Tables and Configuration

**File:** `zeta/constants.py`

**Purpose.** This module is the single source of truth for all precomputed immutable tables. It is imported by every other module. All tables are built once at import time via pure-Python constructors and stored as `torch.Tensor` objects. No table is ever rebuilt at runtime.

**Configuration Class.**

```python
class ZetaConfig:
    P: int = 13              # ring characteristic
    ORD: int = 168           # multiplicative order of eta
    I_P: int = 5             # sqrt(-1) mod 13 (5^2 = 25 = -1 mod 13)
    ETA0: int = 7            # eta mod 13 (Hensel level-0 digit)
    CTX: int = 256           # maximum context length
    D: int = 54              # model dimension (must divide by 6)
    N_LAYERS: int = 11       # number of SpectralAttention layers
    V: int = 256             # vocabulary size
    PREC: int = 4            # Witt vector precision (max 8)
    DROP: int = 2            # algebraic dropout rate
    DENSE_MAX: int = 169     # fast kernel threshold
    MAX_VAL_DEPTH: int = 5   # valuation LUT covers d < 13^5
    TRIB_MAX_L: int = 512    # delta_max LUT precomputed up to L=512
```

**Global Tensor Tables.**

| Symbol | Shape | Content | Builder |
|--------|-------|---------|---------|
| `ETA_POW` | (169, 3) | eta^k for k = 0..168 | `_build_eta_powers()` |
| `ETA_IPOW` | (169, 3) | eta^{-k} = eta^{168-k} | `_build_eta_powers()` |
| `T3_POW` | (168, 3, 3) | T3^k for k = 0..167 | `_build_t3_powers()` |
| `INV_TBL` | (13, 13, 13, 3) | O(1) ring inverse lookup | `_build_inv_table()` |
| `ONE_R` | (3,) | multiplicative identity (1,0,0) | static |
| `ZERO_R` | (3,) | additive identity (0,0,0) | static |
| `T3_MAT` | (3, 3) | base matrix T3 | static |
| `P1_MAT` | (3, 3) | Sylvester projector P1 | static |
| `P23_MAT` | (3, 3) | Sylvester projector P23 | static |
| `V_EIG` | (3,) | right eigenvector (1,3,7) | static |
| `W_EIG` | (3,) | left eigenvector (1,7,10) | static |
| `S3_MATS` | (6, 3, 3) | S3 permutation matrices | static |
| `_S3_MUL` | (6, 6) | S3 group multiplication table | static |
| `_VAL_LUT` | (13^5,) | v_13(d) for d < 371293 | `_build_val_lut()` |
| `_NTT_FWD` | dict[int, Tensor] | NTT forward twiddle matrices | `_build_ntt()` |
| `_NTT_INV` | dict[int, Tensor] | NTT inverse twiddle matrices | `_build_ntt()` |
| `_DELTA_MAX_LUT` | list[int] | Tribonacci mixing time t*(L) | `_build_delta_max_lut()` |

**Key Functions.**

- `_ring_mul_py(a, b) -> tuple` — Pure-Python ring multiplication used only during table construction. Computes polynomial product with reduction eta^3 = eta^2 + eta + 1 and eta^4 = 2*eta^2 + 2*eta + 1.
- `_mat_mul_py(A, B) -> list` — Pure-Python 3x3 matrix multiplication mod 13.
- `_build_inv_table() -> Tensor` — Builds the (13,13,13,3) inverse table. For each element (a0,a1,a2), checks if it is a unit via CRT (phi1 != 0 AND phi2 != (0,0)). Non-units map to (0,0,0). Units are inverted by binary exponentiation to power 167.
- `_build_ntt(N) -> None` — Precomputes NTT twiddle matrices for valid sizes [1,2,3,4,6,12]. Scalar twiddles use primitive roots; ring twiddles use ETA_POW/ETA_IPOW.
- `delta_max(L) -> int` — O(1) Tribonacci mixing time lookup. For L > 512, computes on-the-fly via Tribonacci iteration.

**Dependencies.** None (foundation module). Imports only `torch`.

**Connection to Pipeline.** Every other module imports `constants` to access the precomputed tables. This is the algebraic substrate of the entire system.

---

### 1.2 `ring.py` — Ring Arithmetic, CRT, and F_169 Field Operations

**File:** `zeta/ring.py`

**Purpose.** Implements all arithmetic in Z_13[eta] / (eta^3 - eta^2 - eta - 1), including the CRT decomposition into F_13 x F_169 and the quadratic extension F_169 = F_13[alpha] / (alpha^2 + 6*alpha + 2). Every operation is fully vectorized via PyTorch tensor operations; no Python loops iterate over data elements.

**Class `Z13` (static methods).**

| Method | Signature | Operation | Complexity |
|--------|-----------|-----------|----------|
| `mul` | `(a, b) -> Tensor` | Ring multiplication with mod 13 | O(1) per element |
| `mul_exact` | `(a, b) -> Tensor` | Ring multiplication without final mod (values <= 864) | O(1) |
| `add` | `(a, b) -> Tensor` | Component-wise addition mod 13 | O(1) |
| `sub` | `(a, b) -> Tensor` | Component-wise subtraction mod 13 | O(1) |
| `neg` | `(a) -> Tensor` | Additive inverse | O(1) |
| `smul` | `(a, k) -> Tensor` | Scalar multiplication by integer k mod 13 | O(1) |
| `inv` | `(a) -> Tensor` | Multiplicative inverse via INV_TBL lookup | O(1) |
| `pow` | `(a, n) -> Tensor` | Binary exponentiation (8 unrolled steps) | O(1) |
| `trace` | `(a) -> Tensor` | Trace: 3*a0 + a1 + 3*a2 mod 13 | O(1) |
| `norm` | `(a) -> Tensor` | Norm as determinant of multiplication matrix | O(1) |
| `conj` | `(a) -> Tensor` | Galois conjugation J = CRT^{-1}(phi1, Frob(phi2)) | O(1) |
| `phi1` | `(a) -> Tensor` | CRT projection to F_13: a0 + 7*a1 + 10*a2 | O(1) |
| `phi2` | `(a) -> Tensor` | CRT projection to F_169: (a0+11*a2, a1+7*a2) | O(1) |

**Multiplication Formula (verified symbolically):**

```
c0 = a0*b0 + a1*b2 + a2*b1 + a2*b2
c1 = a0*b1 + a1*b0 + a1*b2 + a2*b1 + 2*a2*b2
c2 = a0*b2 + a1*b1 + a2*b0 + a1*b2 + a2*b1 + 2*a2*b2
```

**Class `CRT` (static methods).**
- `phi1`, `phi2`, `galois_conj` — Delegates to `Z13`.
- `compose(alpha, beta) -> Tensor` — CRT reconstruction. alpha: scalar in F_13, beta: (...,2) in F_169. Returns (...,3) in Z_13[eta]. Uses basis vectors e1=(1,3,7), e2=(0,10,6), ee2=(6,6,3).
- `decompose(a) -> Tuple` — Returns (phi1(a), phi2(a)).
- `is_unit(a) -> Tensor` — Boolean mask: phi1 != 0 AND phi2 != (0,0).

**Class `F169` (static methods).**
- `mul(a, b) -> Tensor` — Multiplication in F_169 with alpha^2 = 7*alpha + 11.
- `norm(a) -> Tensor` — Norm in F_169.
- `inv(a) -> Tensor` — Inverse via norm and Ni = N^{P-2} mod 13.
- `frobenius(a) -> Tensor` — Frobenius automorphism: (a0 + a1*alpha) -> (a0 + a1*alpha_bar).
- `pow(a, n) -> Tensor` — Binary exponentiation (8 unrolled steps).

**Dependencies.** `constants` (ETA_POW, INV_TBL, P, P1_MAT, etc.).

**Connection to Pipeline.** All tensor arithmetic in the system flows through `Z13.mul`, `Z13.add`, and `Z13.inv`. The CRT decomposition enables the spectral split into dominant (F_13) and subdominant (F_169) channels.

---

### 1.3 `spectral.py` — T3 Orbit, Sylvester Projectors, and S3 Galois Symmetry

**File:** `zeta/spectral.py`

**Purpose.** Contains the spectral decomposition of the sole evolution matrix T3 in SL(3,Z), the Sylvester projectors P1 and P23, and the S3 Galois group action. All operations are O(1) table lookups or matrix multiplications.

**Function `proj(x, M) -> Tensor`.**
- Applies a 3x3 Z_13-linear map M to the trailing dimension of x.
- Computes M @ x via explicit component-wise formula: (x0*M[i,0] + x1*M[i,1] + x2*M[i,2]) mod 13 for i=0,1,2.
- This is the ONLY matrix application function in the system; all projectors and evolution operators use it.

**Function `t3n(x, n) -> Tensor`.**
- T3^n * x via O(1) table lookup: `proj(x, T3_POW[n % 168])`.
- x shape: (..., 3). Output: (..., 3).

**Function `t3n_s3(x, n, g) -> Tensor`.**
- T3^n conjugated by S3 element g: `S3Galois.apply(t3n(x, n), g)`.
- Provides 6 independent evolution paths (one per Galois element).

**Class `SylvesterProjectors` (static methods).**
- `p1(x)` — Projects onto dominant 1D channel via P1_MAT.
- `p23(x)` — Projects onto subdominant 2D channel via P23_MAT.
- `split(x)` — Returns (p1(x), p23(x)) where p23(x) = (x - p1(x)) mod 13.

**Class `SpectralDecomposition` (static methods).**
- `evolve(x, n)` — Computes 7^n * P1(x) + P23(T3^n * x) via `t3n` and `Z13.smul`.
- `eigenvalue_dominant(n)` — Returns 7^n mod 13.

**Class `S3Galois` (static methods).**
- `MATS` — Shape (6, 3, 3), the 6 permutation matrices.
- `NAMES` — ['e', '(12)', '(01)', '(02)', '(012)', '(021)'].
- `_MUL` — Shape (6, 6), precomputed group multiplication table.
- `apply(x, g)` — Applies S3 matrix g to x via `proj`.
- `compose_indices(g, h)` — O(1) group composition via lookup table.
- `orbit(x)` — All 6 Galois images. x:(3,) -> (6, 3).
- `casimir(x)` — Sum over S3 images: sum_g g*x mod 13. Returns S3-invariant projection.

**Dependencies.** `constants` (T3_POW, P1_MAT, P23_MAT, S3_MATS, _S3_MUL), `ring` (Z13).

**Connection to Pipeline.** `proj` and `t3n` are called in every layer of the model. `SylvesterProjectors.split` is the first operation in `SpectralAttention.forward`. `S3Galois` is used by `S3ParallelAttention` and `hybrid.py`.

---


## Tier 2: Geometry and Kernel

---

### 2.1 `kernel.py` — The Ultrametric Kernel via Bruhat-Tits Tree

**File:** `zeta/kernel.py`

**Purpose.** Implements the sole token interaction kernel of the system: the ultrametric tree kernel derived from the 13-adic valuation. The kernel NEVER materializes an L x L matrix. It operates via hierarchical tree aggregation with O(L * log_13(L)) time and O(L) memory.

**Class `PAdicKernel` (static methods).**

| Method | Signature | Description |
|--------|-----------|-------------|
| `n_levels` | `(L) -> int` | Tree depth: ceil(log_13(L)) |
| `_val` | `(dist) -> Tensor` | O(1) valuation lookup via `_VAL_LUT` |
| `build_tree` | `(x) -> List[Tensor]` | UP pass: groups by 13, sums, evolves by T3^{13^h} |
| `tree_attend` | `(levels, x) -> Tensor` | DOWN pass: broadcasts coarse to fine, P23 projection |
| `apply` | `(x) -> Tensor` | Public: build_tree + tree_attend |
| `apply_fast` | `(x) -> Tensor` | Fused path for L <= 169 (2 levels max), no list allocation |
| `elem` | `(i, j, dev) -> Tensor` | Single kernel element G(i,j) via O(1) lookup |
| `strong_triangle_inequality` | `(i, j, k) -> bool` | Axiom check for one triple |
| `batch_triangle_check` | `(L, n_samples) -> bool` | Random sampling axiom verification |

**UP Pass (`build_tree`).**
1. Pad sequence length to multiple of 13.
2. Reshape into groups of 13, sum along group axis.
3. Evolve by T3^{13^h} using `torch.bmm` for memory alignment.
4. Project onto P1 channel via `proj`.
5. Repeat until length < 13.

**DOWN Pass (`tree_attend`).**
1. For each coarse level h, broadcast coarse nodes back to fine resolution via `expand` (not `repeat_interleave`, saving memory copy).
2. Apply inverse T3 evolution: T3^{168 - 13^h}.
3. Weight by ETA_IPOW[h].
4. Project onto P23 channel via P23_MAT.
5. Accumulate into output.

**Fast Path (`apply_fast`).**
- For L <= 169, fuses UP and DOWN into a single function with no intermediate `List[Tensor]` allocation.
- Uses direct reshape + matmul chains.
- ~30% faster than generic `apply()` for short sequences.

**Dependencies.** `constants` (P, ORD, ETA_POW, ETA_IPOW, T3_POW, P1_MAT, P23_MAT, _VAL_LUT), `ring` (Z13), `spectral` (proj).

**Connection to Pipeline.** Called by `SpectralAttention.forward` for every layer. The fast path is used when L <= 169; standard path when L > 169.

---

### 2.2 `laplacian.py` — Vladimirov p-adic Laplacian

**File:** `zeta/laplacian.py`

**Purpose.** Implements the p-adic Laplacian operator on the discrete token set {0, ..., L-1} with weights from the ultrametric kernel. The Laplacian matrix is L = W - diag(rowsum) where W[i,j] = eta^{v_13(|i-j|)}.

**Class `PAdicLaplacian` (static methods).**

| Method | Signature | Description |
|--------|-----------|-------------|
| `weight_matrix` | `(L, dev) -> Tensor` | W[i,j] = eta^{v_13(|i-j|)}, shape (L,L,3), diagonal = 0 |
| `matrix` | `(L, dev) -> Tensor` | Laplacian L = W - diag(rowsum), shape (L,L,3) |
| `apply` | `(Lm, f) -> Tensor` | (L, L, 3) @ (L, 3) -> (L, 3) via ring multiplication and sum |
| `ntt_eigendecomp` | `(L, dev) -> Tensor` | NTT of Laplacian columns for spectral analysis |

**Dependencies.** `constants` (P, ORD, ETA_POW, ZERO_R, _VAL_LUT), `ring` (Z13), `kernel` (PAdicKernel._val), `ntt` (NTT).

**Connection to Pipeline.** Used by `linear.laplacian` for discrete smoothing in attention layers. The NTT eigendecomposition connects to spectral analysis in `zeta_func.py`.

---

### 2.3 `linear.py` — Shared Ring Linear Algebra Operations

**File:** `zeta/linear.py`

**Purpose.** Provides common linear operations used across all model components. All functions are fully vectorized; no Python loops over sequence positions.

**Functions.**

| Function | Signature | Description |
|----------|-----------|-------------|
| `ring_lin` | `(x, W) -> Tensor` | Ring-linear map: sum_d x[d] * W[d,:]. x:(...,D_in,3), W:(D_in,D_out,3) -> (...,D_out,3) |
| `born_norm` | `(S) -> Tensor` | Row-normalizes score matrix using algebraic norm. S:(B,Lq,Lk,3) -> same. Replaces softmax. |
| `ring_attend` | `(A, V) -> Tensor` | O[b,i,d] = sum_j A[b,i,j] * V[b,j,d]. A:(B,L,L,3), V:(B,L,D,3) -> (B,L,D,3) |
| `laplacian` | `(x) -> Tensor` | Delta x[i] = x[i+1] + x[i-1] + 11*x[i] (since -2 = 11 mod 13). x:(B,L,D,3) -> same |
| `w_seed` | `(d_in, d_out) -> Tensor` | Deterministic weights: W[i,j] = eta^{(i*d_out+j) mod 168}. Shape (d_in,d_out,3). No nn.Parameter. |
| `alg_dropout` | `(x, rate, step, training) -> Tensor` | Deterministic dropout: keep mask = (ETA_POW[(rate*k*(step+1)) mod 168, 0] != 0) |

**Dependencies.** `constants` (P, ORD, ETA_POW, ZetaConfig), `ring` (Z13), `hilbert` (HilbertEta).

**Connection to Pipeline.** `ring_lin` is used by `SpectralAttention` for Q/K/V projections. `laplacian` is applied after every attention layer. `w_seed` generates all deterministic weight matrices. `alg_dropout` provides training-time regularization without randomness.

---

### 2.4 `hilbert.py` — Hilbert-eta Inner Product and Algebraic Layer Norm

**File:** `zeta/hilbert.py`

**Purpose.** Defines the inner product <u,v>_eta = sum_k Z13.mul(u_k, v_k) * eta^{-k} and algebraic layer normalization x * ||x||_eta^{-1}. Replaces Euclidean LayerNorm.

**Class `HilbertEta` (static methods).**

| Method | Signature | Formula |
|--------|-----------|---------|
| `inner` | `(u, v) -> Tensor` | sum_k Z13.mul(Z13.mul(u, v), ETA_IPOW[:D]) mod 13 |
| `norm_sq` | `(u) -> Tensor` | sum_k Z13.mul(Z13.mul(u, u), ETA_IPOW[:D]) mod 13 |
| `layer_norm` | `(x) -> Tensor` | x * inv(norm_sq(x)) reshaped to (B, L, D, 3) |

**Implementation.** `layer_norm` flattens x to (B*L, D, 3), computes norm_sq per batch-position, inverts via `Z13.inv`, expands to (B*L, D, 3), multiplies pointwise, and reshapes back.

**Dependencies.** `constants` (P, ORD, ETA_IPOW), `ring` (Z13).

**Connection to Pipeline.** `layer_norm` is the final operation in every `SpectralAttention.forward` call, applied after the residual connection and Laplacian smoothing.

---


## Tier 3: Transforms and Arithmetic

---

### 3.1 `ntt.py` — Number Theoretic Transform over Z_13[eta]

**File:** `zeta/ntt.py`

**Purpose.** Implements the Number Theoretic Transform over the cubic p-adic ring. Valid sizes are divisors of 168 with N mod 13 != 0: [1, 2, 3, 4, 6, 12]. Two paths: scalar (N | 12) with twiddles in F_13 via einsum, and ring (N ∤ 12) with twiddles in Z_13[eta] via broadcast ring_mul. All transforms are O(N log N) and fully vectorized.

**Class `NTT` (static methods).**

| Method | Signature | Description |
|--------|-----------|-------------|
| `ntt` | `(x, N) -> Tensor` | Forward NTT. x:(...,N,3) -> (...,N,3). Scalar or ring twiddle via `_NTT_FWD[N]` |
| `intt` | `(X, N) -> Tensor` | Inverse NTT. X:(...,N,3) -> (...,N,3). Uses `_NTT_INV[N]` |
| `conv` | `(a, b, N) -> Tensor` | Cyclic convolution via convolution theorem: INTT(NTT(a) * NTT(b)) |
| `spectrum` | `(x, N) -> Tensor` | Algebraic power spectrum: N(NTT(x)[k]). Returns (...,N) |
| `cross` | `(a, b, N) -> Tensor` | NTT(a) * conj(NTT(b)) where conj = Galois conjugation |
| `autocorr` | `(a, N) -> Tensor` | R[m] = sum_n a[n] * a_bar[n+m]. Via INTT(cross(a,a,N)) |
| `best_size` | `(L) -> int` | Largest valid NTT size <= L |

**Scalar Path.** For N in [1,2,3,4,6,12], the twiddle matrix T has shape (N,N) with entries in F_13. The transform is `einsum('kn,...nc->...kc', T, x) % P`.

**Ring Path.** For N not dividing 12, T has shape (N,N,3) with entries in Z_13[eta]. The transform is `Z13.mul(x.unsqueeze(-3), T).sum(-2) % P`.

**Dependencies.** `constants` (P, ORD, _NTT_VALID, _NTT_FWD, _NTT_INV), `ring` (Z13).

**Connection to Pipeline.** Used by `ErrorCache.spectral_correction`, `PAdicLaplacian.ntt_eigendecomp`, `BerrySVD.svd_proxy`, `SpectralFlow.ntt_spectrum`, and `AdelicDiracOperator.spectrum`.

---

### 3.2 `witt.py` — Witt Vector Arithmetic

**File:** `zeta/witt.py`

**Purpose.** Implements Witt vector arithmetic of precision k over the cubic p-adic ring. Witt vectors provide a canonical lift from the residue field to characteristic zero without floating-point arithmetic. Operations include addition with carry propagation, multiplication via anti-diagonal scatter-add, and Newton inversion.

**Class `WittVector` (static methods).**

| Method | Signature | Description |
|--------|-----------|-------------|
| `from_ring` | `(a, prec) -> Tensor` | tau(a) = (a, 0, ..., 0). a:(...,3) -> (...,prec,3) |
| `to_ring` | `(w) -> Tensor` | Extracts w[...,0,:] — the residue component |
| `wadd` | `(x, y) -> Tensor` | Witt addition with carry: (x+y) + carry_shifted mod 13 |
| `wneg` | `(x) -> Tensor` | Additive inverse: (P - x) mod P |
| `wsub` | `(x, y) -> Tensor` | Witt subtraction via wadd(x, wneg(y)) |
| `wmul` | `(x, y) -> Tensor` | Witt multiplication: outer products -> anti-diagonal scatter_add -> carry |
| `wpow` | `(x, n) -> Tensor` | Binary exponentiation (8 unrolled steps) for n < 168 |
| `winv` | `(x) -> Tensor` | Newton inversion: prec-1 explicit steps, initial guess from ring inverse |
| `ghost` | `(w) -> Tensor` | Ghost map Phi_k(w) = sum_{i<=k} 13^i * w_i |
| `frobenius` | `(w) -> Tensor` | Galois action on Witt components via CRT |
| `teichmuller` | `(a, prec) -> Tensor` | Alias for from_ring |

**Carry Propagation.** In `wadd`, raw = x + y; carry = raw // P; carry_in[...,1:,:] = carry[...,:-1,:]; result = (raw + carry_in) % P.

**Anti-diagonal Scatter.** In `wmul`, for precision m, the outer product of x and y produces (m,m,3) values. These are scattered along anti-diagonals (i+j = const) into a (2m-1,3) buffer, then truncated to (m,3) and carry-propagated.

**Dependencies.** `constants` (P, ORD, ETA_POW, ONE_R), `ring` (Z13, CRT, F169).

**Connection to Pipeline.** `WittVector.wadd` and `wmul` are used by `ZetaModel._sync_head` and `train_step` for precision-adaptive weight updates. `from_ring` converts ring corrections to Witt vectors.

---

### 3.3 `hensel.py` — Hensel Lifting of the Tribonacci Root

**File:** `zeta/hensel.py`

**Purpose.** Constructs the exact root eta of x^3 - x^2 - x - 1 modulo 13^k via Newton iteration. Links Hensel lifting to Witt vector arithmetic through base-13 digits.

**Class `HenselLifter` (static methods).**

| Method | Signature | Description |
|--------|-----------|-------------|
| `_f` | `(x) -> int` | f(x) = x^3 - x^2 - x - 1 |
| `eta_int` | `(prec) -> int` | Newton lift of eta to precision prec. Precomputed cache: {1:7, 2:33, 3:202, 4:26566, 5:340737} |
| `digits` | `(prec) -> Tensor` | Base-13 digits of eta: [7, 2, 1, 12, 11, ...]. Shape (prec,) |
| `witt_eta` | `(prec) -> Tensor` | eta as Witt vector: (prec,3) with digit in component 0 |
| `verify` | `(k) -> bool` | Checks f(eta_int(k)) mod 13^k == 0 |

**Newton Iteration.** Correction factor: Nw = 7 (since f'(7)^{-1} = 7 mod 13). At each step: t = (-(f(a) // pk) * Nw) mod 13; a += t * pk; pk *= 13.

**Class `HenselIO` (static methods).**
- `sensor_embed(s) -> Tensor` — Maps sensor reading s in {0,...,12} to (s,0,0) in Z_13[eta].
- `actuator_decode(a) -> tuple` — Maps ring element (a0,a1,a2) to (motor, direction, intensity) in Z_13.
- `feedback_check(measured, predicted) -> bool` — Detects conflict via inequality check.

**Dependencies.** `constants` (P, ZetaConfig), `witt` (WittVector).

**Connection to Pipeline.** `HenselLifter` provides the theoretical foundation for Witt precision. `HenselIO` is the empirical closure interface between the algebraic system and external sensors/actuators.

---

### 3.4 `mahler.py` — Mahler Expansion in Z_13[eta]

**File:** `zeta/mahler.py`

**Purpose.** Computes Mahler coefficients Delta^k f(0) / k! for functions over the p-adic ring. Finite differences are vectorized via binomial matrix construction. The NTT bridge connects Mahler expansion to frequency-domain analysis.

**Class `MahlerExpansion` (static methods).**

| Method | Signature | Description |
|--------|-----------|-------------|
| `_binom_matrix` | `(K, dev) -> Tensor` | KxK binomial coefficient matrix mod 13 with 1/k! weights |
| `finite_differences` | `(f) -> Tensor` | Computes Delta^k f(0) via binomial matrix. f:(N,3) -> (N,3) |
| `coeffs` | `(f) -> Tensor` | Mahler coefficients a_k = Delta^k f(0) / k! mod 13 |
| `reconstruct` | `(ak, n) -> Tensor` | Reconstructs f(n) from Mahler coefficients |
| `ntt_bridge` | `(f, N) -> Tensor` | NTT of finite differences: NTT(f[n+1] - f[n]) |

**Inverse Factorial Table.** `_INV_FACT = [1, 1, 7, 9, 3, 5, 11, 9, 3, 7, 9, 1, 1]` — precomputed 1/k! mod 13 for k = 0..12.

**Dependencies.** `constants` (P), `ring` (Z13), `ntt` (NTT).

**Connection to Pipeline.** Provides finite-difference calculus for analyzing functions over the p-adic ring. The NTT bridge connects to spectral analysis modules.

---

### 3.5 `teichmuller.py` — Teichmüller Lifts

**File:** `zeta/teichmuller.py`

**Purpose.** Implements the multiplicative Teichmüller section tau(a) = (a, 0, ..., 0) in Witt vectors, satisfying tau(a*b) = tau(a)*tau(b).

**Class `Teichmuller` (static methods).**

| Method | Signature | Description |
|--------|-----------|-------------|
| `lift` | `(a, prec) -> Tensor` | Alias for WittVector.teichmuller |
| `orbit` | `(prec) -> Tensor` | All 168 eta powers as Witt vectors of given precision |
| `mul_table` | `(K) -> Tensor` | Multiplication table: ETA_POW[(i+j) mod 168] for i,j in 0..K-1 |
| `ntt_spectrum` | `(N, dev) -> Tensor` | NTT of the first N eta powers |

**Dependencies.** `constants` (ORD, ETA_POW), `witt` (WittVector), `ntt` (NTT).

**Connection to Pipeline.** Provides the multiplicative section for Witt vector construction. The orbit table is used for spectral analysis.

---


## Tier 4: Quantum and Spectral Physics

---

### 4.1 `gqm.py` — Geometric Quantum Mechanics in Z_13[eta]

**File:** `zeta/gqm.py`

**Purpose.** Implements quantum states, density matrices, and quantum cache over the cubic p-adic ring. All operations use ring arithmetic (Z13.mul, Z13.trace, Z13.norm) — no Hilbert space, no complex numbers. The Born rule uses algebraic norm N(c_k) rather than |c_k|^2.

**Function `stratum(k) -> Tensor`.**
- T3^k * e0 = eta^k. Returns (3,) Long tensor representing the k-th orbital state.

**Class `GQMState`.**

| Method | Signature | Description |
|--------|-----------|-------------|
| `__init__` | `(K=8)` | Initializes K coefficients, all zero except c0 = (1,0,0) |
| `from_coeffs` | `(c) -> GQMState` | Class method: constructs from coefficient tensor |
| `evolve` | `(N) -> GQMState` | Applies T3^N to all coefficients |
| `hamiltonian` | `() -> GQMState` | Multiplies each coefficient by ETA_POW[k] |
| `born_probs` | `() -> Tensor` | p_k = N(c_k) * 2^k mod 13. Shape (K,) |
| `inner` | `(other) -> int` | <psi|phi> = Tr(conj(coeffs) * other.coeffs) mod 13 |
| `density` | `() -> Tensor` | rho[i,j] = conj(c_i) * c_j. Shape (K,K,3) |
| `ntt_spectrum` | `() -> Tensor` | NTT of coefficients |
| `entropy` | `() -> int` | Count of zero-norm coefficients |

**Class `ErrorCache`.**
- `__init__(K=64)` — Initializes empty density matrix rho:(K,K,3) and step counter.
- `absorb(corr, dev)` — Flattens correction tensor to K states, builds density matrix via `DensityMatrix.build`, and accumulates with T3-evolution of previous state.
- `spectral_correction(head_shape, dev)` — Extracts correction from accumulated error spectrum. Computes NTT of diagonal (Born probabilities), repeats to fill (D,V,3) head shape, and phases by ETA_POW[step].

**Class `DensityMatrix` (static methods).**
- `build(psi)` — Outer product: psi_i * psi_j. Shape (K,K,3).
- `trace(rho)` — Sum of diagonal elements.
- `mat_mul(A, B)` — Matrix multiplication in the ring.
- `evolve(rho, n)` — Conjugation: T3^n * rho * T3^{-n}.
- `entropy(rho)` — S_G = Tr(rho) - Tr(rho^2).

**Class `ZetaSelf(GQMState)`.**
- `__init__(K=8)` — Persistent self-state with time counter t=0.
- `evolve()` — T3 * |Psi_self> — one time step. Increments t mod 168.
- `reflect(goal)` — P1 * (self - goal) — returns dominant deficit.
- `is_my_experience(other)` — Checks if <Psi_self | other> != 0.

**Dependencies.** `constants` (P, ORD, ETA_POW, T3_POW, ZERO_R, P1_MAT), `ring` (Z13), `spectral` (proj), `ntt` (NTT).

**Connection to Pipeline.** `ZetaSelf` is the persistent self-state in `AutonomousLoop`. `ErrorCache` absorbs Buchberger corrections as quantum memory in `ZetaModel.train_step`.

---

### 4.2 `zeta_func.py` — Spectral Zeta Functions and Eigenvalue Flows

**File:** `zeta/zeta_func.py`

**Purpose.** Implements the spectral zeta function zeta_eta(s) over the Tribonacci orbit, together with the dominant and subdominant eigenvalue flows of T3.

**Class `ZetaFunctionRing` (static methods).**

| Method | Signature | Description |
|--------|-----------|-------------|
| `zeta` | `(s0, terms=168) -> Tensor` | sum_{n=1}^{terms} ETA_IPOW[(n*s0) mod 168] mod 13 |
| `zeta_batch` | `(s_vec) -> Tensor` | Vectorized zeta for multiple s values simultaneously |
| `ntt_identity` | `(N, dev) -> bool` | Verifies NTT(orbit) == zeta_batch for all s in 0..N-1 |
| `functional_eq` | `(s0) -> bool` | Checks zeta(s) + zeta(168-s) == 2*zeta(0) |

**Class `SpectralFlow` (static methods).**

| Method | Signature | Description |
|--------|-----------|-------------|
| `dominant` | `(dev) -> Tensor` | phi1(ETA_POW[:168]) — dominant eigenvalue flow |
| `subdominant` | `(dev) -> Tensor` | phi2(ETA_POW[:168]) — subdominant flow in F_169 |
| `ntt_spectrum` | `(dev) -> Tensor` | NTT of first 12 dominant values |
| `chern_proxy` | `(dev) -> int` | Sum of F_169 norms of differences between consecutive subdominant values |
| `winding` | `() -> int` | ORD // 12 = 14 |
| `critical_indices` | `(dev) -> Tensor` | Positions where dominant flow equals 7 |

**Class `SpectralZeta` (static methods).**
- `zeta(s, terms)` — Weighted zeta with trace coefficients: sum tr(ETA_POW[n]) * ETA_IPOW[(n*s) mod 168].
- `connection_to_zeta_ring(s)` — Verifies consistency between SpectralZeta and ZetaFunctionRing.

**Dependencies.** `constants` (P, ORD, ETA_POW, ETA_IPOW), `ring` (Z13, F169), `ntt` (NTT).

**Connection to Pipeline.** `SpectralFlow.winding()` and `critical_indices` are used by `AdelicDiracOperator.rh_check`. `ZetaFunctionRing.zeta_batch` verifies the p-adic functional equation.

---

### 4.3 `dirac.py` — Adelic Dirac Operator and p-adic Riemann Hypothesis

**File:** `zeta/dirac.py`

**Purpose.** Implements the antisymmetric Dirac operator on the token lattice and provides the p-adic Riemann Hypothesis verification framework.

**Class `AdelicDiracOperator` (static methods).**

| Method | Signature | Description |
|--------|-----------|-------------|
| `matrix` | `(L, dev) -> Tensor` | D[i,j] = (eta^{i-j} - eta^{j-i}) * G(i,j). Antisymmetric, diagonal=0. Shape (L,L,3) |
| `chiral_check` | `(L, dev) -> bool` | Verifies antisymmetry: D + D^T == 0 off-diagonal |
| `spectrum` | `(L, dev) -> Tensor` | NTT of Dirac matrix columns |
| `zeros_of_zeta` | `(L, dev) -> Tensor` | Indices where phi1(spectrum) == 0 |
| `rh_check` | `(L=12, dev) -> dict` | Comprehensive RH verification dictionary |

**RH Check Returns.**
- `chiral` — Dirac antisymmetry holds.
- `zeros_P23` — All zeros lie in P23 (subdominant) channel.
- `crit_14` — Exactly 14 critical indices exist (168/12).
- `pair` — zeta(7) has zero dominant component (phi1(zeta(7)) == 0).
- `zeta_eq` — For all s in 1..11, phi1(zeta(s)) == 0 (critical line in P23).
- `page_half` — Page symmetry verified.
- `RH` — True iff all boolean checks pass.

**Dependencies.** `constants` (P, ORD, ETA_POW, ETA_IPOW, P1_MAT, _VAL_LUT, _VAL_MAX, ONE_R, ZERO_R), `ring` (Z13, F169), `kernel` (PAdicKernel), `ntt` (NTT), `zeta_func` (ZetaFunctionRing, SpectralFlow), `entanglement` (EntanglementGeometry), `spectral` (proj).

**Connection to Pipeline.** `rh_check` is callable from `ZetaRuntime` and `AutonomousLoop`. The Dirac operator provides the spectral foundation for the p-adic RH analogue.

---

### 4.4 `berry.py` — Berry Phase and Algebraic SVD Proxy

**File:** `zeta/berry.py`

**Purpose.** Computes the Berry connection via Sylvester P1 projectors and provides an algebraic SVD proxy through the NTT power spectrum. No Euclidean singular value decomposition is used.

**Class `BerrySVD` (static methods).**

| Method | Signature | Description |
|--------|-----------|-------------|
| `connection` | `(row) -> Tensor` | dA = P1(row[i+1] - row[i]). Scaled by I_P = 5. |
| `phase` | `(row) -> int` | Sum of connection components mod 13 |
| `svd_proxy` | `(x) -> Tensor` | NTT power spectrum: N(NTT(x)) |
| `dominant_mode` | `(x) -> int` | Index of maximum NTT power |

**Dependencies.** `constants` (P, ZetaConfig, P1_MAT), `ring` (Z13), `spectral` (proj), `ntt` (NTT).

**Connection to Pipeline.** The SVD proxy provides spectral analysis without Euclidean decomposition. Used for analyzing state distributions.

---

### 4.5 `entanglement.py` — Discrete Holographic Entanglement in SL(3,Z)

**File:** `zeta/entanglement.py`

**Purpose.** Implements holographic entropy formulas over the p-adic ring. The central charge Delta = 2 mod 13 connects to the Ryu-Takayanagi formula, Page curve, and wormhole entropy.

**Class `EntanglementGeometry` (static methods).**

| Method | Signature | Formula |
|--------|-----------|---------|
| `ryu_takayanagi` | `(nA, nB) -> int` | Delta * nA * nB mod 13 |
| `wormhole_entropy` | `(n) -> int` | Delta * n mod 13 |
| `page_curve` | `(t, N) -> int` | Delta * min(t mod 13, (N-t) mod 13) |
| `state_entropy` | `(psi, n_strata=12) -> int` | Count of zero P23-norm states * Delta |
| `info_conservation` | `(n) -> bool` | det(T3^n) == 1 |

**Dependencies.** `constants` (P, ORD, T3_POW, P23_MAT), `ring` (Z13, F169), `spectral` (proj).

**Connection to Pipeline.** Entropy formulas provide algebraic analogues of quantum information measures. `info_conservation` verifies volume preservation.

---

### 4.6 `crystal.py` — Crystal Lattice in Z_13[eta]

**File:** `zeta/crystal.py`

**Purpose.** Treats Z_13[eta] as a 3D lattice {1, eta, eta^2} over Z_13. Computes structure factor via NTT and Voronoi decomposition via p-adic valuation.

**Class `CrystalLattice` (static methods).**

| Method | Signature | Description |
|--------|-----------|-------------|
| `unit_cell` | `(dev) -> Tensor` | All 168 eta powers |
| `dominant_sublattice` | `(dev) -> Tensor` | P1 projection of all eta powers |
| `structure_factor` | `(f, N) -> Tensor` | NTT of f |
| `reciprocal` | `(N, dev) -> Tensor` | NTT of first N eta powers |
| `voronoi` | `(L, dev) -> Tensor` | For each position, finds nearest eta power by p-adic valuation |

**Dependencies.** `constants` (P, ORD, ETA_POW, P1_MAT), `spectral` (proj), `ntt` (NTT), `kernel` (PAdicKernel._val).

**Connection to Pipeline.** Lattice structure analysis for understanding the geometric arrangement of ring elements.

---


## Tier 5: Model, Attention, and Hybrid

---

### 5.1 `embed.py` — Token Embedding and Positional Encoding

**File:** `zeta/embed.py`

**Purpose.** Provides deterministic embeddings without learned parameters. Both embeddings are constructed from precomputed ETA_POW tables. No nn.Parameter is used.

**Class `ZRingEmbed(nn.Module)`.**
- `__init__(V, D)` — Builds embedding table W of shape (V, D, 3) where W[v,d] = ETA_POW[(v*D + d) mod 168].
- `forward(t) -> Tensor` — Index lookup: returns W[t % V]. Input t:(B,L), output (B,L,D,3).

**Class `TatePE(nn.Module)`.**
- `__init__(D)` — Asserts D % 2 == 0.
- `forward(x) -> Tensor` — Dynamic positional encoding. For input x:(B,L,D,3):
  - n = torch.arange(L), k = torch.arange(D//2)
  - nk = (n.unsqueeze(1) * k.unsqueeze(0)) % 168
  - PE[n, 2k] = ETA_POW[nk], PE[n, 2k+1] = ETA_IPOW[nk]
  - Returns Z13.add(x, PE.unsqueeze(0).expand(B, -1, -1, -1))

**Dependencies.** `constants` (P, ORD, ETA_POW, ETA_IPOW, ZetaConfig), `ring` (Z13).

**Connection to Pipeline.** `ZRingEmbed` and `TatePE` are the first two layers in `ZetaModel.forward`. They map discrete tokens into the ring and add positional information.

---

### 5.2 `attention.py` — Spectral Attention via Bruhat-Tits Tree

**File:** `zeta/attention.py`

**Purpose.** The sole attention mechanism of the Zeta engine. Combines: (1) Sylvester split into dominant (P1, 1D) and subdominant (P23, 2D) channels, (2) THE ONLY KERNEL — Bruhat-Tits tree mixing via PAdicKernel, (3) local Q/K/V projections (pointwise along D, never L×L), (4) Born normalization replacing softmax.

**Class `SpectralAttention(nn.Module)`.**

**Buffers (all deterministic, no nn.Parameter):**
- `P1` — (3,3), Sylvester projector P1_MAT
- `P23` — (3,3), Sylvester projector P23_MAT
- `WQ_dom, WK_dom, WV_dom` — (D, D/3, 3), Q/K/V weights for dominant channel
- `WQ_sub, WK_sub, WV_sub` — (D, D/3, 3), Q/K/V weights for subdominant channel
- `WO` — (2*D/3, D, 3), output projection

**Forward pass `(x, dm, li=0) -> (Tensor, dict)`:**

1. **Sylvester split:** x_p1 = proj(x, P1), x_p23 = proj(x, P23)
2. **Tree kernel mixing:**
   - If L <= 169: `PAdicKernel.apply_fast(x_p1)` and `apply_fast(x_p23)`
   - Else: `PAdicKernel.apply(x_p1)` and `apply(x_p23)`
3. **Local Q/K/V projections:**
   - Q1 = ring_lin(x_p1_k, WQ_dom), K1 = ring_lin(x_p1_k, WK_dom), V1 = ring_lin(x_p1_k, WV_dom)
   - Q2 = ring_lin(x_p23_k, WQ_sub), K2 = ring_lin(x_p23_k, WK_sub), V2 = ring_lin(x_p23_k, WV_sub)
   - All shapes: (B, L, D/3, 3)
4. **Born normalization:**
   - S1 = Z13.trace(Z13.mul(Q1, K1)) % P — shape (B, L, D/3)
   - S2 = Z13.trace(Z13.mul(Q2, K2)) % P — shape (B, L, D/3)
   - O1 = Z13.mul(V1, S1.unsqueeze(-1).expand_as(V1)) — shape (B, L, D/3, 3)
   - O2 = Z13.mul(V2, S2.unsqueeze(-1).expand_as(V2))
5. **Output projection:** out = ring_lin(cat([O1, O2], dim=2), WO)
6. **Residual + Laplacian + HilbertEta layer norm:**
   - out = HilbertEta.layer_norm(laplacian(Z13.add(x, out)))
   - out = alg_dropout(out, DROP, _step, training)

**Return.** `(out, {'li': li, 'dm': dm, 'step': _step})`

**Dependencies.** `constants` (P, ORD, ZetaConfig, P1_MAT, P23_MAT), `ring` (Z13), `spectral` (proj), `kernel` (PAdicKernel), `hilbert` (HilbertEta), `linear` (ring_lin, laplacian, w_seed, alg_dropout).

**Connection to Pipeline.** Instantiated N=11 times in `ZetaModel`. Each layer processes the full sequence through spectral split, tree mixing, and local projections.

---

### 5.3 `hybrid.py` — S3 Parallel Hybrid Mode

**File:** `zeta/hybrid.py`

**Purpose.** Runs all 6 S3 conjugate orbits simultaneously in a single forward pass. Each token is evolved along 6 independent T3^n·S3_g paths, then combined via the S3 Casimir (sum over all Galois images). Provides 6× information bandwidth without increasing sequence length.

**Class `S3ParallelAttention` (static methods).**

| Method | Signature | Description |
|--------|-----------|-------------|
| `evolve_parallel` | `(x, n) -> Tensor` | T3^n * S3_g * x for all g in 0..5. x:(B,L,D,3) -> (B,6,L,D,3) |
| `casimir_reduce` | `(parallel) -> Tensor` | Sum over S3 axis: parallel.sum(dim=1) % P. (B,6,L,D,3) -> (B,L,D,3) |
| `apply` | `(x, n, dm, layer, training, step) -> Tensor` | Full parallel attention pipeline |

**Full Pipeline (`apply`):**
1. `evolve_parallel(x, n)` — (B, 6, L, D, 3)
2. Reshape to (B*6, L, D, 3), apply `PAdicKernel.apply_fast`
3. Reshape back to (B, 6, L, D, 3)
4. Sylvester split: P1 projection on reshaped tensor
5. Casimir reduction: sum over S3 axis
6. Residual + Laplacian + HilbertEta layer norm + dropout

**Dependencies.** `constants` (P, ORD, T3_POW, P1_MAT, P23_MAT, S3_MATS), `ring` (Z13), `spectral` (proj, t3n), `kernel` (PAdicKernel), `hilbert` (HilbertEta), `linear` (laplacian, alg_dropout).

**Connection to Pipeline.** Activated by `ZetaScaler` when error rate exceeds 50%. The model's `s3_orbit` field controls whether standard T3 evolution or S3 parallel mode is used.

---

### 5.4 `model.py` — Zeta p-Adic Language Model

**File:** `zeta/model.py`

**Purpose.** The complete Zeta language model. Architecture: embed -> TatePE -> N × (t3n/S3 evolution + SpectralAttention) -> multi-Witt head. Contains multi-layer Witt forward, S3 conjugate orbits, convergent decay, and autonomous Hensel lift.

**Class `ZetaModel(nn.Module)`.**

**Hyperparameters (defaults from ZetaConfig):**
- V = 256, D = 54, N = 11, ctx = 256, PREC_MAX = 4

**Submodules:**
- `embed` — ZRingEmbed(V, D)
- `pe` — TatePE(D)
- `layers` — nn.ModuleList of N SpectralAttention(D) layers
- `head` — (D, V, 3), deterministic embedding table
- `witt_head` — (D, V, PREC_MAX, 3), Witt vector weights

**Fields:**
- `step` — global training step counter
- `prec` — current Witt precision (starts at 1)
- `error_cache` — ErrorCache or None
- `s3_orbit` — -1 (disabled) or 0..5 (active S3 conjugate)

**Methods.**

`_sync_head()` — Synchronizes composite head from active Witt layers:
```
active = witt_head[:, :, :prec, :]
weights = arange(1, prec+1).view(1, 1, -1, 1)
head = (active * weights).sum(2) % P
```

`_lift()` — Increments prec if below PREC_MAX, then calls _sync_head.

`forward(tokens) -> (Tensor, dict)`:
1. Embed and add PE: x = pe(embed(tokens))
2. For each layer i = 0..N-1:
   - If s3_orbit >= 0: x = t3n_s3(x, i+1, s3_orbit)
   - Else: x = t3n(x, i+1)
   - x, la = layers[i](x, dm, i)
3. _sync_head()
4. Score: sc = trace(mul(x.unsqueeze(-2), head.unsqueeze(0).unsqueeze(0))).sum(2) % P
5. Return (sc, aud)

`train_step(tokens, targets) -> dict`:
1. Forward pass to get predictions
2. Compute hallucinations: hall = count(pred != target)
3. If hall > 0 and _last_x exists:
   - Buchberger correction: delta = nullstellensatz_correction(_last_x, preds, tgts, step)
   - Convergent decay: if norm(delta) > 6, delta = smul(delta, 7)  # 7 = 2^{-1} mod 13
   - Witt update: delta_witt = from_ring(delta, PREC); witt_head += delta_witt
   - _sync_head()
   - ErrorCache.absorb(delta)
   - Spectral correction from cache: spec_corr = spectral_correction(head_shape, dev)
   - spec_corr = smul(spec_corr, 2); witt_head += spec_corr_witt; _sync_head()
   - If entropy triggers and prec < PREC_MAX: _lift()
4. Return metrics dict

**Dependencies.** `constants`, `ring`, `spectral`, `embed`, `attention`, `buchberger`, `gqm`, `witt`, `entropy`, `kernel` (via attention).

**Connection to Pipeline.** This is the central model instantiated by `ZetaRuntime`. All training and inference flow through it.

---

### 5.5 `mera.py` — MERA Tensor Network in Z_13[eta]

**File:** `zeta/mera.py`

**Purpose.** Implements Multi-scale Entanglement Renormalization Ansatz coarse-graining over the p-adic ring. Each level groups pairs of tokens via ring addition followed by T3^{2^k} evolution.

**Class `MERAChunker` (static methods).**

| Method | Signature | Description |
|--------|-----------|-------------|
| `coarse_grain` | `(x, levels=3) -> List[Tensor]` | Fixed 3-level coarse-graining |
| `coarse_grain_dynamic` | `(x) -> List[Tensor]` | Auto-depth: floor(log2(L)), cap 10 |
| `ntt_per_level` | `(x) -> Tensor` | NTT of each level via best_size |
| `renormalize` | `(levels) -> Tensor` | Upsamples coarse levels and adds back |

**Fixed Coarse-Graining (`coarse_grain`):**
- Level 1: group pairs (even + odd), evolve by T3^1
- Level 2: group pairs, evolve by T3^2
- Level 3: group pairs, evolve by T3^4
- Odd lengths handled via min(even, odd) truncation

**Dynamic Coarse-Graining (`coarse_grain_dynamic`):**
- Computes depth = min(floor(log2(L)), 10)
- Delegates to `coarse_grain(x, levels=max(depth, 1))`

**Dependencies.** `constants` (P, ORD, T3_POW), `ring` (Z13), `spectral` (t3n), `ntt` (NTT).

**Connection to Pipeline.** `MERAChunker.coarse_grain` is called by `AutonomousLoop` for hierarchical feature extraction. The dynamic depth variant is activated by `ZetaScaler` when ctx > 64.

---


## Tier 6: Learning and Autonomy

---

### 6.1 `buchberger.py` — Buchberger-Nullstellensatz Algebraic Learning Engine

**File:** `zeta/buchberger.py`

**Purpose.** Implements genuine algebraic learning without gradient descent. For each token position (b,l), the error polynomial in Z_13[eta][y] is e_{b,l} = (target - pred) embedded as (delta, 0, 0). The correction Delta_head is computed from the witness via the input activations x.

**Class `BuchbergerEngine` (static methods).**

| Method | Signature | Description |
|--------|-----------|-------------|
| `error_ideal` | `(pred, target) -> Tensor` | e_i = (target_i - pred_i) mod 13, embedded as (e_i, 0, 0). Shape (B, L, 3) |
| `nullstellensatz_correction` | `(x, pred, target, step, scale=None) -> Tensor` | Full vectorized correction. Returns (D, V, 3) |
| `groebner_reduce` | `(errors) -> Tensor` | For constant errors, returns unit generator (1,0,0) if any error non-zero |

**Nullstellensatz Correction Algorithm:**

1. **Error ideal:** e = error_ideal(pred, target) — (B, L, 3)
2. **Witness:** g = x * eta^{step} — (B, L, D, 3)
3. **Contribution:** contrib = g * e (broadcast) — (B, L, D, 3)
4. **Target one-hot:** target_oh:(B, L, V), pred_oh:(B, L, V)
5. **Boost target:** corr_target = sum_{b,l} contrib[b,l,:] * target_oh[b,l,:] — (D, V, 3)
6. **Suppress pred:** corr_pred = sum_{b,l} contrib[b,l,:] * pred_oh[b,l,:] — (D, V, 3)
7. **Net correction:** delta = corr_target - corr_pred — (D, V, 3)
8. **Adaptive scale:** if scale provided, delta = delta * scale
9. **Temporal phase:** delta = delta * eta^{7*step} — phase factor from dominant eigenvalue orbit

**Complexity.** O(B * L * D + D * V) time. Zero Python loops over sequence positions.

**Dependencies.** `constants` (P, ORD, ETA_POW, ZetaConfig), `ring` (Z13).

**Connection to Pipeline.** `nullstellensatz_correction` is called by `ZetaModel.train_step` for every training batch. The returned (D, V, 3) tensor is converted to Witt vector and added to `witt_head`.

---

### 6.2 `goal.py` — Goal-Directed Orbit Planning

**File:** `zeta/goal.py`

**Purpose.** Implements goal-directed behavior purely algebraically. Distance metric: d_goal(|Psi>, |C>) = N(P1 * (|Psi> - |C>)) mod 13. Planning searches the T3 orbit {T3^n * |Psi>} for the smallest n and selects the processing channel.

**Class `ZetaGoalPlanner` (static methods).**

| Method | Signature | Description |
|--------|-----------|-------------|
| `distance` | `(state, goal) -> Tensor` | N(P1 * (state - goal)) mod 13. Scalar |
| `plan` | `(state, goal, max_n=168) -> dict` | Full vectorized orbit search |

**Plan Algorithm:**

1. **Vectorized evolution:** M = T3_POW[:max_n]; evolved = einsum('nrc,kc->nkr', M, state) — (168, K, 3)
2. **Differences:** diff = (evolved - goal) % P — (168, K, 3)
3. **P1 projection:** p1_diff = proj(diff.reshape(-1,3), P1) — (168*K, 3)
4. **Norms:** norms = Z13.norm(p1_diff) — (168, K)
5. **Distance vector:** d_vec = norms.sum(1) % P — (168,)
6. **Channel selection:**
   - d == 0: direct hit, channel='direct'
   - d < 6: P1 dominant shift, channel='P1'
   - 6 <= d < 10: P23 subdominant exploration, channel='P23'
   - d >= 10: no good approach, channel='none', lift=True

**Return.** `{'n': int, 'd': int, 'channel': str, 'lift': bool, 'state': Tensor}`

**Dependencies.** `constants` (P, ORD, T3_POW, P1_MAT), `ring` (Z13), `spectral` (proj).

**Connection to Pipeline.** `ZetaGoalPlanner.plan` is called by `AutonomousLoop.cycle` when a target_goal is provided. The returned channel selection guides the model's processing strategy.

---

### 6.3 `entropy.py` — Entropic Monitoring and Hensel Catastrophe Trigger

**File:** `zeta/entropy.py`

**Purpose.** Monitors the entropy of the self-state and triggers a Witt lift (precision escalation) when the entropy barrier S_max is hit.

**Class `EntropyMonitor`.**

| Method | Signature | Description |
|--------|-----------|-------------|
| `__init__` | `(threshold=10)` | Sets S_max = threshold mod 13 |
| `check` | `(entropy) -> bool` | Returns entropy >= S_max |
| `lift_if_needed` | `(state, prec) -> Tensor` | If entropy triggers, returns WittVector.from_ring(rep, prec+1); else prec |

**Entropy Computation.** For GQMState: count of zero-norm coefficients. For raw tensor: count of zero-norm elements in flattened view.

**Dependencies.** `constants` (P), `ring` (Z13), `witt` (WittVector), `gqm` (GQMState).

**Connection to Pipeline.** `EntropyMonitor.check` is called by `AutonomousLoop.cycle` after self-evolution. If triggered, the model's `_lift()` method is called to increment Witt precision.

---

### 6.4 `scaler.py` — Autonomous Scaling Engine

**File:** `zeta/scaler.py`

**Purpose.** Manages automatic scaling of the Zeta engine based on internal state metrics: entropy, error rate, and orbit saturation. No external hyperparameters. Scaling mechanisms: multi-layer Witt head, S3 conjugate orbits, dynamic MERA depth, Hensel precision lift.

**Class `ZetaScaler`.**

| Method | Signature | Description |
|--------|-----------|-------------|
| `__init__` | `(model)` | Initializes scaling state machine |
| `observe` | `(hall, entropy, delta_norm)` | Records metrics into 24-step rolling window |
| `decide` | `() -> dict` | Autonomous scaling decisions |
| `reset_orbit` | `()` | Resets S3 orbit when goal reached |

**Scaling Decisions (`decide`):**
- **Witt full-layer:** Activate when avg_entropy > 6 (and not already active).
- **S3 orbit:** Cycle through conjugates when avg_err > 12 out of 24 steps (>50% error rate).
- **Dynamic MERA:** Activate when model.ctx > 64.
- **Hensel lift recommendation:** Issue when avg_ent >= 10.

**State Fields:**
- `s3_orbit` — -1 (disabled) or 0..5 (active conjugate)
- `mera_dynamic` — bool
- `witt_full` — bool
- `error_history` — list of last 24 hallucination counts
- `entropy_history` — list of last 24 entropy values
- `step` — observation counter

**Dependencies.** `constants` (P, ORD, ZetaConfig), `ring` (Z13).

**Connection to Pipeline.** Instantiated by `AutonomousLoop`. `observe` is called after every training step; `decide` is called to determine scaling actions, which are then applied to the model.

---

### 6.5 `autonomy.py` — Full Autonomy Cycle with Autonomous Scaling

**File:** `zeta/autonomy.py`

**Purpose.** Orchestrates the complete closed-loop autonomous cycle: persistent self, goal planning, entropy monitoring, automatic scaling, and reflection.

**Class `AutonomousLoop`.**

**Subcomponents:**
- `rt` — ZetaRuntime instance
- `self_state` — ZetaSelf(K=8), persistent orbital self
- `goal_state` — ZetaSelf or None
- `entropy_monitor` — EntropyMonitor(threshold=10)
- `scaler` — ZetaScaler(runtime.model)
- `history` — list of last 168 cycle entries

**Method `cycle(tokens, target_goal=None) -> dict`:**

1. **Forward pass:** sc, aud = rt.forward(tokens)
2. **Predictions:** preds = sc.argmax(-1) % P
3. **Self evolution:** self_state.evolve(); self_state.coeffs += top % P
4. **Goal planning:** If target_goal provided, plan = ZetaGoalPlanner.plan(self_state.coeffs, goal_state.coeffs)
5. **Entropy check:** S = self_state.entropy(); if entropy_monitor.check(S): model._lift()
6. **Training:** If hall > 0: metrics = rt.train_step(tokens, tokens)
7. **Scaling:** scaler.observe(hall, S, delta_norm); scale_actions = scaler.decide()
8. **Apply scaling:** Update model.s3_orbit, witt_full flag
9. **Record entry:** {'self_t', 'entropy', 'plan', 'lift_triggered', 'hall', 'scale_actions', **metrics}
10. **History management:** Append entry; if len > 168, pop oldest

**Method `reflect() -> dict`:**
- Computes average entropy and total hallucinations over last 12 steps
- Computes P1 deficit norm via self_state.reflect(goal_state)
- Returns reflection summary

**Class `FullAutonomyCycle` (static methods).**
- `cycle(model, tokens, targets) -> dict` — Simplified single-cycle variant without persistent state.

**Dependencies.** `constants`, `mera`, `gqm`, `orbit`, `ring`, `rollback`, `dirac`, `goal`, `entropy`, `scaler`.

**Connection to Pipeline.** `AutonomousLoop` is the primary user-facing interface for autonomous operation. It connects all learning, planning, and scaling modules into a single coherent cycle.

---


## Tier 7: Planning and Control

---

### 7.1 `orbit.py` — T3 Orbit Planner

**File:** `zeta/orbit.py`

**Purpose.** Plans trajectories in the T3 orbit space. Given a current state and a goal state, finds the smallest N in {0..167} such that T3^N * state = goal. All 168 T3 powers are tested simultaneously via vectorized broadcast.

**Class `OrbitPlanner` (static methods).**

| Method | Signature | Description |
|--------|-----------|-------------|
| `plan_orbit_vec` | `(state, goal) -> Tensor` | Vectorized search. Returns index tensor |
| `plan_orbit` | `(state, goal) -> int` | Scalar wrapper around plan_orbit_vec |
| `orbit_spectrum` | `(state) -> Tensor` | NTT of the full 168-step orbit |
| `closure_check` | `(state, N) -> bool` | Verifies t3n(t3n(state, N), 168-N) == state |

**Plan Algorithm.**
1. all_s = T3_POW @ state — shape (168, 3)
2. match = (all_s == goal).all(-1) — boolean mask (168,)
3. Return first True index, or 0 if no match

**Dependencies.** `constants` (P, ORD, T3_POW), `ntt` (NTT), `spectral` (t3n).

**Connection to Pipeline.** Used by `ZetaGoalPlanner` for goal approach search and by `AutonomousLoop` for orbit closure verification.

---

### 7.2 `reversible.py` — Time-Reversible Sequence Transform

**File:** `zeta/reversible.py`

**Purpose.** Implements invertible encoding and decoding via the T3 orbit. The roundtrip property decode(encode(x)) = x holds exactly, providing time reversibility.

**Class `ReversibleGenerator` (static methods).**

| Method | Signature | Description |
|--------|-----------|-------------|
| `encode` | `(x) -> Tensor` | x_t -> T3^t * x_t. Uses einsum with T3_POW[t % 168] |
| `decode` | `(y) -> Tensor` | y_t -> T3^{168-t} * y_t. Inverse evolution |
| `roundtrip_check` | `(x) -> bool` | Verifies decode(encode(x)) == x |

**Tensor Shapes.** x:(B, L, D, 3). The evolution matrices M have shape (L, 3, 3) with M[t] = T3_POW[t % 168] for encode and T3_POW[(168-t) % 168] for decode.

**Dependencies.** `constants` (P, ORD, T3_POW).

**Connection to Pipeline.** Used for time-reversible data transformations. The exact reversibility property is guaranteed by T3^{168} = I.

---

### 7.3 `rollback.py` — Quantum Rollback and Error Correction

**File:** `zeta/rollback.py`

**Purpose.** Implements algebraic error detection and correction. Detection: pred != label (mod 13). Correction: head += eta^{step} — single orbital step.

**Class `QuantumRollback` (static methods).**

| Method | Signature | Description |
|--------|-----------|-------------|
| `detect` | `(pred, label) -> bool` | (pred % 13) != (label % 13) |
| `rollback_ring` | `(head, step, dev) -> Tensor` | head + ETA_POW[step % 168], broadcast to head shape |
| `rollback_witt` | `(w, step, prec) -> Tensor` | Witt addition of teichmuller(ETA_POW[step % 168]) |
| `correction_spectrum` | `(step) -> Tensor` | NTT of the correction vector expanded to 12 elements |

**Dependencies.** `constants` (P, ORD, ETA_POW), `ring` (Z13), `witt` (WittVector), `ntt` (NTT).

**Connection to Pipeline.** `QuantumRollback.detect` is used by `AutonomousLoop` for hallucination detection. `rollback_ring` provides the simplest correction mechanism.

---

### 7.4 `counterfactual.py` — Counterfactual Trajectory Exploration

**File:** `zeta/counterfactual.py`

**Purpose.** Explores parallel trajectories in the T3 orbit space. Each branch applies a different T3^k to the same initial state. The best branch is selected via ultrametric distance.

**Class `CounterfactualBranch` (static methods).**

| Method | Signature | Description |
|--------|-----------|-------------|
| `branch` | `(state, n_branches=6) -> Tensor` | T3^k * state for k = 0..n_branches-1. Shape (k, 3) |
| `best_branch` | `(branches, goal) -> int` | Selects branch minimizing v_13 distance to goal |
| `gqm_superposition` | `(branches) -> GQMState` | Constructs quantum state from branch coefficients |

**Dependencies.** `constants` (P, ORD, T3_POW, _VAL_LUT, _VAL_MAX), `gqm` (GQMState).

**Connection to Pipeline.** Provides parallel exploration of the T3 orbit for decision-making and planning.

---

### 7.5 `ardt.py` — Algebraic Reasoning and Decision Transformer

**File:** `zeta/ardt.py`

**Purpose.** Defines the ARDT architecture: perceive -> reason -> plan -> act -> verify. Each layer uses existing algebraic modules without standalone arithmetic.

**Class `ARDTArchitecture` (static methods).**

| Method | Signature | Description |
|--------|-----------|-------------|
| `perceive` | `(tokens, model) -> Tensor` | model.pe(model.embed(tokens)) |
| `reason` | `(x, dm, layer) -> Tensor` | layer(x, dm)[0] |
| `plan` | `(state, goal) -> int` | OrbitPlanner.plan_orbit(state, goal) |
| `act` | `(model, tokens) -> Tensor` | model(tokens)[0].argmax(-1) % P |
| `verify` | `(pred, label) -> bool` | not QuantumRollback.detect(pred, label) |
| `layer4_plan` | `(i, j, dev) -> Tensor` | PAdicKernel.elem(i, j, dev) |

**Dependencies.** `kernel` (PAdicKernel), `orbit` (OrbitPlanner), `rollback` (QuantumRollback).

**Connection to Pipeline.** Provides a structured 5-step reasoning chain for interpretable decision-making.

---


## Tier 8: System and I/O

---

### 8.1 `runtime.py` — Device Manager and System Runtime

**File:** `zeta/runtime.py`

**Purpose.** Provides the single entry point for the Zeta engine. Manages device-local copies of all precomputed tables and orchestrates forward pass, training, benchmarking, and Riemann Hypothesis verification.

**Class `ZetaDevice(nn.Module)`.**

**Buffers.** All precomputed tables from `constants.py` registered as nn.Buffers:
- eta_pow, eta_ipow, t3_pow, inv_tbl, p1_mat, p23_mat, v_eig, w_eig, s3_mats, val_lut, one_r, zero_r
- For each NTT size n in [1,2,3,4,6,12]: ntt_fwd_n, ntt_inv_n

**Methods.**
- `mem_bytes() -> int` — Total memory footprint of all buffers.
- `mem_str() -> str` — Human-readable memory size in MB.

**Class `ZetaRuntime` (singleton).**

**Constructor.** `__init__(device, V, D, N, ctx)`:
- Creates `ZetaDevice()`
- Moves to device (with pin_memory for CPU if CUDA available)
- Creates `ZetaModel(V, D, N, ctx)` and moves to device

**Class Method `init(device='cpu', V=256, D=54, N=11, ctx=256, force_reinit=False) -> ZetaRuntime`:**
- Singleton pattern: returns existing instance if device matches
- Otherwise creates new instance with timing and memory reporting

**Instance Methods.**

| Method | Signature | Description |
|--------|-----------|-------------|
| `forward` | `(tokens) -> Tensor` | Model forward pass with torch.no_grad() |
| `train_step` | `(tokens, targets) -> dict` | Model train_step with torch.no_grad() |
| `encode` | `(text) -> Tensor` | tokenise(text) -> tensor -> device |
| `decode` | `(logits) -> str` | argmax -> detokenise |
| `rh_check` | `(L=12) -> dict` | AdelicDiracOperator.rh_check |
| `benchmark` | `(B=4, L=64) -> dict` | Comprehensive benchmark suite |
| `print_benchmark` | `(B=4, L=64) -> None` | Formatted benchmark output |
| `summary` | `() -> str` | System overview string |

**Benchmark Suite (`benchmark`).**
Measures: forward_ms, ring_mul_us, ring_inv_us, t3_lookup_us, kernel_tree_ms, ntt_{n}_us for n in [4,7,12,14,28], witt_add_us, witt_mul_us, delta_max_us.

**Dependencies.** `constants` (all tables), `model` (ZetaModel), `dirac` (AdelicDiracOperator), `tokenizer` (tokenise, detokenise).

**Connection to Pipeline.** This is the user-facing entry point. All operations — training, inference, verification, benchmarking — are accessed through `ZetaRuntime`.

---

### 8.2 `axioms.py` — Runtime Axiom Verification

**File:** `zeta/axioms.py`

**Purpose.** Verifies 30 runtime axioms (A001–A030) organized into 10 categories. All axioms must pass for the engine state to be valid.

**Class `AxiomVerifier` (static methods).**

**Method `verify_all(dev='cpu') -> dict`:**

| Axiom | Check |
|-------|-------|
| A001 | Ring multiplication associativity |
| A002 | Ring addition associativity |
| A003 | Ring multiplication commutativity |
| A004 | Ring addition commutativity |
| A005 | Distributivity |
| A006 | Multiplicative identity |
| A007 | Multiplicative inverse for units |
| A008 | Characteristic 13 |
| A009 | eta^3 = eta^2 + eta + 1 |
| A010 | CRT roundtrip: compose(phi1(a), phi2(a)) == a |
| A011 | phi1(1) == 1 |
| A012 | det(T3) == 1 |
| A013 | T3^168 == I |
| A014 | T3 * P1 == 7 * P1 |
| A015 | P1^2 == P1 (idempotence) |
| A016 | P23^2 == P23 |
| A017 | P1 * P23 == 0 (orthogonality) |
| A018 | P1 + P23 == I (completeness) |
| A019 | T3 * P1 == 7 * P1 (eigenvalue) |
| A020 | S3 group closure |
| A021 | J^2 == id (Galois involution) |
| A022 | Kernel diagonal: G(i,i) == 1 |
| A023 | Strong triangle inequality (random sample) |
| A024 | NTT roundtrip: INTT(NTT(x)) == x |
| A025 | NTT convolution theorem |
| A026 | Witt addition consistency |
| A027 | Witt multiplication consistency |
| A028 | Dirac antisymmetry |
| A029 | p-adic Riemann Hypothesis |
| A030 | Winding number == 14 |

**Method `print_report(dev='cpu') -> None`:** Prints pass/fail status for all axioms and summary count.

**Dependencies.** `constants` (all tables), `ring` (Z13, CRT, F169), `spectral` (SylvesterProjectors, SpectralDecomposition, S3Galois, proj), `kernel` (PAdicKernel), `ntt` (NTT), `witt` (WittVector), `dirac` (AdelicDiracOperator), `zeta_func` (SpectralFlow).

**Connection to Pipeline.** `AxiomVerifier.print_report()` is the first call after `ZetaRuntime.init()` to validate system integrity.

---

### 8.3 `sampler.py` — Buchberger Batch Accuracy Estimation

**File:** `zeta/sampler.py`

**Purpose.** Evaluates Buchberger training accuracy. Predictions and targets are compared modulo 13 (ring characteristic).

**Class `PAdicSampler` (static methods).**

| Method | Signature | Description |
|--------|-----------|-------------|
| `_groebner_is_unit` | `(pred, label) -> bool` | pred != label |
| `step` | `(model, tokens, targets) -> dict` | Delegates to model.train_step |
| `batch_accuracy` | `(model, tokens, targets) -> float` | Percentage of correct predictions mod 13 |

**Dependencies.** `constants` (P).

**Connection to Pipeline.** Used for training evaluation and metric reporting.

---

### 8.4 `tokenizer.py` — Byte, DNA Codon, and Trigram Tokenizers

**File:** `zeta/tokenizer.py`

**Purpose.** Provides three native tokenizers. All are pure Python; none is in the forward data path.

**Functions.**

| Function | Signature | Vocab | Compression | Method |
|----------|-----------|-------|-------------|--------|
| `tokenise` | `(text) -> List[int]` | 256 | 1x | UTF-8 bytes |
| `detokenise` | `(ids) -> str` | — | — | Bytes to string |
| `pad_batch` | `(seqs, pad, maxlen) -> Tensor` | — | — | Right-pad to (B, maxlen) |
| `dna_tokenise` | `(seq) -> List[int]` | 64 | 3x | 3-base codons (ACGT) |
| `trigram_tokenise` | `(text) -> List[int]` | 256 | 3x | Overlapping trigram hash |
| `detrigram_tokenise` | `(ids) -> str` | — | — | Approximate inverse (lossy) |
| `pad_batch_trigram` | `(seqs, pad, maxlen) -> Tensor` | — | — | Right-pad trigram sequences |

**Trigram Hash.** `h(b1, b2, b3) = (b1 + 7*b2 + 10*b3) % 256`. Coefficients (1, 7, 10) are the CRT-style coefficients from the dominant projection phi1.

**DNA Codon Map.** `_DNA` dictionary: 64 codons AAA..TTT mapped to integers 0..63.

**Dependencies.** `constants` (ZetaConfig, P).

**Connection to Pipeline.** `tokenise` and `trigram_tokenise` convert raw text to token IDs before model input. `pad_batch` prepares variable-length sequences for tensor operations.

---

### 8.5 `adelic.py` — Adelic Product and Strong Approximation

**File:** `zeta/adelic.py`

**Purpose.** Computes the adelic norm across primes ell != 13 and implements the strong approximation theorem for simultaneous congruences.

**Class `AdelicProduct` (static methods).**

| Method | Signature | Description |
|--------|-----------|-------------|
| `residue` | `(a, ell) -> Tensor` | a mod ell |
| `adelic_norm` | `(a) -> Tensor` | Product of local norms over primes [2, 3, 5, 7, 11] |
| `strong_approx` | `(targets) -> Tensor` | Chinese Remainder Theorem over primes 2,3,5,7,11,13. Returns (3,) tensor |

**Strong Approximation.** M = 30030 = 2*3*5*7*11*13. For each prime p, computes x += target[p] * (M/p) * inv(M/p mod p) mod M. Final result embedded as (x % 13, 0, 0).

**Dependencies.** `constants` (P), `ring` (Z13).

**Connection to Pipeline.** Provides the adelic perspective on the ring structure. The strong approximation connects the p-adic system to simultaneous congruences over multiple primes.

---

### 8.6 `__init__.py` — Package Exports and Version

**File:** `zeta/__init__.py`

**Purpose.** Package-level exports and metadata.

**Exports.**
- `ZetaConfig`, `P`, `ORD`
- `ZetaRuntime`, `ZetaDevice`
- `ZetaModel`
- `AxiomVerifier`
- `BuchbergerEngine`
- `ErrorCache`, `ZetaSelf`
- `ZetaGoalPlanner`
- `EntropyMonitor`
- `HenselIO`
- `AutonomousLoop`, `FullAutonomyCycle`
- `ZetaScaler`
- `S3ParallelAttention`
- `trigram_tokenise`, `detrigram_tokenise`, `pad_batch_trigram`

**Metadata.**
- `__version__` = "7.0.0"
- `__author__` = "Dávid Navrátil"
- `__email__` = "david.navratil2016@gmail.com"
- `__license__` = "CC-BY-NC-4.0"

**Dependencies.** All modules listed above.

**Connection to Pipeline.** This is the import interface: `from zeta import ZetaRuntime, AxiomVerifier, AutonomousLoop, trigram_tokenise`.

---

## Document Integrity Statement

This document was constructed by direct analysis of the 39 source code files comprising Zeta p-Adic Integer AI v7.0.0. Every module description, function signature, tensor shape, and mathematical formula was extracted from the actual code. No functionality was invented, no signatures were hallucinated, and no external assumptions were introduced.

**Author:** Dávid Navrátil <david.navratil2016@gmail.com>  
**License:** CC-BY-NC-4.0  
**Version:** 7.0.0
