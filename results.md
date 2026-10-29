# QFF2026 QML Track — Results Report

**Team name:** _MGM_
**Selected PS:** [ ] PS-Q1  [x] PS-Q2  [ ] PS-Q3
**Date:** 2026-10-29
**Dependency versions:** numpy==2.4.6, pandas==3.1.0, pennylane==0.44.1, PennyLane-qiskit==0.44.1, qiskit==2.3.0, qiskit-aer==0.17.2, scikit-learn==1.9.1, scipy==1.17.1 (Python 3.12.9; full `pip freeze` in `README.md`)
**Noise model version:** QFF2026-v2 (unchanged)

**Summary.** We built a 6-qubit ZZ-ring fidelity-kernel SVM. Its noisy (QFF2026-v2) kernel scores **0.687** balanced accuracy, 95% CI [0.633, 0.741], on fold-correct CV. That matches a tuned classical RBF-SVM on the same folds (0.693) but is below a tuned polynomial SVM (0.717). Noise removes 72% of the kernel's contrast and costs about 1.6 points. This is a documented **negative result**: we found no quantum advantage on this data. The submitted file is `predictions_q2.csv`.

---

## 1. Approach Summary

**Encoding (fitted on train only).** The 50 features are standardised and reduced to 6 PCA components, then each component is rescaled to unit variance. The scaler, PCA and rescale are fitted on the training rows only, and refitted inside every CV training fold, so test rows are only ever transformed. We chose PCA-6 from data: in the classical grid PCA-6 was the best dimension for RBF-SVM (0.728, vs 0.721 for all 50 features, 0.682 for PCA-4, 0.726 for PCA-10). In the quantum ablation, k = 6 qubits beat k = 4 (0.684) and k = 8 (0.716). The unit-variance rescale makes one bandwidth s act equally on every qubit.

**Feature map (own design, not a copied template).** For each layer we apply H on every qubit, then RZ(s·z_i), then RZZ(s²·z_i·z_j) on the ring edges (i, i+1 mod 6). It uses one layer with s = 0.35. The circuit family is IQP/ZZ-style, but we chose the details ourselves:
- **Angles:** s·z and s²·z_i·z_j, with an explicit bandwidth s, instead of the textbook (π − x_i)(π − x_j) angles.
- **Ring entanglement:** we chose a ring to match the hardware we designed for Task B.
- **One layer:** two layers lowered CV accuracy (0.717 vs 0.729) and concentrated the kernel.

The map is defined once as an op-list in `src/qkernel.py`. That one definition drives a batched numpy statevector simulator (ideal kernels) and the Qiskit circuits (noisy kernels). The two agree with `qiskit.quantum_info.Statevector` to machine precision for every map we tried.

**Kernel evaluation.** Each kernel entry K(x, x′) = |⟨φ(x′)|φ(x)⟩|² is the probability of measuring 0…0 on the compute–uncompute circuit U(x′)†U(x). A barrier stops the transpiler from cancelling U against U†. Noisy kernels use circuits transpiled to {rx, ry, rz, cx} (optimisation level 1, seed 42), checked with `assert_all_gates_noisy`, and run on the Aer density-matrix simulator with QFF2026-v2 and 4096 shots. The diagonal is measured like every other entry. A precomputed-kernel SVM with C = 1 does the classification. C = 1 was the best C for the noisy kernel in the 120-sample study and was then frozen.

**Small, documented tuning.** We searched 168 quantum configurations with fold-correct 5×2 CV on all 800 training rows: 4 maps × k ∈ {4, 6, 8} × reps ∈ {1, 2} × 7 bandwidths (`run_step4_ablation.py`). ZZ-ring with k = 6, s = 0.35, one layer scored 0.729, within 0.002 of the best configuration. The best overall was the non-entangling product-map control (0.730).

**Subsetting (feasibility).** A noisy 800-sample model would need 320,400 training circuits plus 160,000 test circuits. We instead train the submitted noisy model on a fixed, stratified 300-sample subset (150 A / 150 B, seed 42, `src/q2_data.stratified_subset`). That costs 45,150 + 60,000 = **105,150** circuits. The ideal model is also trained on all 800 samples (`predictions_q2_ideal_n800.csv`, comparison only), to measure the cost of subsetting.

**No noise mitigation is applied.** The signal lost to noise is measured and reported as a negative result (§2.4).

---

## 2. Results

Official metric: **balanced accuracy**. PS-Q2 has no validation file, so out-of-fold CV on the training data is our validation. Test labels were never used. "Same folds" rows use the identical 300 rows and `StratifiedKFold(5, shuffle=True, random_state=42)`, so they are paired. CIs are 95% bootstrap percentile intervals over the 300 out-of-fold predictions (2,000 resamples, seed 42). Source: `results/tables/kernel_metrics_zz_ring_k6_r1_n300.{csv,json}` (`run_step9_kernel_metrics.py`).

### 2.1 Classical baseline (same data, tuned on training folds only)
| Model | Balanced acc. [95% CI] | Notes |
|-------|------------------------|-------|
| RBF-SVM, PCA-6, **tuned in each training fold** (nested CV) — 300, same folds | **0.693** [0.640, 0.745] | Inner 5-fold grid over C × γ (25 configs). Chosen per fold: (3, 0.01), (30, 0.01), (0.3, 0.1), (1, 0.03), (30, 0.003). ROC-AUC 0.755 |
| Poly-SVM, PCA-6, **tuned in each training fold** (nested CV) — 300, same folds | **0.717** [0.666, 0.769] | Degree 2 chosen in every fold. ROC-AUC 0.771 |
| RBF-SVM, PCA-6, C = 1, γ = scale (untuned) — 300, same folds | 0.683 [0.627, 0.736] | `run_step7_classical_cv_metrics.py` |
| Best tuned RBF-SVM (PCA-6, C = 1, γ = 0.03) — all 800, 5×3 repeated CV | 0.728 ± 0.023 | `run_step2_baselines.py`, 204-config grid |
| Organizer-style RBF-SVM (all 50 standardised features, C = 1, γ = scale) — all 800, 5×3 | 0.718 ± 0.030 | Same model family as the organizer's m_baseline |

### 2.2 Quantum model (ZZ-ring, k = 6, reps = 1, s = 0.35, C = 1)
| Configuration | Ideal | Noisy (QFF2026-v2) | Notes |
|---------------|-------|--------------------|-------|
| 300 subset, same folds, fold-correct encoding | **0.703** [0.650, 0.756] | **0.687** [0.633, 0.741] | Noisy CV: 216,600 circuits (`run_step7_noisy_cv_metrics.py`) |
| …precision(B) / recall(B) / F1(B) | 0.716 / 0.673 / 0.694 | 0.687 / 0.687 / 0.687 | Noisy confusion matrix [[103, 47], [47, 103]] |
| …ROC-AUC | 0.763 | — | The noisy run stored hard labels only |
| 120 subset, 5×6 repeated CV, C tuned per kernel | 0.678 ± 0.101 | 0.635 ± 0.109 | Paired noisy − ideal: −0.043 ± 0.013 (SE), 30 folds |
| All 800, 5-fold (ideal only) | 0.737 ± 0.026 | not run (≈480k circuits) | ROC-AUC 0.808 |

### 2.3 Paired comparisons (same 300 rows, same folds; bootstrap of differences)
| Comparison | Δ balanced acc. | 95% CI | Reading |
|------------|-----------------|--------|---------|
| Noisy − ideal quantum | −0.016 | [−0.048, +0.016] | Noise hurts in 4 of 5 folds; P(Δ ≤ 0) = 0.85 |
| Noisy quantum − tuned RBF | −0.007 | [−0.041, +0.027] | Indistinguishable |
| Ideal quantum − tuned RBF | +0.010 | [−0.017, +0.038] | Indistinguishable |
| Noisy quantum − tuned poly | −0.030 | [−0.068, +0.008] | Classical poly is better; P(Δ ≤ 0) = 0.94 |
| Ideal quantum − tuned poly | −0.014 | [−0.047, +0.019] | Classical poly is better even without noise |

### 2.4 How noise changes the kernel matrix (negative result, not mitigated)
We compared the saved 300×300 noisy training kernel with the exact ideal kernel on the same points:

| Quantity | Ideal | Noisy | Change |
|----------|-------|-------|--------|
| Mean diagonal K(x, x) | 1.000 | 0.317 | −68%: the self-overlap circuit returns to \|0…0⟩ only 32% of the time |
| Off-diagonal mean | 0.672 | 0.228 | −66% |
| Off-diagonal std (usable contrast) | 0.147 | 0.042 | **−72%** |
| Affine fit noisy ≈ a·ideal + b | — | a = 0.280, b = 0.040, R² = 0.976 | Mostly a global contraction |
| Spearman rank correlation | — | 0.986 | Ordering of similarities is kept |
| Residual after affine fit (relative) | — | 15.6% | The information actually lost |
| Kernel–target alignment | 0.0126 | 0.0101 | −20% |

Figure: `results/figures/noise_effect_zz_ring_k6_r1_n300.png`.

**Why.** Each kernel circuit has 24 CX at depth 41, with p₂ = 0.05 depolarising error per CX, p₁ = 0.01 per single-qubit gate and 2% readout error. The circuit has 24 CX, 36 single-qubit gates and 6 readouts, so the probability of no error at all is roughly 0.95²⁴ · 0.99³⁶ · 0.98⁶ ≈ 0.18. Most of the noise therefore acts like global depolarisation: it pulls every kernel entry toward the same constant. The SVM is robust to most of that. A constant shift in the kernel leaves the SVM decision unchanged (the dual constraint Σα_i·y_i = 0), and scaling the kernel is the same as rescaling C. What the SVM cannot absorb is the 15.6% non-affine residual. It comes from gate-position-dependent noise plus shot noise: the binomial SE at K ≈ 0.23 with 4096 shots is ≈ 0.007, against a contrast of only 0.042. That residual costs about 1.6 points of balanced accuracy on paired folds, and 4.3 points in the 30-fold 120-sample study.

**Noise versus kernel concentration.** The ideal kernel at bandwidth s = 1.5 has almost the same off-diagonal std (0.041) as our noisy kernel (0.042), yet only reaches 0.554 CV accuracy, against 0.687 for the noisy kernel. Low contrast alone is therefore not the problem. Bandwidth concentration scrambles the ordering of similarities, while hardware noise mostly keeps it (Spearman 0.986).

### 2.5 Kernel-concentration and ablation evidence (why it works / where it fails)
ZZ-ring, k = 6, reps = 1, fold-correct 5×2 CV on all 800 training samples (`results/tables/quantum_ablation_grid.csv`):

| s | 0.10 | 0.20 | **0.35** | 0.50 | 0.75 | 1.00 | 1.50 |
|---|------|------|----------|------|------|------|------|
| CV balanced acc. | 0.713 | 0.723 | **0.729** | 0.724 | 0.699 | 0.653 | 0.554 |
| Off-diagonal std of K | 0.017 | 0.062 | **0.148** | 0.196 | 0.158 | 0.095 | 0.041 |

- **Small s:** the kernel is almost constant, nearly linear in z, and needs C = 30.
- **Large s:** the kernel concentrates (exponentially small overlaps) and accuracy collapses to near chance.
- **s = 0.35:** the kernel is in the useful regime between these.
- **Other maps, at the same k = 6, s = 0.35, reps = 1:** product (no entanglement) 0.727, ZZ-all-to-all 0.722, our RY/RZ re-uploading map with CX ring 0.724.
- **Two layers** reduce contrast (0.126) and accuracy (0.717).

**Conclusion:** at the best bandwidth, entanglement adds nothing measurable. The useful part of this kernel is the smooth, bandwidth-controlled similarity on the 6 PCA components, which classical kernels capture equally well.

### 2.6 Hidden-test expectation and validation–test consistency
The objective formula is test_points = 10 × clip((m − 0.5)/(0.654 − 0.5), 0, 1), where 0.654 is the organizer's RBF-SVM baseline on the hidden test.
- Our CV estimate for the submitted noisy model is 0.687, which would earn the full 10 points.
- Our CV of the organizer-style model (RBF, all 50 standardised features) is 0.718, versus the published hidden-test 0.654. That suggests the hidden test is about 0.06 harder than training CV.
- Applying the same shift gives an expected hidden-test balanced accuracy of about 0.62–0.69 for `predictions_q2.csv`, or roughly 8–10 points.

The ideal 800-sample predictions agree with the submitted file on 88% of test samples.

### 2.7 Noise mitigation (optional for PS-Q2)
| Strategy | Mitigated metric | Extra circuit evals | Justified? |
|----------|-----------------|---------------------|-----------|
| None | — | 0 | Deliberately not applied. The contraction is 97.6% affine, so the SVM absorbs most of it. The remaining loss is reported as-is (§2.4) |

---

## 3. Task A — Bell state |Φ+⟩ (required)

Produced by `task_a_bell()` in `benchmark/qff_qml_benchmark.py`, 8192 shots per basis, on circuits transpiled to {rx, ry, rz, cx}. Source: `results/tables/task_a_bell.json`; re-run in `solution.ipynb`.

**Fidelity definition used:** F = (1 + ⟨XX⟩ − ⟨YY⟩ + ⟨ZZ⟩) / 4, with ⟨P⟩ = (N_even − N_odd)/N, SE(⟨P⟩) = √((1 − ⟨P⟩²)/N) and SE(F) = √(SE_XX² + SE_YY² + SE_ZZ²)/4.

| Condition | ⟨XX⟩ ± SE | ⟨YY⟩ ± SE | ⟨ZZ⟩ ± SE | F ± SE |
|-----------|-----------|-----------|-----------|--------|
| Ideal | 1.0000 ± 0.0000 | −1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| Noisy (benchmark) | 0.8191 ± 0.0063 | −0.8279 ± 0.0062 | 0.8735 ± 0.0054 | **0.8801 ± 0.0026** |

All three correlators shrink. ⟨ZZ⟩ (0.874) stays larger than |⟨XX⟩| and |⟨YY⟩| (≈ 0.82), because the XX and YY measurements need extra basis-change rotations, each carrying 1% depolarising error. ZZ is affected only by the CX, the state-preparation rotations and readout. F = 0.880 > 0.5, so the noisy state is still certified entangled. The 12% fidelity loss comes mostly from the single CX (p₂ = 0.05) and the 2% readout flips on each qubit. The same mechanism, compounded over 24 CX, is what shrinks our kernel diagonal to 0.317.

---

## 4. Processor D Benchmark (Task B)

Produced by `python benchmark/qff_qml_benchmark.py processor_d_config.json`. The config validates and its `benchmark_results` equal the official output.

**QFT-3 fidelity definition used:** mean over the 8 computational-basis inputs |j⟩ of F_j = ⟨ψ_j|ρ_j|ψ_j⟩, where ψ_j = QFT|j⟩ (ideal) and ρ_j is the density matrix of the 3 logical qubits after the transpiled circuit runs under the benchmark noise model (gate noise; no measurement).

**Processor ID:** qff26_procD_grid2x3_6q_v1  **n_qubits:** 6  **Topology:** 2×3 grid (qubits 0-1-2 / 3-4-5, rungs 0-3, 1-4, 2-5), native gates rx, ry, rz, cx

| Benchmark | Value |
|-----------|-------|
| QFT-3 depth (before transpile) | 6 |
| QFT-3 depth (after transpile) | 21 |
| Two-qubit gate count | 12 |
| SWAP overhead | 1 |
| Fidelity (ideal) | 1.000 |
| Fidelity (noisy) | 0.6268 |
| Coupling map direction (one-way / two-way links) | two-way (7 links, 14 directed edges) |

**Why this processor fits our circuit.** We compared four topologies on the official QFT-3 benchmark and on our own Q2 kernel circuit (`run_processor_d_sweep.py`, `processor_d_topology_sweep.csv`):

| Topology (two-way) | QFT-3 depth | QFT-3 noisy F | Q2 kernel circuit CX | Q2 kernel circuit depth |
|--------------------|-------------|---------------|----------------------|-------------------------|
| Linear chain | 22 | 0.604 | 46 | 59 |
| Ring | 22 | 0.604 | 24 | 41 |
| Star | 22 | 0.604 | 36 | 57 |
| **2×3 grid (chosen)** | **21** | **0.627** | **24** | **41** |

Our kernel's entanglers form a logical 6-ring. The grid contains a 6-cycle, so the transpiler maps logical qubits 0…5 to physical 1, 0, 3, 4, 5, 2. Every RZZ then lands on a physical link and **no SWAPs are needed**: 24 CX at depth 41, the same as all-to-all connectivity. A linear chain needs 22 extra CX for routing (46), which at p₂ = 0.05 would multiply the no-error probability by about 0.95²² ≈ 0.32.

**Directed links, SWAPs, depth and the trade-off** (`run_processor_d_direction.py`, `processor_d_direction.json`):

| Same 2×3 grid | QFT-3 depth | QFT-3 noisy F | Q2 kernel circuit depth | Q2 kernel circuit ops |
|---------------|-------------|---------------|-------------------------|-----------------------|
| **Two-way links (14 directed edges)** | **21** | **0.627** | **41** | **66** |
| One-way links (7 directed edges) | 31 | 0.585 | 51 | 80 |

- **Directed links:** the benchmark enforces direction. A CX against a one-way link is reversed with extra single-qubit rotations. That adds no CX, but it adds depth (+10 layers on both circuits) and noisy 1-qubit gates, costing 4.2 points of QFT fidelity. Two-way links remove that cost.
- **SWAPs:** QFT-3 needs all three qubit pairs to interact, and a grid has no triangles, so one SWAP (3 CX) remains on every topology tested. Our ring circuit needs none.
- **Trade-off:** the grid uses 7 couplers versus 6 for a ring. That is more hardware and more crosstalk exposure. For our own circuit it gives no gain over a ring; it is chosen for the slightly better QFT-3 routing (+0.023 fidelity). If only the Q2 kernel mattered, a two-way ring would be equally good and cheaper.

---

## 5. Resource Costs (kernel cost, requirement 4)
| Resource | Value |
|----------|-------|
| Qubits used | 6 |
| Circuit depth | 41 per kernel entry (compute–uncompute, transpiled to rx/ry/rz/cx). A single feature map U(x): depth 20, 12 CX |
| Two-qubit gates | **24 CX** per kernel entry (6 RZZ × 2 CX × 2 for U and U†); 66 ops total (24 rz, 24 cx, 12 ry, 6 measure) |
| Trainable parameters | 0 quantum parameters (fixed feature map). Hyperparameters s = 0.35, C = 1. The SVM dual coefficients (≤ 300) are classical |
| Dimensionality reduction | StandardScaler + PCA(50 → 6) + unit-variance rescale, fitted on training data only |
| Subsetting | Noisy model trained on 300 of 800 training rows (stratified, seed 42). Full cost would be n(n+1)/2 + 200·n = 480,400 circuits for n = 800 |
| Kernel evaluations, submitted model | 300·301/2 = 45,150 (train, symmetric, diagonal measured) + 200 × 300 = 60,000 (test) = **105,150** circuits × 4096 shots |
| Kernel evaluations, noisy CV | 5 × (240·241/2 + 60·240) = **216,600** circuits × 4096 shots |
| Kernel evaluations, 120-sample study | 120·121/2 = 7,260 |
| Total noisy circuits | **329,010** (≈ 1.35 × 10⁹ shots) |
| Training time | Noisy kernels for predictions ≈ 1.4 h; noisy 5-fold CV 18,129 s ≈ 5.0 h (Aer density matrix, laptop CPU). Ideal kernels: seconds |

---

## 6. What the results do and do not demonstrate (honest assessment)
**They demonstrate:**
- A 6-qubit fidelity kernel run end to end under the official noise model. It reaches 0.687 balanced accuracy, statistically level with a tuned classical RBF-SVM on the same data (Δ = −0.007, CI [−0.041, +0.027]).
- Hardware-like noise removes 72% of the kernel's contrast, but 97.6% of the damage is affine. Because of that, an SVM loses only about 1.6 points, not most of its accuracy.
- Kernel concentration from too large a bandwidth is far more harmful than the benchmark noise at equal contrast (0.554 vs 0.687).

**They do not demonstrate:**
- **Any quantum advantage.** A tuned degree-2 polynomial SVM (0.717) beats the noisy quantum kernel (Δ = −0.030, P = 0.94) and the ideal one. The non-entangling product map does as well as any entangling map. Performance is set by the PCA-6 representation and the bandwidth, not by entanglement.
- **That noise is harmless.** The loss is consistent in sign (4 of 5 folds; −4.3 points in the 30-fold 120-sample study), and we did not mitigate it.

**Other limitations:**
- **Subset size costs more than noise.** Training on 300 instead of 800 rows costs about 3 points (ideal kernel 0.703 → 0.737), roughly twice the noise loss.
- **Class balance of the submission.** It predicts 121 A / 79 B, while the training data is 386 A / 414 B.
- **Encoding differs between CV and the final model.** In CV, PCA is fitted on each fold's 240 rows. The final model's PCA is fitted on all 800 training rows (unsupervised; no test data).
- **Wide CIs.** Only 5 folds were affordable for the noisy CV, so the 95% intervals are about ±5 points and all quantum-vs-classical differences are within noise.

---

## 7. Reproducibility
```python
SEED = 42          # subset RNG, PCA, StratifiedKFold, bootstrap, transpiler (seed_transpiler), Aer seed_simulator
SHOTS = 4096       # kernel circuits; Task A uses 8192 (official benchmark)
NOISE = "QFF2026-v2"   # noise_model/benchmark_noise_model.py, unchanged
```
`solution.ipynb` runs top to bottom in minutes: it reuses the committed noisy-kernel checkpoints and rebuilds `predictions_q2.csv` byte for byte without new circuits. Flags in its first cell recompute everything from scratch. The 300-sample subset is defined once in `src/q2_data.stratified_subset`. The full run order is in `README.md`.

---

## 8. Optional Extensions
[ ] Second mitigation strategy (PS-Q3)
[x] Ablation studies — 168 quantum feature-map configs (4 maps incl. a no-entanglement control and our own re-uploading map, bandwidth/qubit/depth sweeps) and 204 classical configs
[x] Other: paired ideal-vs-noisy kernel analysis on identical folds with bootstrap CIs; noise-vs-concentration comparison at equal contrast; Processor D topology sweep and one-way vs two-way link study measured on our own circuit
