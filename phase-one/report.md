# PS-Q2 Report: Quantum Kernel Estimation on Two Processor Topologies

## 1. Objective
Estimate the fidelity kernel `K(x,x') = |<phi(x)|phi(x')>|^2` on a 4-qubit register for `make_moons(n=100, noise=0.1, seed=42)`, train a classical SVM on it, and compare the cost of evaluating the kernel on Processor A (5-qubit chain) and Processor B (9-qubit heavy-hex-inspired graph) under identical noise.

## 2. Architecture analysis
The feature map entangles the four logical qubits in a **ring** (0-1, 1-2, 2-3, 3-0). Both fidelity circuits (`U` and `U^dagger`) contain 8 CX each, so 16 logical CX, which a compiler reduces to 14 after optimisation.

* **Processor A (chain).** A path graph has no cycle, so the closing edge (3,0) is never native. The router inserts SWAPs, raising CX from 14 to 24 on the demo circuit (+10, about 3 SWAPs at 3 CX each) and depth from 28 to 35. Averaged over 30 pairs: 23.9 CX and depth 34.8.
* **Processor B (heavy-hex-inspired).** The graph contains the 4-cycle 0-1-6-8-0. Mapping logical qubits 0-3 to physical 0, 1, 6, 8 makes all four ring edges native, so no routing is needed. The compiled circuit equals the all-to-all reference: 14 CX, depth 28 on the demo pair; averaged, 13.8 CX and depth 27.7.

The extra four physical qubits on B are not used. B's advantage here comes from topology (a cycle exists), not from qubit count.

## 3. Results
| Metric | A | B | B vs A |
|---|---|---|---|
| Mean CX | 23.9 | 13.8 | -42% |
| Mean depth | 34.8 | 27.7 | -20% |
| Mean \|noisy - ideal\| | 0.0211 | 0.0168 | -20% |
| Max \|noisy - ideal\| | 0.116 | 0.089 | -23% |
| Mean binomial SE (4096 shots) | 0.0041 | 0.0041 | equal |
| Time / evaluation (simulator) | 16.4 ms | 16.2 ms | not significant |

With identical noise rates, the gap in kernel error follows the CX count: each CX contributes 0.5% two-qubit depolarizing error, so about 10 fewer CX gives a smaller accumulated error. The shot (sampling) error is the same on both, which shows that the A-versus-B difference is a gate-noise effect, not a statistical one. Because the noisy error (about 0.02) is several times the shot error (about 0.004), gate noise rather than sampling dominates.

The full symmetric kernel needs 3,240 + 1,600 = 4,840 evaluations, or about 19.8 million shots per processor.

Ideal-kernel QSVM test accuracy is 0.95 (RBF baseline 0.90). With 20 test points, the 0.05 gap is a single sample and should not be read as a quantum advantage.

## 4. Trade-offs
* **Connectivity versus qubit count.** A 9-qubit device is only better for this task if its graph contains the circuit's interaction graph. A denser chain-like device with 9 qubits would not help, since the ring still needs a cycle.
* **Hand-chosen layouts.** Both layouts were chosen by inspection. A chain offers no better placement for a ring, but B's placement is the best possible (zero overhead).
* **Cost metrics.** On real hardware, the lower CX count and depth on B would reduce error accumulation and decoherence exposure. On a simulator, wall-clock time reflects simulation effort, not hardware time.

## 5. Limitations
1. **Simulator timing is noisy.** Time per evaluation was 13-18 ms across repeated runs, with either processor slightly ahead. Only the circuit metrics (CX, depth) and error are reliable comparisons.
2. **Identical noise by design.** The problem fixes equal rates, so effects of connectivity-dependent error (crosstalk, per-edge error variation) are not modelled.
3. **Estimated SWAP count.** SWAPs are inferred as routing CX overhead divided by 3, since the final circuit is in the {rz, sx, x, cx} basis. This is an estimate, not a direct count.
4. **Small sample.** The cost benchmark uses 30 random pairs, a single seed, and one data split. Test accuracy is on 20 points.
5. **SVM trained on the ideal kernel.** The noisy kernel is evaluated for cost and error but not used to train the SVM in the final pipeline.
6. **Feature map is not tuned.** Hyperparameters (C, feature scaling, entangler depth) were not optimised, and no claim of quantum advantage is made.
