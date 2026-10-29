# Project status and method notes

Status (2026-10-29): **all experiments complete; submission files filled in.**

- Submitted predictions: `predictions_q2.csv`, from the noisy QFF2026-v2 ZZ-ring kernel SVM (k=6, reps=1, s=0.35, C=1, 4096 shots). It is trained on the fixed stratified 300-sample subset and took 105,150 circuits. Checkpoints in `results/tables/K_noisy_{train,test}_zz_ring_k6_r1_n300*.npy` are complete. Re-running `run_step6_noisy_test.py` reproduces the file byte for byte without new circuits.
- Comparison predictions: `predictions_q2_ideal_n800.csv` (ideal simulation, all 800 samples). They agree with the submitted file on 88% of samples.
- Evaluation metrics: `run_step7_noisy_cv_metrics.py` (noisy, 216,600 circuits), `run_step7_classical_cv_metrics.py` and `run_step9_kernel_metrics.py` (ideal vs noisy vs classical on identical folds, plus kernel signal loss). Balanced accuracy on the 300 subset: ideal 0.703, noisy 0.687, classical RBF 0.683.
- Noise is **not mitigated**. The signal loss (kernel contrast −72%, paired −1.7 points balanced accuracy) is reported as a negative result in `results.md`.
- The 300-sample subset is defined once in `src/q2_data.stratified_subset` and used by every n=300 script. All saved n=300 outputs were verified to use this subset.
- Processor D: 6-qubit 2×3 grid with two-way links, chosen by `run_processor_d_sweep.py`. Official benchmark reproduced: noisy QFT-3 fidelity 0.6268.
- Task A: noisy Bell fidelity 0.8801 ± 0.0026 (`results/tables/task_a_bell.json`).
- Still to do by hand: team name in `results.md`; push to a public GitHub repo and submit the URL before Day 2, 5:40 PM IST.
