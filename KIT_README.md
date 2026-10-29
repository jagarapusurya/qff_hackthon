# QFF2026 QML Challenge Kit — v3

**Qiskit Fall Fest 2026 — Quantum Machine Learning Track**

The participant problem statement is **QFF2026_QML_Participant_Problem_Statements_v3.docx**. It is the only authoritative description of the tasks. Earlier copies (the `_v2.md` file and earlier versions of the docx) are withdrawn.

## Deadline and channels

| What | Where / when |
|------|--------------|
| Hackathon window | Day 1, 6:00 PM IST → Day 2, 6:00 PM IST |
| **Final submission** | **Day 2, 5:40 PM IST** — public GitHub repository URL on the hackathon submission portal: https://qffsrmap2026.com/learning/hackathon |
| Questions | Discord **#qml-track**, or the help desk in the venue. Discord is **not** a submission channel |

## Quick start (dedicated environment)

The QML track uses its own environment. Do **not** install it into the Optimization-track environment.

```bash
python3 -m venv qml-env            # Python 3.11, 3.12 or 3.13
source qml-env/bin/activate        # Windows: qml-env\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python noise_model/benchmark_noise_model.py                       # prints the noise parameters
python benchmark/qff_qml_benchmark.py submission_templates/processor_d_example.json   # Task B self-test
jupyter notebook starter_kit/starter_kit.ipynb
```

## Contents

| Path | Description |
|------|-------------|
| data/q1/ | ECG heartbeat features (PS-Q1) |
| data/q2/ | High-dimensional synthetic dataset (PS-Q2) |
| data/q3/ | Quantum-noise benchmark dataset (PS-Q3) |
| schemas/ | Feature dictionaries + Processor D JSON schema |
| noise_model/ | `benchmark_noise_model.py` — organizer noise model **QFF2026-v2**, used for ALL noisy runs |
| benchmark/ | `qff_qml_benchmark.py` — official Task A and Task B measurement code |
| tools/ | `validate_predictions.py` — check your predictions file before submitting |
| starter_kit/ | Scaffold notebook |
| submission_templates/ | results_template.md, predictions_q1/q2/q3_format.csv, processor_d_example.json |
| CHANGELOG.md | Everything that changed from the earlier kit |

Test labels are **not** in this kit. Test-feature files start with a `sample_id` column (s0, s1, …); it is an identifier, not a feature.

## Rules

- Choose exactly ONE PS (Q1, Q2 or Q3) and complete both common tasks (Task A: Bell state, Task B: Processor D).
- Never train or tune on test data. Fit scalers/PCA on training data only; use validation data for model selection only.
- Every noisy run uses `noise_model/benchmark_noise_model.py` unchanged, on circuits transpiled to a fully-noised basis (`prepare_for_noisy_run()` / `run_noisy()` / `pennylane_noisy_device()`).
- Task A and Task B numbers come from `benchmark/qff_qml_benchmark.py`.
- Set all random seeds. The notebook must run end to end without errors in the pinned environment.

## Submission checklist

- [ ] `solution.ipynb` runs top to bottom
- [ ] `processor_d_config.json` passes `python benchmark/qff_qml_benchmark.py processor_d_config.json`, with `benchmark_results` filled in
- [ ] `predictions_q1.csv` / `predictions_q2.csv` / `predictions_q3.csv` passes `python tools/validate_predictions.py <q1|q2|q3> <file>`
- [ ] `results.md` completed from the template (fidelity definitions included)
- [ ] `README.md` with run instructions and `pip freeze` output
- [ ] Repository URL submitted on the portal before 5:40 PM IST on Day 2
