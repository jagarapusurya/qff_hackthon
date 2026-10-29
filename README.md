# QFF2026 QML Track — PS-Q2 submission (noisy ZZ-ring quantum-kernel SVM)

Selected problem: **PS-Q2** (high-dimensional synthetic data, metric: balanced accuracy), plus Task A (Bell state) and Task B (Processor D).
Team **MGM**. Report: [results.md](results.md). Organizer kit README: [KIT_README.md](KIT_README.md). The folder `phase-one/` holds our earlier, separate phase-one project and is not part of this submission.

## Deliverables
| File | What it is |
|------|------------|
| `solution.ipynb` | End-to-end notebook (Task A, Task B, PS-Q2 model, metrics, predictions) |
| `predictions_q2.csv` | **Submitted predictions.** Noisy QFF2026-v2 ZZ-ring kernel SVM, trained on a fixed stratified 300-sample subset |
| `predictions_q2_ideal_n800.csv` | Comparison only: the same kernel in ideal simulation, trained on all 800 samples |
| `processor_d_config.json` | Processor D (6-qubit 2×3 grid, two-way links) with official `benchmark_results` |
| `results.md` | Results report |

## Model in one paragraph
StandardScaler → PCA(6) → unit-variance rescale (fitted on training data only) → 6-qubit ZZ feature map (H, RZ(s·z), RZZ(s²·z_i·z_j) on a ring, s = 0.35, 1 layer) → fidelity kernel |⟨φ(x′)|φ(x)⟩|², estimated as P(0…0) of the compute–uncompute circuit under QFF2026-v2. Circuits are transpiled to rx/ry/rz/cx, 4096 shots → precomputed-kernel SVM, C = 1. No noise mitigation. The signal lost to noise is reported in `results.md` as a negative result.

## Run instructions (Windows PowerShell; Python 3.11–3.13)
```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt

python -m nbconvert --to notebook --execute --inplace solution.ipynb   # or open it in Jupyter and Run All
python tools/validate_predictions.py q2 predictions_q2.csv
python benchmark/qff_qml_benchmark.py processor_d_config.json
```
The notebook reuses the saved noisy-kernel checkpoints in `results/tables/`, so it finishes in minutes. To recompute the expensive noisy parts from scratch, set the flags in its first code cell. Or run the scripts directly from the project root:

| Step | Script | Cost |
|------|--------|------|
| Classical baselines (204 configs) | `python run_step2_baselines.py` | minutes |
| Quantum feature-map ablation (168 configs, ideal) | `python run_step4_ablation.py` | minutes |
| 120-sample ideal vs noisy study | `python run_step6_noisy_kernel.py zz ring 6 1 0.35 zz_ring_k6_r1` then `python run_step6_analysis.py zz_ring_k6_r1 6` | 7,260 circuits |
| **Submitted predictions** (noisy, resumable) | `python run_step6_noisy_test.py` | 105,150 circuits, ≈1.4 h |
| Noisy 5-fold CV metrics | `python run_step7_noisy_cv_metrics.py` | 216,600 circuits, ≈5 h |
| Classical 5-fold CV metrics (same folds) | `python run_step7_classical_cv_metrics.py` | seconds |
| Ideal model on all 800 (comparison predictions) | `python run_step8_ideal_predictions.py` | ~1 min |
| Ideal vs noisy vs tuned classical metrics (nested CV), bootstrap CIs, signal loss | `python run_step9_kernel_metrics.py` | ~1 min |
| Processor D topology sweep | `python run_processor_d_sweep.py` | minutes |
| Processor D one-way vs two-way links, on QFT-3 and our circuit | `python run_processor_d_direction.py` | seconds |

All seeds are 42. The 300-sample subset is defined once in `src/q2_data.stratified_subset`. Test labels are never used.

## Key numbers (balanced accuracy, out-of-fold CV on training data)
| Model (300-sample subset, identical folds) | Balanced acc. [95% bootstrap CI] |
|---|---|
| Quantum kernel, ideal | 0.703 [0.650, 0.756] |
| Quantum kernel, noisy QFF2026-v2 (**submitted model**) | 0.687 [0.633, 0.741] |
| Classical RBF-SVM, tuned in each training fold | 0.693 [0.640, 0.745] |
| Classical poly-SVM, tuned in each training fold | 0.717 [0.666, 0.769] |

Noise removes 72% of the kernel contrast and costs ≈1.6 points. No quantum advantage is shown; this negative result is analysed in `results.md`.

Task A noisy Bell fidelity: 0.880 ± 0.003. Task B noisy QFT-3 fidelity: 0.627.

## `pip freeze` (environment used for all results)
```
annotated-types==0.8.0
anyio==4.15.1
appdirs==1.4.4
argon2-cffi==25.1.0
argon2-cffi-bindings==26.1.0
arrow==1.4.0
asttokens==3.0.2
astunparse==1.6.3
async-lru==2.4.0
attrs==26.1.0
autograd==1.9.1
autoray==0.8.2
babel==2.18.0
beautifulsoup4==4.15.0
bleach==6.4.0
cachetools==7.2.7
certifi==2026.10.23
cffi==2.1.1
charset-normalizer==3.5.2
cloudpickle==3.1.2
colorama==0.4.6
comm==0.2.3
contourpy==1.4.0
cryptography==50.0.2
cycler==0.12.1
debugpy==1.8.22
defusedxml==0.7.1
diastatic-malt==2.15.2
dill==0.4.1
executing==2.3.0
fastjsonschema==2.22.2
fonttools==4.68.0
fqdn==1.6.0
gast==0.7.0
h11==0.16.0
httpcore==1.0.9
httpx==0.28.1
ibm-cloud-sdk-core==3.28.0
ibm-platform-services==0.77.1
idna==3.20
ipykernel==7.4.0
ipython==9.17.1
ipython_pygments_lexers==1.1.1
isoduration==20.11.0
jedi==0.20.1
Jinja2==3.1.6
joblib==1.6.0
json5==0.17.3
jsonpointer==3.6
jsonschema==4.26.0
jsonschema-specifications==2025.9.1
jupyter-events==0.12.1
jupyter-lsp==2.3.1
jupyter_builder==1.2.4
jupyter_client==8.10.0
jupyter_core==5.9.1
jupyter_server==2.21.1
jupyter_server_terminals==0.5.4
jupyterlab==4.6.4
jupyterlab_pygments==0.3.0
jupyterlab_server==2.28.1
kiwisolver==1.5.1
lark==1.3.1
MarkupSafe==3.0.4
matplotlib==3.11.2
matplotlib-inline==0.2.2
mistune==3.4.0
mpmath==1.3.0
narwhals==2.28.0
nbclient==0.11.0
nbconvert==7.17.2
nbformat==5.11.1
nest-asyncio2==1.7.4
networkx==3.7
notebook==7.6.3
notebook_shim==0.2.4
numpy==2.4.6
packaging==26.3
pandas==3.1.0
pandocfilters==1.5.1
parso==0.8.7
pennylane==0.44.1
PennyLane-qiskit==0.44.1
pennylane_lightning==0.45.0
pillow==13.0.0
platformdirs==4.14.0
prometheus_client==0.26.0
prompt_toolkit==3.0.53
psutil==7.2.2
pure_eval==0.2.4
pycparser==3.12
pydantic==2.14.1
pydantic_core==2.50.1
Pygments==2.21.0
PyJWT==2.15.1
pyparsing==3.3.3
pyspnego==0.12.4
python-dateutil==2.9.0.post0
python-json-logger==4.2.0
pywinpty==3.1.2
PyYAML==6.0.3
pyzmq==27.2.0
qiskit==2.3.0
qiskit-aer==0.17.2
qiskit-ibm-runtime==0.45.1
referencing==0.37.0
requests==2.34.2
requests_ntlm==1.3.0
rfc3339-validator==0.1.4
rfc3986-validator==0.1.1
rfc3987-syntax==1.1.0
rpds-py==2026.9.1
rustworkx==0.18.1
scikit-learn==1.9.1
scipy==1.17.1
scipy-openblas32==0.3.34.237.2
Send2Trash==2.1.0
six==1.17.0
soupsieve==3.0.4
sspilib==0.6.0
stack-data==0.6.3
stevedore==5.9.1
sympy==1.14.0
termcolor==3.3.0
terminado==0.18.1
threadpoolctl==3.7.0
tinycss2==1.5.1
tomlkit==0.15.2
tornado==6.6
traitlets==5.16.1
typing-inspection==0.4.4
typing_extensions==4.16.0
tzdata==2026.6
uri-template==1.3.0
urllib3==2.8.0
wcwidth==0.9.4
webcolors==25.10.0
webencodings==0.6.1
websocket-client==1.9.2
wheel==0.48.0
```
