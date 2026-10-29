# Changes in kit v3 (replaces all earlier kits)

| Area | Before | Now (v3) |
|------|--------|----------|
| Test labels | An `organizer_answer_keys/` folder was included by mistake | Removed. Test labels are held only by the organizers |
| Problem statement | `.docx` and an older `_v2.md` that disagreed (e.g. "32 features") | Only `QFF2026_QML_Participant_Problem_Statements_v3.docx` is authoritative |
| Environment | Python 3.10, qiskit 1.1.0, numpy 1.26.4 — would not install on current Python; no PennyLane–Qiskit plugin | `requirements.txt` with tested pins for Python 3.11–3.13, including `PennyLane-qiskit==0.44.1`; use a dedicated venv |
| Noise model | v1: gates such as sx, u, cp, rzz ran noise-free | v2: sx/sdg/tdg added, and every noisy run is transpiled to a fully-noised basis; un-noised gates raise an error |
| Task A fidelity | Starter hint used P(00)+P(11) (a population, not a fidelity) | F = (1 + ⟨XX⟩ − ⟨YY⟩ + ⟨ZZ⟩)/4 from XX, YY, ZZ correlations (now required, 8192 shots) |
| Task B fidelity | No input state or metric defined | Mean state fidelity over the 8 basis inputs, density-matrix simulation; code provided |
| Processor D | Direction of coupling map unclear; `benchmark_results` optional; minimum 2 qubits | Direction is enforced; `benchmark_results` required; 3–9 qubits; native gates limited to noised gates (cx/cz for two-qubit) |
| PS-Q3 predictions | Hard label only, but ranked by ROC-AUC | `sample_id,label,score`; AUC uses `score` (probability of label 1) |
| Test IDs | No ID column; key and template used different column orders | Test files start with `sample_id`; all files use `sample_id,label` |
| PS-Q1 provenance | "preprocessed from" vs "modelled on" MIT-BIH | "modelled on the statistics of" MIT-BIH everywhere |
| Deadline / submission | Not stated | Day 2, 5:40 PM IST, via the hackathon submission portal |
