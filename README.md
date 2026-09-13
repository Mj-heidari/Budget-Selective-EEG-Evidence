# R3-Budget

Reliability-guided conditional GAN augmentation and selective EEG evidence use under release shift.

**Research software and manuscript draft. No HBN classification results are included.** Synthetic runs verify software only.

## Windows quick start

Install 64-bit Python 3.12 and Git. Open a PowerShell terminal:

```powershell
git clone https://github.com/Mj-heidari/Repo3.git
cd Repo3
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[bdf]"
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m hbn_r3.cli demo --output demo_data/features.csv
.\.venv\Scripts\python.exe -m hbn_r3.cli run --features demo_data/features.csv --config configs/smoke.json --output outputs/smoke
```

Use a new output folder on every run. See [the full Windows guide](docs/Windows_Run_Guide.pdf) for the real-data workflow and Repo4.

## Your data

Default root: `D:\EEG_Dataset\HBN_100Hz`.

Folders: `R4_L100_bdf`, `R5_L100_bdf`, `R7_L100_bdf`, `R9_L100_bdf`. These normalize to internal release names without `_bdf`. Sampling frequency comes from file metadata.

The folder names do not reveal diagnosis labels, EEG channel mapping, condition event codes or cleaning history. Supply verified labels and family IDs, EEG-only channel names, six scalp regions, and EO/EC intervals. Schema-only examples intentionally contain no guessed participant data or event times.

```powershell
.\.venv\Scripts\python.exe -m hbn_r3.cli inspect-data --root "D:\EEG_Dataset\HBN_100Hz" --output local/inventory.csv
```

Read [DATA_CONTRACT.md](docs/DATA_CONTRACT.md), [BDF_PREPARATION.md](docs/BDF_PREPARATION.md), and [EXPERIMENTS.md](docs/EXPERIMENTS.md). Keep recordings, local manifests, and experiment outputs outside git; `local/`, `outputs/`, and recording formats are ignored.

## Methods

- 315 spectral, phase-lag connectivity, strength and EO/EC contrast candidates.
- Real-training-only imputation, optional demographic adjustment, reliability/release selection and scaling.
- Random forest, RBF-SVM and regularized logistic regression.
- Conditional least-squares GAN implemented in NumPy; no PyTorch or GPU needed.
- Duplication, Gaussian jitter, unguided GAN, support-filtered GAN and reliability-guided GAN comparisons.
- Nested clean-data budgets of 20/40/60 seconds per condition; margin-only versus full/half-view agreement policies.
- Real inner-validation policy fitting; abstention if the selected policy cannot meet the prespecified development target.
- Family-blocked release holdout, saved outer models, split records, augmentation diagnostics and paired comparisons.

A 90% selective-performance target is not a guaranteed real-data accuracy. Always report coverage and fixed-budget performance. These budgets are retrospective clean-data use, not prospective acquisition times. Generated vectors are not raw EEG and are not privacy-certified.

## Configurations

| File | Purpose |
|---|---|
| `smoke.json` | Short synthetic exercise of nine classifier/augmentation variants |
| `research.json` | Primary classifier and selection comparisons |
| `research_adjusted.json` | Age/age-squared/sex residualization sensitivity |
| `gan_research.json` | Matched augmentation comparisons; ratio includes zero |
| `budget_smoke.json` | Small synthetic policy check |
| `budget_research.json` | Real-data budget/abstention experiment |

Read [VALIDATION.md](docs/VALIDATION.md) for tests and limitations, and [REVIEW.md](docs/REVIEW.md) for the scientific review. The paper is in `paper/main.tex` and `paper/main.pdf`. To rebuild: run `python scripts/render_bibliography.py`, then run `pdflatex main.tex` twice inside `paper/`. The included bibliography fragment avoids a missing `IEEEtran.bst` dependency.
