# Budget-Selective EEG Evidence

Reliability-guided conditional GAN augmentation and a selective evidence-budget
policy for release-blocked pediatric EEG decoding.

**Headline result.** On 514 children and adolescents from HBN-EEG releases R4,
R5, R7 and R9, a stopping rule tuned only on inner training folds answers for
84% of held-out children at balanced accuracy 0.739 using 24.2 s of clean data
per condition, against 0.720 for a fixed 40 s protocol applied to everyone, and
reaches 0.827 on the 48% it accepts most confidently. Acquisition batch
dominates every modelling choice: held-out accuracy spans 0.638 to 0.785 across
the four releases, roughly six times the largest modelling effect. See
[RESULTS.md](docs/RESULTS.md).

## Windows quick start

Install 64-bit Python 3.12 and Git. Open a PowerShell terminal:

```powershell
git clone https://github.com/Mj-heidari/Budget-Selective-EEG-Evidence.git
cd Budget-Selective-EEG-Evidence
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[bdf]"
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m hbn_r3.cli demo --output demo_data/features.csv
.\.venv\Scripts\python.exe -m hbn_r3.cli run --features demo_data/features.csv --config configs/smoke.json --output outputs/smoke
```

Use a new output folder on every run. See [the full Windows
guide](docs/Windows_Run_Guide.pdf) for the real-data workflow.

## Your data

Default root: `D:\EEG_Dataset\HBN_100Hz`.

Folders: `R4_L100_bdf`, `R5_L100_bdf`, `R7_L100_bdf`, `R9_L100_bdf`. These
normalize to internal release names without `_bdf`. Sampling frequency comes
from file metadata.

The folder names do not reveal diagnosis labels, EEG channel mapping, condition
event codes or cleaning history. Supply verified labels and family IDs, EEG-only
channel names, six scalp regions, and EO/EC intervals. Schema-only examples
intentionally contain no guessed participant data or event times.

```powershell
.\.venv\Scripts\python.exe -m hbn_r3.cli inspect-data --root "D:\EEG_Dataset\HBN_100Hz" --output local/inventory.csv
```

Read [DATA_CONTRACT.md](docs/DATA_CONTRACT.md),
[BDF_PREPARATION.md](docs/BDF_PREPARATION.md), and
[EXPERIMENTS.md](docs/EXPERIMENTS.md). Keep recordings, local manifests, and
experiment outputs outside git; `local/`, `outputs/`, and recording formats are
ignored.

## Reproducing the study

The reported study decodes age band — the median split of the cohort at 10.0
years — from the same participants and the same 315 measurements throughout.
A decodable target is required: a coverage–accuracy curve computed on a
chance-level classifier measures nothing. Build the label from the feature
table, then run the three studies and the policy-target sweep:

```powershell
.\.venv\Scripts\python.exe -m hbn_r3.cli run --features local\features_ageband.csv --config configs\gan_research.json    --output outputs\augmentation
.\.venv\Scripts\python.exe -m hbn_r3.cli run --features local\features_ageband.csv --config configs\budget_research.json --output outputs\budget
.\.venv\Scripts\python.exe -m hbn_r3.cli run --features local\features_ageband.csv --config configs\research.json        --output outputs\classifiers
.\.venv\Scripts\python.exe -m hbn_r3.cli run --features local\features_ageband.csv --config configs\budget_t35.json      --output outputs\budget_t35
```

Repeat the last line for `budget_t30`, `budget_t25` and `budget_t20`, then
render the figures:

```powershell
.\.venv\Scripts\python.exe scripts\make_figures.py outputs docs\figures
```

## Methods

- 315 spectral, phase-lag connectivity, strength and EO/EC contrast candidates.
- Real-training-only imputation, optional demographic adjustment,
  reliability/release selection and scaling.
- Random forest, RBF-SVM and regularized logistic regression.
- Conditional least-squares GAN implemented in NumPy; no PyTorch or GPU needed.
- Duplication, Gaussian jitter, unguided GAN, support-filtered GAN and
  reliability-guided GAN comparisons, with the augmentation ratio tuned on
  inner folds and allowed to be zero.
- Nested clean-data budgets of 20 and 40 seconds per condition; margin-only
  versus full/half-view agreement policies.
- Real inner-validation policy fitting; abstention if the selected policy
  cannot meet the prespecified development target.
- Participant-blocked release holdout, saved outer models, split records,
  augmentation diagnostics and paired comparisons.

A selective-performance target is a development constraint, not a guaranteed
real-data accuracy. Always report coverage and fixed-budget performance
alongside selective results. These budgets are retrospective clean-data use,
not prospective acquisition times. Generated vectors are not raw EEG and are
not privacy-certified. No family identifiers are distributed with these
releases, so blocking is participant-level and residual sibling dependence is
disclosed rather than assumed away.

## Configurations

| File | Purpose |
|---|---|
| `smoke.json` | Short synthetic exercise of nine classifier/augmentation variants |
| `research.json` | Primary classifier and selection comparisons |
| `research_adjusted.json` | Age/age-squared/sex residualization sensitivity |
| `gan_research.json` | Matched augmentation comparisons; ratio includes zero |
| `budget_smoke.json` | Small synthetic policy check |
| `budget_research.json` | Real-data budget/abstention experiment |
| `budget_t35.json` … `budget_t20.json` | Policy-target sweep, accepted-accuracy targets 0.65 to 0.80 |

`research.json` includes a `demographics` model. For an age-band target it
reaches 0.994 because age is one of its inputs; the number measures the leak,
not the model, and it is excluded from the reported comparison.

