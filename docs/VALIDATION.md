# Validation record

Local environment: Python 3.12.14, Linux; exact scientific package versions are in `requirements-tested.txt`.

- 19 focused tests passed: original spectral/connectivity and manifest checks, analytic network/moment gradients against central finite differences, deterministic GAN behavior, augmentation support filtering, classifier transform immutability, BDF pure-array epoching, nested duration subsets, and selective-policy behavior.
- Nine-variant synthetic nested run completed: 128 fixture participants across four releases and three durations produced 3,456 unique OOF predictions.
- Two-model/two-policy synthetic budget run completed: 512 unique participant/model/policy rows.
- SVM margins were verified to have no false probability values; output identity and duration constraints were checked.
- MNE installation was unavailable locally. Actual BDF disk I/O and Windows PowerShell execution were not tested here. CI is configured on Linux and Windows; its status is separate from these local checks.
- No HBN EEG, clinical outcomes, or real-data performance were evaluated. Synthetic fixtures are intentionally separable and do not establish any method's scientific benefit.

Run tests with `python -m unittest discover -s tests -v`. Smoke output folders are ignored by git. Model files must be treated as trusted local artifacts; only load model files you created or trust.
