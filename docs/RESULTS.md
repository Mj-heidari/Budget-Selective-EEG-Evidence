# Study results

Release-blocked leave-one-release-out on 514 participants from HBN-EEG
releases R4, R5, R7 and R9, at the 20 s and 40 s clean-data budgets. Every
figure and number below is reproduced by the commands in the README.

## Target

The target is age band, the median split of the cohort at 10.0 years. It is a
demographic rather than a clinical construct and is chosen deliberately: a
coverage–accuracy analysis is only interpretable when the underlying signal is
real, and maturational structure in resting-state spectra is well established.
An earlier run of this pipeline against a CBCL attention extreme-group contrast
is retained in the repository history; there the classifier sat at chance, no
selective policy was feasible at any target, and the policy correctly abstained
on every participant. That outcome is a property of the label, not of the
policy, and it is why the reported study uses a decodable target.

Because no family identifiers are distributed, partitions are blocked at
participant level and residual sibling dependence is unquantified. The same 514
participants pass quality control at both budgets, so every across-budget
comparison is paired.

## The acquisition batch dominates

Held-out balanced accuracy for the reliability-guided model:

| Held-out release | 20 s | 40 s |
|---|---:|---:|
| R4 | 0.735 | 0.765 |
| R5 | 0.698 | 0.706 |
| R7 | 0.754 | 0.785 |
| R9 | 0.617 | 0.638 |
| **Pooled** | **0.698** | **0.720** |

The spread across curation tranches is 14.7 points at the 40 s budget, with
identical preprocessing and no site change. Doubling the clean budget is worth
+2.1 points (95% CI [−0.6, +4.5]) by paired participant-cluster bootstrap, and
no augmentation scheme moves the pooled figure by more than one point.

## What augmentation is worth

Change in pooled held-out balanced accuracy against the unaugmented model, 95%
paired participant-cluster intervals:

| Scheme | Δ BA (points) | 95% CI |
|---|---:|---|
| Reliability-guided cLSGAN | +1.0 | [−0.4, +2.4] |
| Random oversampling | +0.8 | [−0.6, +2.2] |
| cLSGAN + support filter | +0.0 | [−2.0, +2.2] |
| Plain cLSGAN | −0.8 | [−2.5, +1.0] |
| Gaussian jitter | −1.0 | [−2.8, +1.0] |

No interval excludes zero. The ordering is still informative: coupling the
generator to feature reliability and filtering to the real support are what
separate a mildly useful generator from a mildly harmful one.

The augmentation ratio is itself tuned on inner folds, and for the
reliability-guided model it is driven to zero in two of the four folds (R5 and
R9), where the augmented model is identical to the unaugmented one. Where the
procedure does augment (R4, R7) the held-out change is −0.7 and +4.7 points.
The capacity to decline is what keeps the scheme from being harmful.

## Selective evidence budget

Operating points on untouched releases for the reliability-guided model. Mean s
is clean seconds per condition, counting the full budget for abstentions.

| Rule | Target | Coverage | Accepted BA | Mean s |
|---|---:|---:|---:|---:|
| Fixed 20 s | — | 1.000 | 0.698 | 20.0 |
| Fixed 40 s | — | 1.000 | 0.720 | 40.0 |
| Agreement | 0.65 | 0.905 | 0.719 | 24.7 |
| Margin | 0.70 | 0.842 | 0.739 | 24.2 |
| Agreement | 0.75 | 0.700 | 0.767 | 27.9 |
| Margin | 0.75 | 0.626 | 0.789 | 29.0 |
| Margin | 0.80 | 0.482 | 0.827 | 31.8 |
| Agreement | 0.90 | 0.000 | — | 40.0 |

At a 0.70 target the rule answers for 84.2% of held-out children at balanced
accuracy 0.739 using 24.2 s of clean data per condition — higher accuracy than
a fixed 40 s protocol applied to everyone, at 40% less recording. The fixed
protocols are endpoints of the same curve: at the loosest target the margin
rule accepts everyone at 20 s and reproduces the fixed-20 s point exactly.

At a 0.90 target, the level a high-stakes screening protocol would ask for, no
inner-feasible policy exists and the rule abstains on every participant. This
is reported rather than relaxed post hoc.

## Representation and classifier study

Wider tuning grid, 40 s budget, conditional participant-cluster intervals:

| Feature set / classifier | BA | AUC | 95% CI |
|---|---:|---:|---|
| Mutual information only | 0.732 | 0.798 | [0.690, 0.768] |
| Reliability-scored (RF) | 0.730 | 0.791 | [0.688, 0.763] |
| Static fusion | 0.724 | 0.794 | [0.681, 0.759] |
| Reliability-scored (logistic) | 0.724 | 0.783 | [0.680, 0.761] |
| Reliability-scored (SVM) | 0.722 | 0.788 | [0.678, 0.755] |
| Consistency-weighted only | 0.710 | 0.785 | [0.667, 0.748] |
| Conventional band powers | 0.698 | 0.774 | [0.656, 0.738] |

The spread from weakest to strongest is 3.4 points and no pairwise contrast
excludes zero: reliability scoring is +1.9 points over consistency weighting
alone ([−0.0002, +0.039]) and −0.2 against mutual-information selection
([−0.021, +0.016]). Only conventional band powers fall clearly behind.

The `demographics` model in `configs/research.json` is deliberately omitted
from this table. For an age-band target it reaches 0.994 because age is one of
its inputs; the number measures the leak, not the model.

## Ordering of effects

    acquisition batch (14.7 pts)  >>  recording budget (2.1 pts)
      >  representation, classifier and augmentation (< 2 pts each)

The selective policy is what converts the second of these into a better
accuracy per second of recording. It does not come from a better model.

## Relationship to the companion study

The SeqMap study (github.com/Mj-heidari/Repo4) analyses the same 514
participants and the same 315 measurements with sequential panel discovery
rather than reliability-weighted selection, and reaches a compatible
conclusion: the transferable signal in these recordings is spectral and
duplicated across scalp regions.
