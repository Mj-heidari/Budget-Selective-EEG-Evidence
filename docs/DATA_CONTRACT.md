# Input contract and release audit

Only the IEEE template archive and the four folder labels were supplied. No EEG file, clinical table, event-code documentation, or montage was available. This contract avoids pretending to know those schemas.

## Participant NPZ

Load with `numpy.load(..., allow_pickle=False)`:

| Key | Type/shape | Definition |
|---|---|---|
| `eo` | float, `(E_open, C, 400)` | Clean nonoverlapping EO epochs in volts |
| `ec` | float, `(E_closed, C, 400)` | Clean nonoverlapping EC epochs in volts |
| `sfreq` | scalar float | Actual sampling rate; must equal 100 |
| `ch_names` | Unicode array `(C,)` | Unique channel labels in actual array order |
| `units` | scalar Unicode | Literal `V` |
| `cleaned` | scalar Boolean | True only after actual preprocessing audit |

All values must be finite; conditions must share channel order, units, and reference. Exclude transitions and artifact epochs before writing the arrays. Preserve temporal ordering within each condition. The input contract does not imply that artifact removal can be inferred from the values alone.

`60` means 60 **clean seconds per condition**, not 60 seconds of acquisition. Four-second epochs imply at least 15 EO and 15 EC epochs. The `L100` suffix is not assumed to mean one-second epochs, 100 participants, or 100 channels. If files only retain one-second fragments and original temporal adjacency is unknown, do not concatenate arbitrary fragments into four-second epochs; reconstruct from the source or redesign and validate the features.

## Manifest

| Column | Requirement |
|---|---|
| `participant_id` | Unique HBN participant ID; resolve duplicate releases first |
| `release` | One of `R4_L100`, `R5_L100`, `R7_L100`, `R9_L100` |
| `path` | Participant NPZ path, relative to the manifest |
| `label` | Explicit 0/1 derived from the agreed clinical target |
| `age` | Numeric age; default extractor includes 6–17 |
| `sex` | Dataset's recorded sex category; unknown must remain explicit |
| `group_id` | Family identifier; use participant ID only if relatives are unavailable, and disclose this limitation |
| `diagnosis_confirmed` | `true` only after diagnostic/control verification |
| `cleaning_provenance` | Pipeline version, original/reference sampling rate, filter/anti-alias steps, reference, artifact rules |

Retain medication, comorbidity, acquisition site, original release membership, exclusions, and clinical assessment details in a separate private audit table. These are not automatically inferred or included as predictors.

## Six-region map

JSON maps six unique region names to disjoint, nonempty channel-label lists. Use the sensor coordinates/channel documentation in your release. The supplied 10–20 example explains the schema only. Do not rename EGI channels to 10–20 labels based on guesses. Scalp regions are sensor groupings, not source-localized anatomy.

## What the repository deliberately rejects

- Duplicate participant IDs across releases or absent clinical labels.
- Unknown release names, unverified diagnoses, or undocumented cleaning.
- Wrong sampling rate, mismatched axis shape, nonfinite EEG, unknown units, duplicate channels.
- Participants without the longest requested duration in both conditions.
- Missing mapped channels or overlapping regions.
- Different participant sets or labels across duration comparisons.
- Infeasible validation folds or insufficient training releases for the proposed score.

## Frequency limitations at 100 Hz

Use verified upstream anti-alias processing. A 60 Hz artifact may alias to 40 Hz after careless downsampling; a 60 Hz digital notch is invalid on a 100 Hz sampled signal. This implementation restricts features to 4–30 Hz and the spectral fit to 2–30 Hz, but those choices cannot undo upstream aliasing. Unknown derivative preprocessing is a data limitation, not an invitation to invent a filter history.

## Current BDF extension

Folder suffix `_bdf` is accepted in manifest releases and normalized away. Optional BDF preparation and interval/channel contracts are documented in BDF_PREPARATION.md. Epoch duration subsets now use a fixed 15-epoch anchor with nested 5/10/15 subsets. Real-data labels, events and montage remain explicit user inputs.
