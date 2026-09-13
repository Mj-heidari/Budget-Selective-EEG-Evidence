# BDF preparation

Install `python -m pip install -e ".[bdf]"`. The NumPy GAN does not need this extra; MNE is needed only for BDF I/O.

1. Inventory `D:\EEG_Dataset\HBN_100Hz` with `inspect-data`.
2. Inspect a representative BDF with `inspect-bdf --input FILE --output local/inspect_one`. The command writes channel/type/sampling metadata and annotation descriptions. After identifying the actual trigger channel, add `--stim-channel NAME`; optional `--event-mask` defaults to 65535. Verify any bit mask against acquisition documentation.
3. Create a JSON list of vetted EEG-only channel labels. Do not include Status/trigger/EOG channels in an average reference. Exclude or repair bad channels consistently before the regional analysis; this path does not interpolate them.
4. Create a separate interval CSV for every recording: `onset_seconds,duration_seconds,condition`. Values must come from verified events/annotations. Conditions are exactly `eo` or `ec`. Times are seconds from file start. Rows must not overlap or exceed the recording. The supplied schema file has no fabricated event values.
5. Prepare a raw manifest using the supplied schema and confirmed clinical metadata. `path` points to the BDF and `intervals_path` to its CSV. Relative paths resolve against the raw manifest's directory. Absolute Windows paths are supported on Windows. Release names may include `_bdf`. `cleaning_provenance` initially records known source preprocessing; the output manifest records the preparation sidecar.
6. Run `prepare-cohort` with an explicit reference and filter mode. `--filter-mode keep` preserves documented derivative filtering; `bandpass` applies a fourth-order 1-40 Hz Butterworth filter. `--reference keep` retains an audited reference; `average` rereferences all explicitly selected EEG channels. Inspect real data before choosing.
7. The defaults trim two seconds at each interval boundary, reject any channel above 150 microvolts peak-to-peak or below 0.1 microvolts peak-to-peak, and exclude BAD annotations. Four-second nonoverlapping epochs are formed at 100 Hz. At least 15 accepted epochs in each condition are needed. Already aliased input cannot be repaired by resampling.
8. Review the resulting epoch/rejection CSVs and provenance JSONs plus representative EEG traces. `--confirm-preprocessing` documents your review of the preprocessing policy; it is not automatic certification of clean EEG. ICA and automatic channel interpolation are not performed.
9. Create the six-region mapping from the actual montage, then use `validate-data` and `extract`.

Each participant is prepared to `record_NNNNN.npz`, avoiding unsafe filenames from participant IDs. The prepared manifest is checkpointed after each success, while failures go to `preparation_exclusions.csv`. Choose a fresh destination directory; do not overwrite prior preparation.

The BDF reader could not be executed in the preparation environment because MNE was unavailable. The pure-array epoch boundary, trim, bad-span and overlap behavior was tested. Validate a representative real BDF locally before a full cohort run.
