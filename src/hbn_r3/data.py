"""Explicit data contracts; never infer a clinical label from a filename."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

RELEASES = ("R4_L100", "R5_L100", "R7_L100", "R9_L100")
REQUIRED = ("participant_id", "release", "path", "label", "age", "sex", "group_id",
            "diagnosis_confirmed", "cleaning_provenance")


def read_manifest(path):
    path = Path(path).resolve()
    df = pd.read_csv(path, dtype={"participant_id": str, "group_id": str, "release": str, "sex": str})
    missing = set(REQUIRED) - set(df)
    if missing:
        raise ValueError(f"Missing manifest columns: {sorted(missing)}")
    if df.empty or df.participant_id.isna().any() or df.group_id.isna().any():
        raise ValueError("Manifest is empty or participant/family group IDs are missing")
    if df.participant_id.duplicated().any():
        raise ValueError("Duplicate participant IDs across files/releases: resolve explicitly before analysis")
    df["release"]=df.release.str.replace(r"_bdf$","",regex=True)
    if not set(df.release).issubset(RELEASES):
        raise ValueError(f"Expected release labels from {RELEASES}; these are not site labels")
    if df.label.isna().any() or not set(df.label).issubset({0, 1}):
        raise ValueError("Explicit binary labels 0/1 are required; missing does not mean control")
    if not df.diagnosis_confirmed.astype(str).str.lower().isin(["true", "1"]).all():
        raise ValueError("All diagnoses/control definitions must be explicitly confirmed")
    if df.cleaning_provenance.fillna("").str.strip().eq("").any():
        raise ValueError("Document preprocessing and anti-alias provenance before extracting features")
    df["label"] = df.label.astype(int)
    df["age"] = pd.to_numeric(df.age, errors="coerce")
    df["sex"] = df.sex.fillna("unknown")
    df["path"] = [str((path.parent / p).resolve()) for p in df.path]
    absent = [p for p in df.path if not Path(p).is_file()]
    if absent:
        raise FileNotFoundError(f"Missing input file: {absent[0]}")
    return df


def load_epochs(path, expected_sfreq=100.0, epoch_seconds=4.0):
    with np.load(path, allow_pickle=False) as z:
        needed = {"eo", "ec", "sfreq", "ch_names", "cleaned", "units"}
        if not needed.issubset(z.files):
            raise ValueError(f"NPZ must contain {sorted(needed)}")
        fs = float(z["sfreq"])
        if not np.isclose(fs, expected_sfreq):
            raise ValueError(f"Expected {expected_sfreq} Hz; found {fs}. Do not relabel the sampling rate")
        if z["cleaned"].shape != () or z["cleaned"].dtype.kind != "b" or not z["cleaned"].item() or str(z["units"].item()) != "V":
            raise ValueError("Supply audited, artifact-cleaned EEG in volts")
        names = z["ch_names"].astype(str).tolist()
        if len(names) != len(set(names)):
            raise ValueError("Channel names must be unique")
        out = {}
        for cond in ("eo", "ec"):
            x = np.asarray(z[cond], float)
            if x.ndim != 3 or x.shape[1] != len(names):
                raise ValueError("Each condition must have shape (epochs, channels, samples)")
            if x.shape[2] != round(fs * epoch_seconds):
                raise ValueError("Input must use nonoverlapping 4-second epochs")
            if not np.isfinite(x).all():
                raise ValueError("Nonfinite EEG values found")
            if len(x) < 4 or np.std(x) < 1e-12:
                raise ValueError("Too few epochs or degenerate recording")
            out[cond] = x
    return out, fs, names


def load_regions(path, ch_names):
    with open(path) as f:
        regions = json.load(f)
    if len(regions) != 6:
        raise ValueError("This protocol requires exactly six prespecified scalp regions")
    flat = [ch for values in regions.values() for ch in values]
    if len(flat) != len(set(flat)) or any(not v for v in regions.values()):
        raise ValueError("Regions must be nonempty, disjoint channel lists")
    if not set(flat).issubset(ch_names):
        raise ValueError(f"Missing mapped channels: {sorted(set(flat)-set(ch_names))}")
    return {k: [ch_names.index(c) for c in v] for k, v in regions.items()}
