"""Explicit spectral model and debiased squared wPLI; no black-box fitting dependency."""
import warnings
from itertools import combinations
import numpy as np
from scipy.signal import periodogram, windows
from scipy.optimize import least_squares

BANDS = {"theta": (4., 8.), "alpha": (8., 13.), "beta": (13., 30.)}


def spectral_model(f, p):
    """Fixed aperiodic background plus ONE alpha Gaussian in log10 power."""
    offset, exponent, height, center, width = p
    return offset - exponent * np.log10(f) + height * np.exp(-0.5*((f-center)/width)**2)


def spectral_parameters(freq, power):
    """Research approximation, NOT the specparam/FOOOF implementation.

    Fit 2--30 Hz using soft-L1 loss. Gaussian center 7--13 Hz, sigma .5--3 Hz.
    Exponent bounds 0--4; log10 peak height 0--3. An absent/poor peak is NaN.
    """
    keep = (freq >= 2) & (freq <= 30)
    f = freq[keep]
    y = np.log10(np.maximum(power[keep], np.finfo(float).tiny))
    if len(f) < 30 or not np.isfinite(y).all():
        return np.full(4, np.nan), np.nan
    base = (f < 7) | (f > 14)
    slope, offset = np.polyfit(np.log10(f[base]), y[base], 1)
    residual = y - (offset + slope*np.log10(f))
    alpha = (f >= 7) & (f <= 13)
    center = f[alpha][np.argmax(residual[alpha])]
    start = [offset, np.clip(-slope, .01, 3.99), np.clip(np.max(residual[alpha]), .05, 2.9), center, 1.]
    fit = least_squares(lambda p: spectral_model(f,p)-y, start,
                        bounds=([-40,0,0,7,.5], [20,4,3,13,3]),
                        loss="soft_l1", f_scale=.1, max_nfev=150)
    yhat = spectral_model(f,fit.x)
    r2 = 1 - np.sum((y-yhat)**2)/max(np.sum((y-y.mean())**2),1e-12)
    b, chi, amp, cf, width = fit.x
    if not fit.success or r2 < .8:
        return np.full(4,np.nan), float(r2)
    if amp < .1 or width >= 2.99 or cf <= 7.01 or cf >= 12.99:
        amp, cf = np.nan, np.nan
    return np.array([chi,b,cf,amp]), float(r2)


def debiased_wpli(imaginary):
    """Vinck et al. debiased SQUARED wPLI, trial axis 0.

    Keep finite-sample negatives for edge estimates. Return NaN on zero denominator.
    """
    s = np.sum(imaginary,axis=0)
    a = np.sum(np.abs(imaginary),axis=0)
    q = np.sum(imaginary**2,axis=0)
    denominator = a*a-q
    return np.divide(s*s-q, denominator, out=np.full_like(s,np.nan,dtype=float),
                     where=denominator > np.finfo(float).tiny)


def condition_features(epochs, fs, regions):
    """Average spectra, never raw signals, within a scalp region."""
    freq, p = periodogram(epochs,fs=fs,window="hann",detrend="constant",axis=-1,scaling="density")
    psd = np.mean(p,axis=0)
    values, names, fits = [], [], []
    for region, channels in regions.items():
        spectrum = np.mean(psd[channels],axis=0)
        for band,(lo,hi) in BANDS.items():
            mask=(freq>=lo)&(freq<hi)
            power=np.sum(spectrum[mask])*(freq[1]-freq[0])
            values.append(np.log10(max(power,np.finfo(float).tiny)))
            names.append(f"sp_{region}_{band}")
        params,r2=spectral_parameters(freq,spectrum)
        for name,val in zip(("exponent","offset","alpha_cf","alpha_height"),params):
            names.append(f"sp_{region}_{name}"); values.append(val)
        fits.append(r2)
    centered=epochs-epochs.mean(axis=-1,keepdims=True)
    ft=np.fft.rfft(centered*windows.hann(epochs.shape[-1],sym=False),axis=-1)
    region_names=list(regions)
    matrices={band:np.zeros((6,6)) for band in BANDS}
    for i,j in combinations(range(6),2):
        pairs=[]
        for ci in regions[region_names[i]]:
            for cj in regions[region_names[j]]:
                im=np.imag(ft[:,ci,:]*ft[:,cj,:].conj())
                pairs.append(debiased_wpli(im))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore",RuntimeWarning)
            edge=np.nanmean(np.array(pairs),axis=0)
            for band,(lo,hi) in BANDS.items():
                val=np.nanmean(edge[(freq>=lo)&(freq<hi)])
                names.append(f"co_{region_names[i]}_{region_names[j]}_{band}")
                values.append(float(val))
                matrices[band][i,j]=matrices[band][j,i]=val
    for band,mat in matrices.items():
        # Strength is explicitly computed on clipped nonnegative edges; no connectivity causality.
        positive=np.clip(mat,0,1)
        for i,region in enumerate(region_names):
            names.append(f"gr_{region}_{band}_strength")
            values.append(float(np.sum(positive[i])/5))
    return np.asarray(values),names,float(np.nanmean(fits))


def feature_vector(eo,ec,fs,regions):
    a,names,qa=condition_features(eo,fs,regions)
    b,names2,qb=condition_features(ec,fs,regions)
    assert names==names2
    return np.r_[a,b,b-a], [f"{c}_{n}" for c in ("eo","ec","delta") for n in names], (qa,qb)


def progressive_indices(total, n, anchor=15):
    """Nested subsets of a fixed distributed 60-second anchor; not acquisition prefixes."""
    count=min(total,max(anchor,n))
    if n>total:raise ValueError("Not enough epochs")
    pool=np.linspace(0,total-1,count,dtype=int)
    order=[0]
    while len(order)<n:
        remaining=[i for i in range(count) if i not in order]
        order.append(max(remaining,key=lambda i:min(abs(i-j) for j in order)))
    return np.sort(pool[order])


def extract_participant(data,fs,regions,duration=60):
    n=int(duration/4)
    if duration%4 or n<4:
        raise ValueError("Duration must be a multiple of 4 seconds and at least 16 seconds")
    if min(len(data["eo"]),len(data["ec"]))<n:
        raise ValueError(f"Need {duration} clean seconds in EACH condition")
    # Deterministic, temporally ordered subsampling distributed over the clean recording.
    selected={c:x[progressive_indices(len(x),n)] for c,x in data.items()}
    full,names,qa=feature_vector(selected["eo"],selected["ec"],fs,regions)
    # Disjoint equal-sized halves. If odd, the last epoch contributes only to full features.
    even=2*(n//2)
    first,_,_=feature_vector(selected["eo"][:even:2],selected["ec"][:even:2],fs,regions)
    second,_,_=feature_vector(selected["eo"][1:even:2],selected["ec"][1:even:2],fs,regions)
    row={f"f__{name}":float(v) for name,v in zip(names,full)}
    row.update({f"a__{name}":float(v) for name,v in zip(names,first)})
    row.update({f"b__{name}":float(v) for name,v in zip(names,second)})
    row.update({"qc_eo_r2":qa[0],"qc_ec_r2":qa[1],"duration":duration})
    return row
