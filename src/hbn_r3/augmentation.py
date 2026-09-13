"""Training-only tabular conditional least-squares GAN augmentation; never synthesizes raw EEG.

Each fit sees only its current training partition. Conditions are observed
class/release cells. Quality bounds use real training neighbors only.
"""
import numpy as np
from scipy.spatial.distance import cdist


def support_mask(real, generated):
    """Reject exact copies and samples beyond the 95th percentile real LOO radius.

    Distances use already training-standardized coordinates. This heuristic is
    neither a physiological validity test nor a differential privacy guarantee.
    """
    distances = cdist(real, real)
    np.fill_diagonal(distances, np.inf)
    radius = float(np.quantile(distances.min(axis=1), .95))
    nearest = cdist(generated, real).min(axis=1)
    return (nearest > 1e-6) & (nearest <= radius), nearest, radius


def augment(z, y, releases, method, ratio, seed, steps=400, batch_size=32,
            critic_steps=3, moment_weight=0., feature_weights=None):
    if method not in {'none', 'ros', 'jitter', 'gan', 'gan_gated'}:
        raise ValueError(f'Unknown augmentation: {method}')
    if not 0 <= ratio <= 1:
        raise ValueError('augmentation_ratio must be between 0 and 1')
    if method == 'none' or ratio == 0:
        return z, y, {'method': method, 'requested': 0, 'accepted': 0, 'cells': []}
    rng = np.random.default_rng(seed)
    cells = [(int(c), r) for c in (0, 1) for r in sorted(set(releases))]
    indices = [np.flatnonzero((y == c) & (releases == r)) for c, r in cells]
    if min(map(len, indices)) < 4:
        raise ValueError('Augmentation needs at least 4 real participants per class/release cell')
    # Same class/release budget for every augmentation control, without changing
    # relative class/release frequencies beyond integer rounding.
    counts = [int(np.floor(ratio * len(ix))) for ix in indices]
    generator = None
    trace = []
    if method.startswith('gan'):
        from .gan import generate
        weights=np.ones(z.shape[1]) if feature_weights is None else np.asarray(feature_weights)
        candidates,trace=generate(z,indices,counts,rng,steps,batch_size,critic_steps,
            moment_weight,weights,method=="gan_gated")
        generator=True
    else:
        candidates=[]
        for ix,n in zip(indices,counts):
            fake=z[rng.choice(ix,n,replace=True)].copy()
            if method=='jitter':fake += rng.normal(0,.1,fake.shape)
            candidates.append(fake)
    synthetic=[];labels=[];reports=[]
    for (c,r),ix,n,fake in zip(cells,indices,counts,candidates):
        if len(fake):
            keep,near,radius=support_mask(z[ix],fake)
            used=fake[keep][:n] if method=='gan_gated' else fake[:n]
            finite=np.isfinite(used).all(axis=1)
            if not finite.all():raise ValueError('Nonfinite synthetic data')
            reports.append({'label':c,'release':str(r),'n_real':len(ix),'requested':n,
                'candidates':len(fake),'accepted':len(used),'support_fraction':float(keep.mean()),
                'nearest_median':float(np.median(near)),'real_loo_radius95':radius})
            synthetic.append(used);labels.extend([c]*len(used))
    zz=np.concatenate([z]+synthetic) if synthetic else z.copy()
    yy=np.r_[y,np.asarray(labels,int)]
    return zz,yy,{'method':method,'requested':sum(counts),'accepted':len(labels),
        'moment_weight':moment_weight,'steps':steps if generator is not None else 0,
        'cells':reports,'loss_trace':trace}
