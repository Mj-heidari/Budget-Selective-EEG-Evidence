"""Explicit BDF inspection and conservative, auditable epoch preparation.

No event code is assigned a condition automatically. The user provides verified
intervals in seconds from file start and a vetted EEG-only channel list.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd


def _mne():
    try:import mne
    except ImportError as exc:raise ImportError('Install BDF support: python -m pip install -e ".[bdf]"') from exc
    return mne


def inspect_bdf(args):
    mne=_mne();raw=mne.io.read_raw_bdf(args.input,preload=False,verbose='ERROR')
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    info={'sfreq':float(raw.info['sfreq']),'seconds':raw.n_times/raw.info['sfreq'],
          'channels':raw.ch_names,'types':raw.get_channel_types(),'first_samp':int(raw.first_samp)}
    (out/'bdf_info.json').write_text(json.dumps(info,indent=2))
    pd.DataFrame({'onset_seconds':raw.annotations.onset,'duration_seconds':raw.annotations.duration,
        'description':raw.annotations.description}).to_csv(out/'annotations.csv',index=False)
    if args.stim_channel:
        if args.stim_channel not in raw.ch_names:raise ValueError('Stim channel not in recording')
        events=mne.find_events(raw,stim_channel=args.stim_channel,shortest_event=1,
            uint_cast=True,mask=args.event_mask,mask_type='and',verbose='ERROR')
        pd.DataFrame({'sample':events[:,0],'onset_seconds':(events[:,0]-raw.first_samp)/raw.info['sfreq'],
            'previous_value':events[:,1],'code':events[:,2]}).to_csv(out/'trigger_events.csv',index=False)
    raw.close();print(f'Saved BDF metadata and event inventory in {out}')


def epoch_arrays(signal,fs,intervals,reject_uv=150.,flat_uv=.1,trim_seconds=2.,bad_intervals=None):
    """Pure array epoching; preserves condition boundaries and rejects bad spans."""
    required={'onset_seconds','duration_seconds','condition'}
    if not required.issubset(intervals):raise ValueError(f'Intervals need {required}')
    if not np.isclose(fs,100):raise ValueError('Epoch input must be resampled to 100 Hz')
    if not np.isfinite(signal).all():raise ValueError('Nonfinite EEG signal')
    if not 0<=flat_uv<reject_uv or trim_seconds<0:raise ValueError('Invalid rejection/trim settings')
    intervals=intervals.sort_values('onset_seconds');last_end=0.;out={'eo':[],'ec':[]};audit=[]
    for row in intervals.itertuples():
        start=float(row.onset_seconds);duration=float(row.duration_seconds);condition=str(row.condition).lower()
        stop=start+duration
        if not np.isfinite([start,duration]).all() or start<0 or duration<=0 or stop>signal.shape[1]/fs+1e-8:
            raise ValueError('Interval outside recording or invalid duration')
        if start<last_end-1e-8:raise ValueError('Condition intervals overlap')
        if condition not in out:raise ValueError('condition must be eo or ec')
        last_end=stop
        lo=int(np.ceil((start+trim_seconds)*fs));hi=int(np.floor((stop-trim_seconds)*fs))
        for first in range(lo,hi-399,400):
            x=signal[:,first:first+400];ptp=np.ptp(x,axis=1)*1e6
            bad_span=any(first/fs<b and (first+400)/fs>a for a,b in (bad_intervals or []))
            reason='bad_annotation' if bad_span else ('amplitude' if np.any(ptp>reject_uv) else ('flat' if np.any(ptp<flat_uv) else 'accepted'))
            audit.append({'condition':condition,'onset_seconds':first/fs,'status':reason})
            if reason=='accepted':out[condition].append(x)
    for condition in out:
        out[condition]=np.stack(out[condition]) if out[condition] else np.empty((0,signal.shape[0],400))
    return out,audit


def prepare_bdf(args):
    if not args.confirm_preprocessing:
        raise ValueError('Review filtering/reference/channel/artifact choices, then use --confirm-preprocessing')
    if Path(args.output).exists():raise ValueError('Output already exists; select a new path')
    mne=_mne();channels=json.loads(Path(args.channels).read_text())
    if not isinstance(channels,list) or len(channels)<2 or len(set(channels))!=len(channels):
        raise ValueError('Supply a JSON list of unique, vetted EEG channels')
    raw=mne.io.read_raw_bdf(args.input,preload=True,verbose='ERROR')
    source_fs=float(raw.info['sfreq'])
    if source_fs<100:raise ValueError('Sampling rate below protocol requirement')
    if not set(channels).issubset(raw.ch_names):raise ValueError('EEG channel list contains missing channels')
    raw.pick(channels)
    raw.set_channel_types({c:'eeg' for c in channels},verbose='ERROR')
    # Filtering may be skipped only for verified existing derivatives.
    if args.filter_mode=='bandpass':
        raw.filter(1.,40.,method='iir',iir_params={'order':4,'ftype':'butter'},verbose='ERROR')
    if args.reference=='average':raw.set_eeg_reference('average',projection=False,verbose='ERROR')
    if not np.isclose(source_fs,100):raw.resample(100.,verbose='ERROR')
    bad=[(float(a['onset']),float(a['onset']+a['duration'])) for a in raw.annotations
         if str(a['description']).lower().startswith('bad')]
    intervals=pd.read_csv(args.intervals)
    arrays,audit=epoch_arrays(raw.get_data(),float(raw.info['sfreq']),intervals,
        args.reject_uv,args.flat_uv,args.trim_seconds,bad)
    if min(len(arrays['eo']),len(arrays['ec']))<15:
        raise ValueError('Fewer than 60 accepted seconds in a condition; revise cohort eligibility, not event labels')
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out,**arrays,sfreq=100.,ch_names=np.asarray(raw.ch_names,dtype='U'),units='V',cleaned=True)
    pd.DataFrame(audit).to_csv(out.with_suffix('.epochs.csv'),index=False)
    provenance={'source':str(Path(args.input).resolve()),'source_sfreq':source_fs,'output_sfreq':100.,
        'filter_mode':args.filter_mode,'reference':args.reference,'reject_uv':args.reject_uv,'flat_uv':args.flat_uv,
        'trim_seconds':args.trim_seconds,'channels':raw.ch_names,'intervals':intervals.to_dict('records'),
        'mne_version':mne.__version__,'method':'threshold rejection and BAD annotation exclusion; no ICA',
        'preprocessing_review_confirmed':True,'n_eo':len(arrays['eo']),'n_ec':len(arrays['ec'])}
    out.with_suffix('.provenance.json').write_text(json.dumps(provenance,indent=2))
    raw.close();print(f'Saved {out}; inspect the epoch log and review representative EEG before analysis')


def prepare_cohort(args):
    from types import SimpleNamespace
    from .data import read_manifest
    source=Path(args.manifest).resolve();manifest=read_manifest(source)
    if 'intervals_path' not in manifest:raise ValueError('Raw manifest needs intervals_path for every recording')
    destination=Path(args.output_dir).resolve()
    if destination.exists() and any(destination.iterdir()):raise ValueError('Choose an empty output directory')
    destination.mkdir(parents=True,exist_ok=True)
    successful=[];failures=[]
    for index,row in manifest.iterrows():
        # Output name uses row index, so participant IDs cannot become path syntax.
        output=destination/f'record_{index:05d}.npz'
        try:
            if pd.isna(row.intervals_path):raise ValueError('Missing intervals_path')
            intervals=(source.parent/str(row.intervals_path)).resolve()
            options=vars(args).copy();options.update(input=row.path,output=str(output),intervals=str(intervals))
            prepare_bdf(SimpleNamespace(**options))
            new=row.to_dict();new['path']=output.name
            new['cleaning_provenance']=str(output.with_suffix('.provenance.json'))
            successful.append(new)
        except (ValueError,FileNotFoundError,KeyError) as exc:
            failures.append({'participant_id':row.participant_id,'release':row.release,'label':row.label,'reason':str(exc)})
        pd.DataFrame(failures,columns=['participant_id','release','label','reason']).to_csv(destination/'preparation_exclusions.csv',index=False)
        if successful:pd.DataFrame(successful).to_csv(destination/'manifest.csv',index=False)
    if not successful:raise ValueError('No recordings prepared; read preparation_exclusions.csv')
    print(f'Prepared {len(successful)}; excluded {len(failures)}. Manifest: {destination / "manifest.csv"}')
