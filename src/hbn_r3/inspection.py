"""Local schema inspection without inventing labels, events, or channel mappings."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.io import whosmat
from .data import read_manifest, load_epochs, load_regions


def inspect_mat(args):
    try:
        print(json.dumps([{'variable':n,'shape':s,'type':t} for n,s,t in whosmat(args.input)],indent=2))
    except NotImplementedError as exc:
        raise ValueError('MAT v7.3 requires an explicit HDF5 adapter; export audited arrays as MAT v5 or NPZ') from exc


def inspect_data(args):
    root=Path(args.root)
    if not root.is_dir():raise FileNotFoundError(f'Dataset root not found: {root}')
    rows=[]
    for p in sorted(root.rglob('*')):
        if not p.is_file():continue
        row={'relative_path':str(p.relative_to(root)),'suffix':p.suffix.lower(),'bytes':p.stat().st_size,'schema':'','error':''}
        try:
            if p.suffix.lower()=='.npz':
                with np.load(p,allow_pickle=False) as z:
                    row['schema']=json.dumps({k:{'shape':z[k].shape,'dtype':str(z[k].dtype)} for k in z.files})
            elif p.suffix.lower()=='.mat':
                row['schema']=json.dumps([{'variable':n,'shape':s,'type':t} for n,s,t in whosmat(p)])
        except Exception as exc:row['error']=str(exc)
        rows.append(row)
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows,columns=['relative_path','suffix','bytes','schema','error']).to_csv(out,index=False)
    print(f'Inventory saved: {out}; {len(rows)} files. No labels or conditions inferred.')


def validate_data(args):
    manifest=read_manifest(args.manifest);rows=[]
    for _,person in manifest.iterrows():
        row={'participant_id':person.participant_id,'release':person.release,'label':person.label,'status':'ok','reason':''}
        try:
            data,fs,names=load_epochs(person.path);load_regions(args.regions,names)
            if not np.isfinite(person.age) or not 6<=person.age<=17:raise ValueError('Age outside primary 6-17 range or missing')
            row.update(eo_seconds=len(data['eo'])*4,ec_seconds=len(data['ec'])*4,channels=len(names),sfreq=fs)
            if min(row['eo_seconds'],row['ec_seconds'])<60:raise ValueError('Need 60 clean seconds per condition')
        except (ValueError,KeyError,FileNotFoundError) as exc:row.update(status='excluded',reason=str(exc))
        rows.append(row)
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    df=pd.DataFrame(rows);df.to_csv(out,index=False)
    print(df.groupby(['release','label','status']).size().to_string())
    if not (df.status=='ok').any():raise ValueError(f'No eligible records. See {out}')
