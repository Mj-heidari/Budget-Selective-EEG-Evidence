"""Command-line workflow. Input adapters must be configured from the real data schema."""
import argparse
import json
import platform
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.io import loadmat
from .data import read_manifest,load_epochs,load_regions,RELEASES
from .features import extract_participant
from .evaluation import evaluate


def extract(args):
    manifest=read_manifest(args.manifest)
    durations=sorted(set(args.durations))
    records=[];excluded=[]
    for _,person in manifest.iterrows():
        try:
            if not np.isfinite(person.age) or not args.age_min <= person.age <= args.age_max:
                raise ValueError(f"Age missing or outside prespecified {args.age_min}--{args.age_max} range")
            data,fs,names=load_epochs(person.path)
            regions=load_regions(args.regions,names)
            if min(len(data["eo"]),len(data["ec"]))*4<max(durations):
                raise ValueError("Insufficient clean data for longest requested duration")
            rows=[]
            for d in durations:
                row={k:person[k] for k in ("participant_id","release","label","age","sex","group_id")}
                row.update(extract_participant(data,fs,regions,d))
                rows.append(row)
            records.extend(rows)
            print(f"Extracted {person.participant_id}",flush=True)
        except (ValueError,FileNotFoundError,KeyError) as exc:
            excluded.append({"participant_id":person.participant_id,"release":person.release,
                             "label":int(person.label),"reason":str(exc)})
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(excluded,columns=["participant_id","release","label","reason"]).to_csv(out.with_suffix(".excluded.csv"),index=False)
    if not records:raise ValueError("No eligible participants; inspect the exclusion log")
    pd.DataFrame(records).to_csv(out,index=False)
    audit={"input_participants":len(manifest),"included_participants":len(records)//len(durations),
           "excluded_participants":len(excluded),"durations":durations,"synthetic":False,
           "sampling_hz":100,"epoch_seconds":4,"age_range":[args.age_min,args.age_max],"regions":json.loads(Path(args.regions).read_text()),
           "python":platform.python_version()}
    out.with_suffix(".audit.json").write_text(json.dumps(audit,indent=2))


def demo(args):
    """Synthetic feature-level fixture; NOT simulated evidence for the method."""
    rng=np.random.default_rng(args.seed);rows=[]
    names=[]
    base=[f"sp_R{r}_{n}" for r in range(6) for n in ("theta","alpha","beta","exponent","offset","alpha_cf","alpha_height")]
    base += [f"co_R{i}_R{j}_{band}" for i in range(6) for j in range(i+1,6) for band in ("theta","alpha","beta")]
    base += [f"gr_R{r}_{band}_strength" for band in ("theta","alpha","beta") for r in range(6)]
    names=[f"{c}_{n}" for c in ("eo","ec","delta") for n in base]
    for r,release in enumerate(RELEASES):
        for i in range(args.n_per_release):
            label=i%2;age=float(rng.uniform(6,17));sex="F" if i%3 else "M"
            latent=rng.normal(size=len(base))
            effect=np.zeros(len(base));effect[:10]=.65*label
            eo=latent+rng.normal(0,.2,len(base))+.15*r
            ec=latent+effect+rng.normal(0,.2,len(base))+.15*r
            full=np.r_[eo,ec,ec-eo]
            epsa=rng.normal(0,.12,210);epsb=rng.normal(0,.12,210)
            a=full+np.r_[epsa,epsa[105:]-epsa[:105]]
            b=full+np.r_[epsb,epsb[105:]-epsb[:105]]
            for duration in (20,40,60):
                noise=rng.normal(0,.3*(60/duration-1),210)
                vals=full+np.r_[noise,noise[105:]-noise[:105]]
                row={"participant_id":f"SYNTH_R{r}_{i:03d}","group_id":f"SYNTH_R{r}_{i:03d}",
                     "release":release,"label":label,"age":age,"sex":sex,"duration":duration}
                for prefix,v in (("f__",vals),("a__",a),("b__",b)):
                    row.update({prefix+n:float(z) for n,z in zip(names,v)})
                rows.append(row)
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(out,index=False)
    out.with_suffix(".audit.json").write_text(json.dumps({"synthetic":True,"purpose":"Software verification only","n_per_release":args.n_per_release},indent=2))
    print(f"Wrote SYNTHETIC software fixture to {out}")


def convert_mat(args):
    raw=loadmat(args.input)
    actual_fs=float(np.asarray(raw[args.sfreq_key]).squeeze())
    if not np.isclose(actual_fs,100.):
        raise ValueError(f"MAT metadata reports {actual_fs} Hz; resample with anti-alias filtering upstream")
    permutation=[args.axes.index(v) for v in "ect"]
    names=json.loads(Path(args.channels).read_text())
    arrays={c:np.transpose(np.asarray(raw[k],float),permutation)*args.scale_to_volts
            for c,k in (("eo",args.eo_key),("ec",args.ec_key))}
    for c,x in arrays.items():
        if x.ndim!=3 or x.shape[1]!=len(names) or x.shape[2]!=400:
            raise ValueError(f"{c}: expected aligned 4-second epochs, axes e/c/t, at 100 Hz")
    if not args.confirm_cleaned:raise ValueError("MAT conversion is not artifact cleaning; confirm actual preprocessing")
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out,**arrays,ch_names=np.asarray(names,dtype="U"),sfreq=100.,units="V",cleaned=True)
    print(f"Converted {out}; no labels were inferred")


def plots(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import balanced_accuracy_score
    df=pd.read_csv(Path(args.results)/"oof_predictions.csv")
    fig,ax=plt.subplots(figsize=(6.8,4.2))
    for model,rows in df.groupby("model"):
        ds=sorted(rows.duration.unique())
        ys=[balanced_accuracy_score(rows[rows.duration==d].label,rows[rows.duration==d].prediction) for d in ds]
        ax.plot(ds,ys,"o-",label=model)
    ax.axhline(.5,color="gray",ls="--",lw=1)
    ax.set(xlabel="Clean seconds per condition",ylabel="Balanced accuracy",ylim=(0,1),xticks=ds)
    ax.legend(fontsize=8,ncol=2);ax.spines[["top","right"]].set_visible(False)
    if df.participant_id.astype(str).str.startswith("SYNTH_").any():
        ax.set_title("SYNTHETIC SOFTWARE CHECK - NOT HBN RESULTS",fontsize=10)
    fig.tight_layout();fig.savefig(Path(args.results)/"duration_curve.pdf");plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest="command",required=True)
    p=sub.add_parser("extract");p.add_argument("--manifest",required=True);p.add_argument("--regions",required=True)
    p.add_argument("--age-min",type=float,default=6);p.add_argument("--age-max",type=float,default=17)
    p.add_argument("--output",required=True);p.add_argument("--durations",type=int,nargs="+",default=[20,40,60]);p.set_defaults(func=extract)
    p=sub.add_parser("run");p.add_argument("--features",required=True);p.add_argument("--config",required=True);p.add_argument("--output",required=True)
    p.set_defaults(func=lambda a:evaluate(pd.read_csv(a.features,dtype={"participant_id":str,"group_id":str,"release":str,"sex":str}),json.loads(Path(a.config).read_text()),a.output))
    p=sub.add_parser("report");p.add_argument("--results",required=True);p.add_argument("--output",required=True)
    p.add_argument("--duration",type=int,default=40)
    p.set_defaults(func=lambda a:__import__("hbn_r3.report",fromlist=["summarise"]).summarise(a.results,a.output,a.duration))
    p=sub.add_parser("demo");p.add_argument("--output",required=True);p.add_argument("--n-per-release",type=int,default=32);p.add_argument("--seed",type=int,default=2026);p.set_defaults(func=demo)
    p=sub.add_parser("convert-mat");p.add_argument("--input",required=True);p.add_argument("--output",required=True)
    p.add_argument("--eo-key",required=True);p.add_argument("--ec-key",required=True)
    p.add_argument("--sfreq-key",required=True)
    p.add_argument("--axes",required=True,choices=["ect","etc","cet","cte","tec","tce"])
    p.add_argument("--channels",required=True);p.add_argument("--scale-to-volts",required=True,type=float)
    p.add_argument("--confirm-cleaned",action="store_true");p.set_defaults(func=convert_mat)
    p=sub.add_parser("plot");p.add_argument("--results",required=True);p.set_defaults(func=plots)
    from .inspection import inspect_data, validate_data, inspect_mat
    p=sub.add_parser("inspect-data");p.add_argument("--root",default=r"D:\EEG_Dataset\HBN_100Hz")
    p.add_argument("--output",default="outputs/inventory.csv");p.set_defaults(func=inspect_data)
    p=sub.add_parser("inspect-mat");p.add_argument("--input",required=True);p.set_defaults(func=inspect_mat)
    p=sub.add_parser("validate-data");p.add_argument("--manifest",required=True);p.add_argument("--regions",required=True)
    p.add_argument("--output",default="outputs/data_validation.csv");p.set_defaults(func=validate_data)
    from .bdf import inspect_bdf, prepare_bdf, prepare_cohort
    p=sub.add_parser("inspect-bdf");p.add_argument("--input",required=True);p.add_argument("--output",required=True)
    p.add_argument("--stim-channel");p.add_argument("--event-mask",type=int,default=65535);p.set_defaults(func=inspect_bdf)
    p=sub.add_parser("prepare-bdf");p.add_argument("--input",required=True);p.add_argument("--output",required=True)
    p.add_argument("--channels",required=True);p.add_argument("--intervals",required=True)
    p.add_argument("--reference",required=True,choices=["average","keep"])
    p.add_argument("--filter-mode",required=True,choices=["bandpass","keep"])
    p.add_argument("--reject-uv",type=float,default=150.);p.add_argument("--flat-uv",type=float,default=.1)
    p.add_argument("--trim-seconds",type=float,default=2.);p.add_argument("--confirm-preprocessing",action="store_true")
    p.set_defaults(func=prepare_bdf)
    p=sub.add_parser("prepare-cohort");p.add_argument("--manifest",required=True);p.add_argument("--output-dir",required=True)
    p.add_argument("--channels",required=True);p.add_argument("--reference",required=True,choices=["average","keep"])
    p.add_argument("--filter-mode",required=True,choices=["bandpass","keep"])
    p.add_argument("--reject-uv",type=float,default=150.);p.add_argument("--flat-uv",type=float,default=.1)
    p.add_argument("--trim-seconds",type=float,default=2.);p.add_argument("--confirm-preprocessing",action="store_true")
    p.set_defaults(func=prepare_cohort)
    args=parser.parse_args();args.func(args)


if __name__=="__main__":main()
