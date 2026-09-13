"""Release holdout with nested family-grouped tuning and honest uncertainty."""
from pathlib import Path
import json
import hashlib
import platform
import importlib.metadata
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedGroupKFold,ParameterGrid
from sklearn.metrics import balanced_accuracy_score,roc_auc_score,f1_score,confusion_matrix
from .model import R3Forest


def release_splits(X):
    for release in sorted(X.release.unique()):
        test=np.flatnonzero(X.release.to_numpy()==release)
        held_groups=set(X.iloc[test].group_id)
        train=np.flatnonzero((X.release.to_numpy()!=release)&~X.group_id.isin(held_groups).to_numpy())
        if len(train)==0:raise ValueError("No training observations remain after family blocking")
        assert not (set(X.iloc[train].participant_id)&set(X.iloc[test].participant_id))
        assert not (set(X.iloc[train].group_id)&set(X.iloc[test].group_id))
        yield release,train,test


def metrics(y,p,threshold=.5):
    y=np.asarray(y,int);pred=(np.asarray(p)>=threshold).astype(int)
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return {"balanced_accuracy":float(balanced_accuracy_score(y,pred)),
            "roc_auc":float(roc_auc_score(y,p)) if len(np.unique(y))==2 else float("nan"),
            "sensitivity":float(tp/(tp+fn)) if tp+fn else float("nan"),
            "specificity":float(tn/(tn+fp)) if tn+fp else float("nan"),
            "macro_f1":float(f1_score(y,pred,average="macro",zero_division=0)),
            "tn":int(tn),"fp":int(fp),"fn":int(fn),"tp":int(tp)}


def cluster_interval(rows,n_boot=1000,seed=2026):
    """Conditional uncertainty of fixed OOF predictions; not full training uncertainty."""
    rng=np.random.default_rng(seed)
    groups=rows.group_id.unique();pieces={g:rows[rows.group_id==g] for g in groups};samples=[]
    for _ in range(n_boot):
        boot=pd.concat([pieces[g] for g in rng.choice(groups,len(groups),replace=True)])
        if boot.label.nunique()==2:samples.append(balanced_accuracy_score(boot.label,boot.prediction))
    return np.quantile(samples,[.025,.975]).tolist() if samples else [None,None]


def paired_interval(rows,model="proposed",baseline="mi",duration=60,n_boot=1000,seed=2026):
    rows=rows.copy()
    if "prediction" not in rows:rows["prediction"]=(rows.probability>=.5).astype(int)
    p=rows[(rows.model==model)&(rows.duration==duration)]
    b=rows[(rows.model==baseline)&(rows.duration==duration)]
    joined=p.merge(b,on="participant_id",suffixes=("_p","_b"),validate="one_to_one")
    if len(joined)==0:return None
    if len(joined)!=len(p) or len(joined)!=len(b):raise ValueError("Paired models have different participant cohorts")
    if not np.array_equal(joined.group_id_p,joined.group_id_b):raise ValueError("Paired family IDs differ")
    if not np.array_equal(joined.label_p,joined.label_b):raise ValueError("Paired labels do not match")
    groups=joined.group_id_p.unique();pieces={g:joined[joined.group_id_p==g] for g in groups}
    rng=np.random.default_rng(seed);vals=[]
    for _ in range(n_boot):
        z=pd.concat([pieces[g] for g in rng.choice(groups,len(groups),replace=True)])
        if z.label_p.nunique()<2:continue
        vals.append(balanced_accuracy_score(z.label_p,z.prediction_p)-balanced_accuracy_score(z.label_p,z.prediction_b))
    point=balanced_accuracy_score(joined.label_p,joined.prediction_p)-balanced_accuracy_score(joined.label_p,joined.prediction_b)
    return {"model":model,"baseline":baseline,"duration":duration,"delta_balanced_accuracy":float(point),
            "conditional_cluster_ci95":(np.quantile(vals,[.025,.975]).tolist() if vals else [None,None]),"n_pairs":len(joined)}


def evaluate(table,config,outdir):
    outdir=Path(outdir)
    if outdir.exists() and any(outdir.iterdir()):
        raise ValueError("Output directory is not empty; choose a new run directory")
    outdir.mkdir(parents=True,exist_ok=True)
    required={"participant_id","group_id","release","label","duration","age","sex"}
    if not required.issubset(table):raise ValueError(f"Missing feature metadata: {required-set(table)}")
    if table[["participant_id","group_id","release","label","duration"]].isna().any().any():
        raise ValueError("Missing participant/group/release/label/duration")
    if not table.label.isin([0,1]).all():raise ValueError("Labels must be binary 0/1")
    from .data import RELEASES
    if set(table.release)!=set(RELEASES):raise ValueError("Primary protocol requires all four specified releases")
    packages={}
    for name in ["numpy","scipy","pandas","scikit-learn","mne","joblib"]:
        try:packages[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:pass
    fingerprint=hashlib.sha256(pd.util.hash_pandas_object(table,index=True).values.tobytes()).hexdigest()
    (outdir/"environment.json").write_text(json.dumps({"python":platform.python_version(),
        "platform":platform.platform(),"packages":packages,"feature_table_sha256":fingerprint,
        "synthetic":bool(table.participant_id.astype(str).str.startswith("SYNTH_").any())},indent=2))
    (outdir/"run_config.json").write_text(json.dumps(config,indent=2))
    training_duration=config.get("training_duration",60)
    long=table[table.duration==training_duration].copy().reset_index(drop=True)
    if long.participant_id.duplicated().any():raise ValueError("Duplicate participants at training duration")
    if long.empty:raise ValueError("No participants at training duration")
    durations=sorted(table.duration.unique())
    # Enforce paired participant cohort across durations, rather than quietly changing sample size.
    for d in durations:
        sub=table[table.duration==d]
        if sub.participant_id.duplicated().any() or set(sub.participant_id)!=set(long.participant_id):
            raise ValueError("Every duration must contain exactly the same unique participants")
        aligned=sub.set_index("participant_id").loc[long.participant_id]
        for field in ("label","group_id","release"):
            if not np.array_equal(aligned[field].to_numpy(),long[field].to_numpy()):
                raise ValueError(f"Inconsistent {field} across durations")
    predictions=[];summary=[];tuning=[];splits=[];budget_predictions=[]
    seed=config.get("seed",2026)
    for held,train,test in release_splits(long):
        tr=long.iloc[train];te=long.iloc[test];y=tr.label.to_numpy(int)
        if te.label.nunique()!=2:raise ValueError(f"Held-out release {held} lacks one class")
        strata=tr.release.astype(str)+"|"+tr.label.astype(str)
        cv=StratifiedGroupKFold(config.get("inner_folds",3),shuffle=True,random_state=seed)
        inner=list(cv.split(tr,strata,groups=tr.group_id))
        for it,iv in inner:
            if set(tr.iloc[it].group_id)&set(tr.iloc[iv].group_id):raise AssertionError("Family leakage")
            if tr.iloc[it].label.nunique()!=2 or tr.iloc[iv].label.nunique()!=2:raise ValueError("Inner fold missing class; reduce folds")
        splits.append({"held_release":held,"train_ids":tr.participant_id.tolist(),"test_ids":te.participant_id.tolist(),
            "inner":[{"train_ids":tr.iloc[a].participant_id.tolist(),"validation_ids":tr.iloc[b].participant_id.tolist()} for a,b in inner]})
        for mode in config.get("models",["bands","static","mi","reliability","proposed","demographics"]):
            print(f"Training {mode}; held release {held}; n={len(tr)}/{len(te)}",flush=True)
            spec=config.get("model_specs",{}).get(mode,{"mode":mode})
            base=R3Forest(n_estimators=config.get("n_estimators",500),
                residualize=config.get("residualize",False),random_state=seed,**spec)
            grid=dict(config.get("model_grids",{}).get(mode,config.get("grid",{"k":[20,40]})))
            if base.mode=="proposed" and "penalty" not in grid:grid["penalty"]=config.get("penalties",[.5,1.])
            if base.mode=="demographics":grid.pop("k",None)
            best_score=-np.inf;best_params=None
            for params in ParameterGrid(grid):
                scores=[]
                for it,iv in inner:
                    fitted=clone(base).set_params(**params).fit(tr.iloc[it],y[it])
                    scores.append(balanced_accuracy_score(y[iv],fitted.predict(tr.iloc[iv])))
                mean=float(np.mean(scores))
                tuning.append({"held_release":held,"model":mode,"params":json.dumps(params),"inner_ba":mean})
                if mean>best_score:best_score=mean;best_params=params
            fitted=clone(base).set_params(**best_params).fit(tr,y)
            joblib.dump(fitted,outdir/f"model_{held}_{mode}.joblib")
            (outdir/f"augmentation_{held}_{mode}.json").write_text(json.dumps(fitted.augmentation_report_,indent=2))
            fitted.feature_report().to_csv(outdir/f"features_{held}_{mode}.csv",index=False)
            if config.get("budget_policy",False) and base.mode!="demographics":
                from .budget import evidence,fit_policy,apply_policy
                inner_models=[(clone(base).set_params(**best_params).fit(tr.iloc[it],y[it]),iv) for it,iv in inner]
                for agreement in (False,True):
                    calibration=[];held_rows=[]
                    for source_models,target in [(inner_models,calibration),([(fitted,None)],held_rows)]:
                        for fitted_inner,iv in source_models:
                            ids=te.participant_id if iv is None else tr.iloc[iv].participant_id
                            for duration in durations:
                                frame=table[table.duration==duration].set_index("participant_id").loc[ids].reset_index()
                                margin,confidence=evidence(fitted_inner,frame,agreement)
                                part=frame[["participant_id","group_id","release","label","duration"]].copy()
                                part["margin"]=margin;part["confidence"]=confidence;target.append(part)
                    calibration=pd.concat(calibration,ignore_index=True)
                    policy=fit_policy(calibration,target_error=config.get("budget_target_error",.1),
                        min_coverage=config.get("budget_min_coverage",.5),min_per_class=config.get("budget_min_per_class",5))
                    policy_name="agreement" if agreement else "margin"
                    (outdir/f"policy_{held}_{mode}_{policy_name}.json").write_text(json.dumps(policy,indent=2))
                    calibration.to_csv(outdir/f"policy_inner_{held}_{mode}_{policy_name}.csv",index=False)
                    chosen=apply_policy(pd.concat(held_rows,ignore_index=True),policy)
                    chosen["model"]=mode;chosen["policy"]=policy_name;budget_predictions.append(chosen)
            for duration in durations:
                test_short=table[table.duration==duration].set_index("participant_id").loc[te.participant_id].reset_index()
                is_margin=fitted.classifier=="svm"
                p=fitted.decision_function(test_short) if is_margin else fitted.predict_proba(test_short)[:,1]
                threshold=0. if is_margin else .5
                m=metrics(test_short.label,p,threshold)
                summary.append({"held_release":held,"model":mode,"duration":int(duration),"n":len(p),**m})
                for (_,row),prob in zip(test_short.iterrows(),p):
                    predictions.append({"participant_id":row.participant_id,"group_id":row.group_id,"release":held,
                      "label":int(row.label),"score":float(prob),"threshold":threshold,
                      "prediction":int(prob>=threshold),"score_kind":"margin" if is_margin else "probability",
                      "probability":None if is_margin else float(prob),"model":mode,"duration":int(duration)})
    pred=pd.DataFrame(predictions);scores=pd.DataFrame(summary)
    pred.to_csv(outdir/"oof_predictions.csv",index=False)
    scores.to_csv(outdir/"release_metrics.csv",index=False)
    pd.DataFrame(tuning).to_csv(outdir/"inner_tuning.csv",index=False)
    with open(outdir/"splits.json","w") as f:json.dump(splits,f,indent=2)
    pooled=[]
    for (mode,duration),rows in pred.groupby(["model","duration"]):
        pooled.append({"model":mode,"duration":int(duration),"n":len(rows),**metrics(rows.label,rows.score,float(rows.threshold.iloc[0])),
             "ba_conditional_ci95":cluster_interval(rows,config.get("bootstrap",1000),seed)})
    with open(outdir/"pooled_metrics.json","w") as f:json.dump(pooled,f,indent=2)
    primary=config.get("primary_contrast",["proposed","mi"])
    comparison=paired_interval(pred,model=primary[0],baseline=primary[1],duration=training_duration,n_boot=config.get("bootstrap",1000),seed=seed)
    with open(outdir/"primary_comparison.json","w") as f:json.dump(comparison,f,indent=2)
    contrasts=config.get("comparisons",[["proposed","mi"]])
    comparisons=[paired_interval(pred,a,b,training_duration,config.get("bootstrap",1000),seed) for a,b in contrasts]
    (outdir/"paired_comparisons.json").write_text(json.dumps(comparisons,indent=2))
    selected=[]
    for report in sorted(outdir.glob("features_*.csv")):
        tab=pd.read_csv(report);selected.extend(tab.loc[tab.selected,"feature"].tolist())
    # Model-specific cross-release selection frequencies (descriptive, four fits only).
    frequencies=[]
    for mode in config.get("models",[]):
        tabs=[pd.read_csv(outdir/f"features_{r}_{mode}.csv") for r in sorted(long.release.unique())]
        counts=pd.concat([t.loc[t.selected,["feature"]] for t in tabs]).feature.value_counts()
        frequencies.extend({"model":mode,"feature":f,"outer_folds_selected":int(n),"total_outer_folds":len(tabs)} for f,n in counts.items())
    pd.DataFrame(frequencies).to_csv(outdir/"selection_frequency.csv",index=False)
    if budget_predictions:
        from .budget import policy_metrics
        budget=pd.concat(budget_predictions,ignore_index=True)
        budget.to_csv(outdir/"budget_predictions.csv",index=False)
        reports=[]
        for (model,policy),rows in budget.groupby(["model","policy"]):
            reports.append({"model":model,"policy":policy,"release":"pooled",**policy_metrics(rows)})
            for release,part in rows.groupby("release"):
                reports.append({"model":model,"policy":policy,"release":release,**policy_metrics(part)})
        (outdir/"budget_metrics.json").write_text(json.dumps(reports,indent=2))
    return pred,scores
