"""Nested, empirically tuned clean-data budget and abstention policies.

No risk guarantee is asserted under release shift. Targets apply to inner real
out-of-fold predictions; held-out results must be reported with coverage.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score


def evidence(model,frame,split_agreement=True):
    full=model.decision_function(frame)
    if not split_agreement:return full,np.abs(full)
    margins=[full]
    for prefix in ('a__','b__'):
        alternate=frame.copy()
        for name in model.feature_names_:
            if prefix+name not in frame:raise ValueError('Split-view budget policy requires half-feature columns')
            alternate['f__'+name]=frame[prefix+name]
        margins.append(model.decision_function(alternate))
    values=np.stack(margins)
    agree=np.all((values>=0)==(values[0]>=0),axis=0)
    confidence=np.where(agree,np.min(np.abs(values),axis=0),0.)
    return full,confidence


def apply_policy(rows,policy):
    output=[]
    for participant,person in rows.groupby('participant_id',sort=False):
        person=person.sort_values('duration');chosen=person.iloc[-1];accepted=False
        if policy['feasible']:
            for _,row in person.iterrows():
                threshold=policy['thresholds'][str(int(row.duration))]
                if row.confidence>threshold:
                    chosen=row;accepted=True;break
        output.append({'participant_id':participant,'group_id':chosen.group_id,'release':chosen.release,
            'label':int(chosen.label),'accepted':accepted,'duration':int(chosen.duration),
            'prediction':int(chosen.margin>=0) if accepted else -1,'confidence':float(chosen.confidence)})
    return pd.DataFrame(output)


def policy_metrics(rows):
    accepted=rows[rows.accepted];n=len(rows)
    return {'n':n,'accepted':len(accepted),'coverage':len(accepted)/n,
        'coverage_control':float(rows[rows.label==0].accepted.mean()) if (rows.label==0).any() else None,
        'coverage_adhd':float(rows[rows.label==1].accepted.mean()) if (rows.label==1).any() else None,
        'mean_clean_seconds_per_condition':float(rows.duration.mean()),
        'accepted_accuracy':float((accepted.prediction==accepted.label).mean()) if len(accepted) else None,
        'accepted_balanced_accuracy':float(balanced_accuracy_score(accepted.label,accepted.prediction)) if accepted.label.nunique()==2 else None,
        'accepted_controls':int((accepted.label==0).sum()),'accepted_adhd':int((accepted.label==1).sum())}


def fit_policy(inner_rows,target_error=.1,min_coverage=.5,min_per_class=5):
    if not 0<target_error<.5 or not 0<min_coverage<=1:raise ValueError('Invalid budget policy target')
    best=None;report=[]
    # Shared quantile level across durations: a deliberately small policy class.
    for quantile in [0.,.25,.5,.7,.85,.95]:
        thresholds={str(int(d)):float(np.quantile(t.confidence,quantile)) for d,t in inner_rows.groupby('duration')}
        policy={'feasible':True,'thresholds':thresholds,'quantile':quantile}
        used=apply_policy(inner_rows,policy);m=policy_metrics(used)
        feasible=(m['coverage']>=min_coverage and min(m['accepted_controls'],m['accepted_adhd'])>=min_per_class
            and m['accepted_balanced_accuracy'] is not None and m['accepted_balanced_accuracy']>=1-target_error
            and m['accepted_accuracy']>=1-target_error)
        report.append({'quantile':quantile,'feasible':feasible,**m})
        rank=(m['mean_clean_seconds_per_condition'],-m['coverage'])
        if feasible and (best is None or rank<best[0]):best=(rank,policy)
    result=best[1] if best else {'feasible':False,'thresholds':{},'quantile':None}
    return {**result,'target_error':target_error,'min_coverage':min_coverage,'min_per_class':min_per_class,
        'selection_data':'real inner out-of-fold predictions only','risk_guarantee':False,'candidates':report}
