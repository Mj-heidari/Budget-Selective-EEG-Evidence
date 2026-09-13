"""All cohort-fitted transformations live inside this cloneable estimator."""
import warnings
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator,ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from .augmentation import augment
from sklearn.feature_selection import mutual_info_classif


def consistency(a,b,y,releases):
    """Within-class/release centered Pearson consistency, Spearman-Brown corrected.

    This is an exploratory internal-consistency estimate, not test-retest reliability.
    Missing fits are excluded pairwise; at least 12 observations are required.
    """
    out=np.zeros(a.shape[1])
    for j in range(a.shape[1]):
        valid=np.isfinite(a[:,j])&np.isfinite(b[:,j])
        if valid.sum()<12:continue
        aa=a[valid,j].copy();bb=b[valid,j].copy()
        groups=np.array([str(r)+"|"+str(t) for r,t in zip(releases[valid],y[valid])])
        for g in np.unique(groups):
            mask=groups==g
            aa[mask]-=aa[mask].mean();bb[mask]-=bb[mask].mean()
        if aa.std()<1e-10 or bb.std()<1e-10:continue
        rho=np.corrcoef(aa,bb)[0,1]
        out[j]=np.clip(2*rho/(1+rho),0,1) if rho>0 else 0
    return out


def release_statistics(z,y,releases,seed):
    """Equal release weighting for MI and standardized effect heterogeneity."""
    information=[];effects=[]
    for rel in np.unique(releases):
        keep=releases==rel
        yk=y[keep];x=z[keep]
        if min(np.sum(yk==0),np.sum(yk==1))<3:continue
        mi=mutual_info_classif(x,yk,random_state=seed,n_neighbors=2)
        information.append(mi)
        x0=x[yk==0];x1=x[yk==1]
        pooled=np.sqrt(((len(x0)-1)*x0.var(0,ddof=1)+(len(x1)-1)*x1.var(0,ddof=1))/(len(x)-2))
        effects.append(np.clip((x1.mean(0)-x0.mean(0))/np.maximum(pooled,1e-8),-5,5))
    if not information:
        raise ValueError("No training release has at least three independent observations per class")
    mi=np.mean(information,axis=0)
    heterogeneity=np.std(effects,axis=0) if len(effects)>1 else np.zeros(z.shape[1])
    return mi,heterogeneity,len(effects)


class R3Forest(ClassifierMixin,BaseEstimator):
    def __init__(self,mode="proposed",k=40,penalty=1.,n_estimators=500,
                 min_samples_leaf=5,max_depth=None,residualize=False,random_state=2026,
                 classifier="rf",C=1.,gamma="scale",augmentation="none",augmentation_ratio=.5,
                 gan_steps=400,gan_batch_size=32,gan_critic_steps=3,moment_weight=0.):
        self.mode=mode;self.k=k;self.penalty=penalty;self.n_estimators=n_estimators
        self.min_samples_leaf=min_samples_leaf;self.max_depth=max_depth
        self.residualize=residualize;self.random_state=random_state
        self.classifier=classifier;self.C=C;self.gamma=gamma
        self.augmentation=augmentation;self.augmentation_ratio=augmentation_ratio
        self.gan_steps=gan_steps;self.gan_batch_size=gan_batch_size
        self.gan_critic_steps=gan_critic_steps;self.moment_weight=moment_weight

    def _covariates(self,X,fit=False):
        age=pd.to_numeric(X.age,errors="coerce").to_numpy(float)
        if fit:
            if not np.isfinite(age).any():raise ValueError("Age entirely missing")
            self.age_median_=np.nanmedian(age);self.age_mean_=np.nanmean(age)
            self.age_sd_=max(np.nanstd(age),1.)
            self.sex_levels_=sorted(X.sex.fillna("unknown").astype(str).unique())
        age=np.where(np.isfinite(age),age,self.age_median_)
        a=(age-self.age_mean_)/self.age_sd_
        sex=X.sex.fillna("unknown").astype(str).to_numpy()
        dummy=np.column_stack([(sex==s).astype(float) for s in self.sex_levels_[1:]]) if len(self.sex_levels_)>1 else np.empty((len(X),0))
        return np.column_stack([np.ones(len(X)),a,a*a,dummy])

    def fit(self,X,y):
        if not isinstance(X,pd.DataFrame):raise TypeError("Use the documented feature DataFrame")
        if self.mode not in {"bands","static","mi","reliability","proposed","demographics"}:
            raise ValueError("Unknown selection mode")
        y=np.asarray(y)
        if not np.isin(y,[0,1]).all():raise ValueError("Labels must be binary 0/1")
        y=y.astype(int)
        if set(y)!={0,1}:raise ValueError("Every model training fold needs both classes")
        self.classes_=np.array([0,1]);self.n_features_in_=X.shape[1]
        cov=self._covariates(X,fit=True)
        if self.mode=="demographics":
            self.feature_names_=["age","age_squared"]+[f"sex_{s}" for s in self.sex_levels_[1:]]
            z=cov[:,1:];self.selected_=np.arange(z.shape[1]);self.scores_=np.ones(z.shape[1])
        else:
            names=[c[3:] for c in X.columns if c.startswith("f__")]
            if self.mode=="bands":
                names=[n for n in names if not n.startswith("delta_") and "_sp_" in n and n.rsplit("_",1)[-1] in ("theta","alpha","beta")]
            elif self.mode=="static":names=[n for n in names if not n.startswith("delta_")]
            raw=X[["f__"+n for n in names]].to_numpy(float)
            keep=np.mean(np.isfinite(raw),axis=0)>=.5
            self.feature_names_=[n for n,k in zip(names,keep) if k]
            raw=raw[:,keep]
            if raw.shape[1]==0:raise ValueError("No feature passes 50% finite training-value rule")
            raw=np.where(np.isfinite(raw),raw,np.nan)
            self.medians_=np.nanmedian(raw,axis=0)
            z=np.where(np.isfinite(raw),raw,self.medians_)
            self.coef_=None
            if self.residualize:
                reg=np.eye(cov.shape[1])*1e-6;reg[0,0]=0
                self.coef_=np.linalg.solve(cov.T@cov+reg,cov.T@z)
                z=z-cov@self.coef_
            releases=X.release.to_numpy(str)
            mi,hetero,nrel=release_statistics(z,y,releases,self.random_state)
            rel=np.ones(z.shape[1])
            if self.mode in ("reliability","proposed"):
                a=X[["a__"+n for n in self.feature_names_]].to_numpy(float)
                b=X[["b__"+n for n in self.feature_names_]].to_numpy(float)
                if self.coef_ is not None:a=a-cov@self.coef_;b=b-cov@self.coef_
                rel=consistency(a,b,y,releases)
            if self.mode=="proposed" and nrel<2:
                raise ValueError("Release heterogeneity requires at least two eligible training releases")
            lam=self.penalty if self.mode=="proposed" else 0.
            self.scores_=mi*np.sqrt(rel)*np.exp(-lam*hetero)
            if not np.any(self.scores_>0):
                raise ValueError("No positive feature score: insufficient reliability or discriminative evidence")
            self.selected_=np.argsort(-self.scores_,kind="stable")[:min(self.k,len(self.scores_))]
            self.reliability_=rel;self.heterogeneity_=hetero;self.mi_=mi
        selected=z[:,self.selected_]
        self.scaler_=StandardScaler().fit(selected)
        selected=self.scaler_.transform(selected)
        weights=(np.sqrt(self.reliability_[self.selected_])*np.exp(-self.penalty*self.heterogeneity_[self.selected_])
                 if hasattr(self,"reliability_") else np.ones(selected.shape[1]))
        fit_z,fit_y,self.augmentation_report_=augment(selected,y,X.release.to_numpy(str),
            self.augmentation,self.augmentation_ratio,self.random_state,self.gan_steps,
            self.gan_batch_size,self.gan_critic_steps,self.moment_weight,weights)
        if self.classifier=="rf":
            self.forest_=RandomForestClassifier(n_estimators=self.n_estimators,max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,max_features="sqrt",class_weight="balanced",
                random_state=self.random_state,n_jobs=1)
        elif self.classifier=="logistic":
            self.forest_=LogisticRegression(C=self.C,class_weight="balanced",max_iter=5000,random_state=self.random_state)
        elif self.classifier=="svm":
            # No hidden, non-grouped probability-calibration CV.
            self.forest_=SVC(C=self.C,gamma=self.gamma,kernel="rbf",class_weight="balanced",probability=False)
        else:raise ValueError("classifier must be rf, logistic, or svm")
        self.forest_.fit(fit_z,fit_y)
        return self

    def _transform(self,X):
        cov=self._covariates(X)
        if self.mode=="demographics":return self.scaler_.transform(cov[:,1:])
        z=X[["f__"+n for n in self.feature_names_]].to_numpy(float)
        z=np.where(np.isfinite(z),z,self.medians_)
        if self.coef_ is not None:z=z-cov@self.coef_
        return self.scaler_.transform(z[:,self.selected_])

    def predict_proba(self,X):
        if self.classifier=="svm":
            raise AttributeError("SVM outputs margins, not calibrated probabilities; use decision_function")
        return self.forest_.predict_proba(self._transform(X))

    def decision_function(self,X):
        if self.classifier=="svm":return self.forest_.decision_function(self._transform(X))
        return self.predict_proba(X)[:,1]-.5

    def predict(self,X):return (self.decision_function(X)>=0).astype(int)

    def feature_report(self):
        chosen=set(self.selected_)
        out={"feature":self.feature_names_,"score":self.scores_,
             "selected":[i in chosen for i in range(len(self.feature_names_))]}
        if hasattr(self,"mi_"):out.update(mi=self.mi_,consistency=self.reliability_,heterogeneity=self.heterogeneity_)
        return pd.DataFrame(out).sort_values("score",ascending=False)
