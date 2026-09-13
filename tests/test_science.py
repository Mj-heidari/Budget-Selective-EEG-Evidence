import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
from scipy.io import savemat
from sklearn.base import clone
from hbn_r3.features import spectral_model,spectral_parameters,debiased_wpli,extract_participant
from hbn_r3.data import read_manifest,load_epochs,RELEASES
from hbn_r3.model import R3Forest,consistency
from hbn_r3.evaluation import release_splits,paired_interval
from hbn_r3.cli import demo,convert_mat


def fixture():
    rng=np.random.default_rng(18);n=96;p=24;y=np.tile([0,1],n//2)
    latent=rng.normal(size=(n,p));latent[:,:5]+=y[:,None]
    a=latent+rng.normal(0,.1,(n,p));b=latent+rng.normal(0,.1,(n,p))
    X=pd.DataFrame({"participant_id":[f"s{i}" for i in range(n)],"group_id":[f"g{i}" for i in range(n)],
      "age":rng.uniform(6,17,n),"sex":np.tile(["M","F","F"],n//3),"release":np.repeat(RELEASES,n//4)})
    for j in range(p):
        for prefix,z in (("f__",latent),("a__",a),("b__",b)):X[prefix+f"eo_sp_LF_theta{j}"]=z[:,j]
    return X,y


class ScientificChecks(unittest.TestCase):
    def test_spectral_fit_recovers_known_parameters(self):
        f=np.arange(0,50.25,.25);p=np.zeros_like(f)
        p[1:]=10**spectral_model(f[1:],[-10,1.6,.6,10,1.2])
        vals,r2=spectral_parameters(f,p)
        np.testing.assert_allclose(vals,[1.6,-10,10,.6],atol=.02)
        self.assertGreater(r2,.999)

    def test_absent_alpha_is_not_invented(self):
        f=np.arange(0,50.25,.25);p=np.ones_like(f);p[1:]=1e-10/f[1:]**1.5
        vals,_=spectral_parameters(f,p)
        self.assertTrue(np.isnan(vals[2:]).all())

    def test_wpli_handles_lag_and_no_information(self):
        np.testing.assert_allclose(debiased_wpli(np.ones((12,3))),1)
        self.assertTrue(np.isnan(debiased_wpli(np.zeros((12,3)))).all())
        im=np.array([1.,-1.,1.,-1.])[:,None]
        self.assertLess(debiased_wpli(im)[0],0)

    def test_feature_count_and_reactivity_identity(self):
        rng=np.random.default_rng(4);t=np.arange(400)/100
        x=rng.normal(0,1e-6,(6,6,400))
        for c in range(6):x[:,c]+=2e-6*np.sin(2*np.pi*10*t+c*.2)
        regions={str(i):[i] for i in range(6)}
        row=extract_participant({"eo":x,"ec":x*1.1},100,regions,24)
        self.assertEqual(sum(k.startswith("f__") for k in row),315)
        for k,v in row.items():
            if k.startswith("f__delta_"):
                name=k[len("f__delta_"):]
                a=row["f__ec_"+name]-row["f__eo_"+name]
                if np.isfinite(a):self.assertAlmostEqual(v,a)

    def test_reliability_does_not_reward_only_class_means(self):
        rng=np.random.default_rng(7);n=1000;y=np.tile([0,1],n//2);r=np.repeat(RELEASES,n//4)
        a=10*y[:,None]+rng.normal(size=(n,1));b=10*y[:,None]+rng.normal(size=(n,1))
        self.assertLess(consistency(a,b,y,r)[0],.15)
        self.assertGreater(consistency(a,a,y,r)[0],.999)

    def test_release_split_blocks_cross_release_relatives(self):
        X,y=fixture();X.loc[0,"group_id"]="family";X.loc[25,"group_id"]="family"
        for held,tr,te in release_splits(X):
            self.assertFalse(set(X.iloc[tr].group_id)&set(X.iloc[te].group_id))
            self.assertFalse(set(X.iloc[tr].release)&{held})

    def test_prediction_cannot_refit_imputation_or_selector(self):
        X,y=fixture();train=np.arange(72);test=np.arange(72,96)
        model=R3Forest(k=8,n_estimators=12,min_samples_leaf=2).fit(X.iloc[train],y[train])
        med=model.medians_.copy();score=model.scores_.copy();chosen=model.selected_.copy()
        hostile=X.iloc[test].copy();cols=[c for c in X if c.startswith("f__")]
        hostile.loc[:,cols]=1e10
        p=model.predict_proba(hostile)
        np.testing.assert_array_equal(model.medians_,med)
        np.testing.assert_array_equal(model.scores_,score)
        np.testing.assert_array_equal(model.selected_,chosen)
        self.assertEqual(p.shape,(24,2));self.assertTrue(np.isfinite(p).all())
        self.assertEqual(clone(model).k,8)

    def test_missing_or_duplicate_diagnoses_fail(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/"dummy.npz").touch()
            df=pd.DataFrame({"participant_id":["x","x"],"release":[RELEASES[0]]*2,"path":["dummy.npz"]*2,
                "label":[0,1],"age":[10,11],"sex":["F","M"],"group_id":["x","x"],
                "diagnosis_confirmed":[True,True],"cleaning_provenance":["audited"]*2})
            df.to_csv(p/"m.csv",index=False)
            with self.assertRaisesRegex(ValueError,"Duplicate"):read_manifest(p/"m.csv")
            df.loc[1,"participant_id"]="y";df.loc[1,"label"]=np.nan;df.to_csv(p/"m.csv",index=False)
            with self.assertRaisesRegex(ValueError,"labels"):read_manifest(p/"m.csv")

    def test_mat_adapter_axes_units_and_sampling(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);rng=np.random.default_rng(2);x=rng.normal(size=(3,400,6))
            savemat(p/"source.mat",{"open":x,"closed":x,"fs":100});(p/"names.json").write_text('["A","B","C"]')
            args=SimpleNamespace(input=p/"source.mat",output=p/"out.npz",eo_key="open",ec_key="closed",axes="cte",
                    channels=p/"names.json",scale_to_volts=1e-6,confirm_cleaned=True,sfreq_key="fs")
            convert_mat(args);data,fs,names=load_epochs(p/"out.npz")
            self.assertEqual(data["eo"].shape,(6,3,400));self.assertEqual(fs,100)
            np.testing.assert_allclose(data["eo"],x.transpose(2,0,1)*1e-6)


if __name__=="__main__":unittest.main()
