import unittest
import tempfile
import json
from pathlib import Path
import numpy as np
from hbn_r3.gan import Network, moment_loss_gradient
from hbn_r3.augmentation import augment, support_mask
from hbn_r3.model import R3Forest
from test_science import fixture

class ExtensionChecks(unittest.TestCase):
    def test_network_gradients_finite_difference(self):
        rng=np.random.default_rng(5);net=Network([3,4,2],rng);x=rng.normal(size=(5,3));out,cache=net.forward(x)
        dx,grads=net.backward(out,cache);eps=1e-6
        for pi,index in [(0,(1,2)),(1,(2,)),(2,(0,1)),(3,(1,))]:
            old=net.params[pi][index]
            net.params[pi][index]=old+eps;plus=.5*np.sum(net.forward(x)[0]**2)
            net.params[pi][index]=old-eps;minus=.5*np.sum(net.forward(x)[0]**2)
            net.params[pi][index]=old
            self.assertAlmostEqual((plus-minus)/(2*eps),grads[pi][index],places=6)
        xp=x.copy();xp[0,0]+=eps;xm=x.copy();xm[0,0]-=eps
        self.assertAlmostEqual((np.sum(net.forward(xp)[0]**2)-np.sum(net.forward(xm)[0]**2))/(4*eps),dx[0,0],places=6)

    def test_moment_gradient_finite_difference(self):
        rng=np.random.default_rng(3);fake=rng.normal(size=(7,3));real=rng.normal(size=(12,3));w=np.array([.2,.5,1.])
        loss,g=moment_loss_gradient(fake,real,w);eps=1e-6
        for idx in [(0,0),(2,1),(5,2)]:
            p=fake.copy();m=fake.copy();p[idx]+=eps;m[idx]-=eps
            self.assertAlmostEqual((moment_loss_gradient(p,real,w)[0]-moment_loss_gradient(m,real,w)[0])/(2*eps),g[idx],places=6)

    def test_gate_rejects_copy_and_far_outlier(self):
        real=np.array([[0.,0.],[1.,0.],[0.,1.],[1.,1.]])
        keep,_,_=support_mask(real,np.array([[0.,0.],[100.,100.],[.5,.5]]))
        np.testing.assert_array_equal(keep,[False,False,True])

    def test_gan_is_deterministic_and_keeps_real_rows(self):
        rng=np.random.default_rng(9);z=rng.normal(size=(32,4));y=np.tile([0,1],16);r=np.repeat(['a','b'],16)
        for method in ['ros','jitter','gan','gan_gated']:
            a,b,report=augment(z,y,r,method,.5,8,steps=4,batch_size=8,moment_weight=.1)
            aa,bb,_=augment(z,y,r,method,.5,8,steps=4,batch_size=8,moment_weight=.1)
            np.testing.assert_array_equal(a,aa);np.testing.assert_array_equal(b,bb)
            np.testing.assert_array_equal(a[:32],z);self.assertEqual(report['requested'],16)
            self.assertLessEqual(report['accepted'],16);self.assertTrue(np.isfinite(a).all())

    def test_all_classifiers_freeze_training_state(self):
        X,y=fixture()
        for method in ['rf','svm','logistic']:
            model=R3Forest(classifier=method,k=8,n_estimators=8,augmentation='gan_gated',gan_steps=3,gan_batch_size=8).fit(X.iloc[:72],y[:72])
            med=model.medians_.copy();mean=model.scaler_.mean_.copy();scores=model.scores_.copy()
            hostile=X.iloc[72:].copy();hostile.loc[:,[c for c in X if c.startswith('f__')]]=1e5
            score=model.decision_function(hostile)
            self.assertTrue(np.isfinite(score).all())
            np.testing.assert_array_equal(model.medians_,med);np.testing.assert_array_equal(model.scaler_.mean_,mean)
            np.testing.assert_array_equal(model.scores_,scores)
            if method=='svm':
                with self.assertRaises(AttributeError):model.predict_proba(hostile)

    def test_invalid_modes_and_nonbinary_labels_fail(self):
        X,y=fixture()
        with self.assertRaises(ValueError):R3Forest(mode='typo').fit(X,y)
        with self.assertRaises(ValueError):R3Forest().fit(X,y+.1)

class BDFEpochChecks(unittest.TestCase):
    def test_boundaries_trim_rejection_and_overlap(self):
        import pandas as pd
        from hbn_r3.bdf import epoch_arrays
        t=np.arange(8000)/100;x=np.tile(np.sin(2*np.pi*10*t)*1e-5,(3,1))
        intervals=pd.DataFrame({'onset_seconds':[0,40],'duration_seconds':[40,40],'condition':['eo','ec']})
        arrays,audit=epoch_arrays(x,100,intervals)
        self.assertEqual(arrays['eo'].shape,(9,3,400));self.assertEqual(arrays['ec'].shape,(9,3,400))
        self.assertEqual(audit[0]['onset_seconds'],2)
        arrays,audit=epoch_arrays(x,100,intervals,bad_intervals=[(2,6)])
        self.assertEqual(len(arrays['eo']),8)
        intervals.loc[1,'onset_seconds']=30
        with self.assertRaisesRegex(ValueError,'overlap'):epoch_arrays(x,100,intervals)

class BudgetChecks(unittest.TestCase):
    def test_progressive_subsets_are_nested(self):
        from hbn_r3.features import progressive_indices
        for total in [15,20,27,51]:
            a,b,c=[set(progressive_indices(total,n)) for n in [5,10,15]]
            self.assertTrue(a<b<c);self.assertEqual(len(c),15)

    def test_policy_cannot_accept_impossible_target(self):
        import pandas as pd
        from hbn_r3.budget import fit_policy,apply_policy,policy_metrics
        rows=pd.DataFrame([{'participant_id':str(i),'group_id':str(i),'release':'r','label':i%2,
            'duration':d,'margin':1 if i%2==0 else -1,'confidence':.8} for i in range(20) for d in [20,40,60]])
        policy=fit_policy(rows);self.assertFalse(policy['feasible'])
        result=apply_policy(rows,policy);self.assertEqual(policy_metrics(result)['coverage'],0.)
        self.assertTrue((result.prediction==-1).all());self.assertTrue((result.duration==60).all())

    def test_policy_is_frozen_and_counts_budget_once(self):
        import pandas as pd
        from hbn_r3.budget import fit_policy,apply_policy,policy_metrics
        rows=pd.DataFrame([{'participant_id':str(i),'group_id':str(i),'release':'r','label':i%2,
            'duration':d,'margin':1 if i%2 else -1,'confidence':.1+i/40} for i in range(20) for d in [20,40,60]])
        policy=fit_policy(rows,min_per_class=3);saved=json.dumps(policy,sort_keys=True)
        result=apply_policy(rows,policy)
        self.assertTrue(policy['feasible']);self.assertEqual(len(result),20)
        self.assertEqual(json.dumps(policy,sort_keys=True),saved)
        self.assertEqual(policy_metrics(result)['accepted_accuracy'],1.)
