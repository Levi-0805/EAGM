import unittest
from unittest.mock import patch
import numpy as np
from eagm2.data import collect, sequences
from eagm2.extra_data import collect_extra_training
from eagm2.runtime import Budget,stream_seed

class CountingEnv:
    control_dt=1/30;alpha=.5
    lower=np.full(14,-1.);upper=np.full(14,1.)
    def __init__(self,fail_at=None):self.calls=0;self.fail_at=fail_at
    def reset(self,seed,delay,jitter):
        self.seed=seed;self.delay=delay;self.x=np.r_[np.random.default_rng(seed).normal(0,jitter,14),np.zeros(14)]
        self.target=self.x[:14].copy();self.filtered=self.target.copy();self.last_true_command=self.target.copy()
        return self.x.copy()
    def observed_control_state(self):return np.r_[self.target,self.filtered,float(self.delay)]
    def snapshot(self):return dict(seed=self.seed,x=self.x.copy(),calls=self.calls)
    def reference(self,n,start_frame=0):return np.zeros((n,14))
    def reward(self,action,prev):return -float(np.sum(self.x**2))
    def step(self,a):
        self.calls+=1
        if self.calls==self.fail_at:raise RuntimeError('synthetic infrastructure failure')
        self.target=np.clip(self.target+a,-1,1);self.filtered=.5*self.target+.5*self.filtered
        self.last_true_command=self.filtered.copy();self.x[:14]=self.filtered
        return self.x.copy()

class ExtraDataTests(unittest.TestCase):
    def test_exact_cap_and_unused_remainder(self):
        for cap,expected in [(0,0),(40,0),(63,0),(64,64),(191,191),(192,192),(232,192),(300,300),(384,384)]:
            env=CountingEnv();budget=Budget()
            eps,manifest=collect_extra_training(env,'tracking',200001,cap,budget)
            self.assertEqual(env.calls,expected)
            self.assertEqual(budget.as_dict()['total_control_steps'],expected)
            self.assertEqual(manifest['unused_control_steps'],cap-expected)
            self.assertLessEqual(env.calls,cap)
            self.assertTrue(all(e['split']=='train' for e in eps))
            self.assertEqual([e['id'] for e in eps],[f'extra_train:{i}' for i in range(len(eps))])
    def test_no_heldout_and_disjoint_streams(self):
        main=collect(CountingEnv(),'tracking',200001,Budget(),train_eps=3,val_eps=3,test_eps=3,length=64)
        extra,manifest=collect_extra_training(CountingEnv(),'tracking',200001,384,Budget())
        self.assertFalse({e['id'] for e in main['episodes']}&{e['id'] for e in extra})
        self.assertFalse({e['episode_seed'] for e in main['episodes']}&set(manifest['episode_seeds']))
        original_test=list(main['starts']['development_test'])
        main['episodes'].extend(extra)
        seq=sequences(main,'train',horizon=8)
        self.assertTrue(len(seq['x'])>0)
        self.assertEqual(main['starts']['development_test'],original_test)
        for e in extra:
            np.testing.assert_array_equal(e['prev'][1:],e['a'][:-1])
            np.testing.assert_array_equal(e['fprev'][1:],e['c'][:-1,14:28])
            self.assertIn(32,e['snapshots'])
    def test_deterministic_and_contact(self):
        e1,m1=collect_extra_training(CountingEnv(),'contact',201001,300,Budget())
        e2,m2=collect_extra_training(CountingEnv(),'contact',201001,300,Budget())
        self.assertEqual(m1['episode_seeds'],m2['episode_seeds'])
        for a,b in zip(e1,e2):np.testing.assert_array_equal(a['x'],b['x'])
        self.assertEqual(m1['horizon'],32)
    def test_authorization_and_validation_before_queries(self):
        env=CountingEnv()
        for task,seed in [('tracking',200101),('contact',201101),('tracking',0)]:
            with self.assertRaises(PermissionError):collect_extra_training(env,task,seed,300,Budget())
        with self.assertRaises(ValueError):collect_extra_training(env,'vision',202001,300,Budget())
        with self.assertRaises(ValueError):collect_extra_training(env,'tracking',200001,-1,Budget())
        self.assertEqual(env.calls,0)
    def test_failed_calls_are_accounted(self):
        env=CountingEnv(fail_at=17);budget=Budget()
        with self.assertRaisesRegex(RuntimeError,'synthetic infrastructure failure'):
            collect_extra_training(env,'tracking',200001,192,budget)
        self.assertEqual(env.calls,17)
        self.assertEqual(budget.steps['collection_extra_train'],16)
        self.assertEqual(budget.steps['collection_extra_train_failed'],1)
        self.assertEqual(budget.as_dict()['total_control_steps'],17)
    def test_collector_arguments_are_train_only(self):
        with patch('eagm2.extra_data.collect',wraps=collect) as spy:
            collect_extra_training(CountingEnv(),'tracking',200001,300,Budget())
            for call in spy.call_args_list:
                self.assertEqual(call.kwargs['val_eps'],0);self.assertEqual(call.kwargs['test_eps'],0)

if __name__=='__main__':unittest.main(verbosity=2)
