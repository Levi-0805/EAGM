"""Independent integration checks; synthetic controller only, no performance claim."""
import unittest
import numpy as np
import torch
from eagm2.data import collect, sequences, stats
from eagm2.runtime import Budget, check_seed, stream_seed
from eagm2.models import ExecutionWorldModel
from eagm2.rollout import imagined
from eagm2.gradients import finite_difference

class SmallPublicEnv:
    control_dt=1/30
    alpha=.5
    lower=np.full(14,-1.)
    upper=np.full(14,1.)
    def reset(self,seed,delay,jitter):
        rng=np.random.default_rng(seed)
        self.x=np.r_[rng.normal(0,jitter,14),np.zeros(14)]
        self.target=self.x[:14].copy();self.filtered=self.target.copy()
        self.queue=[self.filtered.copy(),self.filtered.copy()];self.delay=delay
        self.last_true_command=self.target.copy()
        return self.x.copy()
    def observed_control_state(self):
        return np.r_[self.target,self.filtered,float(self.delay)]
    def snapshot(self):
        return dict(x=self.x.copy(),c=self.observed_control_state().copy(),queue=np.array(self.queue))
    def step(self,a):
        self.target=np.clip(self.target+np.clip(a,-.03,.03),self.lower,self.upper)
        new=.5*self.target+.5*self.filtered
        self.last_true_command=(new if self.delay==0 else self.queue[2-self.delay]).copy()
        self.queue=[self.queue[1],new.copy()];self.filtered=new.copy()
        dq=.08*(self.last_true_command-self.x[:14])
        self.x=np.r_[self.x[:14]+dq,dq/self.control_dt]
        return self.x.copy()

class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.budget=Budget()
        cls.data=collect(SmallPublicEnv(),'tracking',200001,cls.budget,
                         train_eps=3,val_eps=3,test_eps=3,length=64,warmup=32,history=32)
    def test_confirmatory_and_historical_roots_are_rejected(self):
        for task,good,bad in [('tracking',200001,200101),('contact',201001,201101),('vision',202001,202101)]:
            check_seed(task,good)
            with self.assertRaises(PermissionError):check_seed(task,bad)
            with self.assertRaises(PermissionError):check_seed(task,0)
            with self.assertRaises(PermissionError):check_seed(task,good+100)
        with self.assertRaises(PermissionError):check_seed('tracking',201001)
    def test_episodes_and_rng_streams_are_disjoint(self):
        episodes=self.data['episodes']
        self.assertEqual(len({e['id'] for e in episodes}),len(episodes))
        self.assertEqual(len({e['episode_seed'] for e in episodes}),len(episodes))
        for split,starts in self.data['starts'].items():
            self.assertEqual(len(starts),3)
            self.assertTrue(all(s['episode_id'].startswith(split+':') for s in starts))
        self.assertNotEqual(stream_seed(200001,'train'),stream_seed(200001,'development_test'))
    def test_history_and_filter_cache_have_correct_pre_step_time(self):
        for e in self.data['episodes']:
            np.testing.assert_array_equal(e['prev'][1:],e['a'][:-1])
            np.testing.assert_array_equal(e['fprev'][1:],e['c'][:-1,14:28])
            target=np.clip(e['c'][:-1,:14]+e['a'][:-1],-1,1)
            np.testing.assert_allclose(e['c'][1:,:14],target)
            np.testing.assert_allclose(e['c'][1:,14:28],.5*target+.5*e['c'][:-1,14:28])
        seq=sequences(self.data,'train',history=32,horizon=3)
        torch.testing.assert_close(seq['xhist'][:,-1],seq['x'][:,0])
        torch.testing.assert_close(seq['chist'][:,-1],seq['c'][:,0])
        torch.testing.assert_close(seq['ahist'][:,-1],seq['prev'][:,0])
    def test_start_and_training_history_contract_agrees(self):
        seq=sequences(self.data,'train',history=32,horizon=3)
        for split,starts in self.data['starts'].items():
            for s in starts:
                self.assertEqual(len(s['xhist']),seq['xhist'].shape[1])
                np.testing.assert_array_equal(s['xhist'][-1],s['x'])
                np.testing.assert_array_equal(s['chist'][-1],s['c'])
                np.testing.assert_array_equal(s['ahist'][-1],s['prev'])
    def test_normalization_ignores_development_test(self):
        import copy
        reference=stats(self.data)
        altered=copy.deepcopy(self.data)
        for e in altered['episodes']:
            if e['split']!='train':
                for k in ['x','y','u']:e[k]+=1e5
        actual=stats(altered)
        for k in ['xmean','xstd','ymean','ystd','umean','ustd']:
            np.testing.assert_array_equal(reference[k],actual[k])
    def test_collection_budget_matches_real_step_count(self):
        self.assertEqual(self.budget.as_dict()['total_control_steps'],9*96)
    def test_imagined_closed_loop_autograd_matches_finite_difference(self):
        statistic=stats(self.data)
        for memory in ('none','recurrent','window'):
            torch.manual_seed(13)
            model=ExecutionWorldModel(statistic,width=12,memory_dim=4,memory=memory,prior='delay').double()
            start=self.data['starts']['validation'][1]
            theta=torch.full((28,),.01,dtype=torch.float64,requires_grad=True)
            objective=imagined(model,theta,start,horizon=3)
            autograd=torch.autograd.grad(objective,theta)[0].detach().numpy()
            def fn(v):
                with torch.no_grad():return float(imagined(model,torch.tensor(v,dtype=torch.float64),start,3))
            fd=finite_difference(fn,theta.detach().numpy(),epsilon=1e-5,scales=(1.,))
            np.testing.assert_allclose(autograd,fd['gradient'],rtol=2e-5,atol=2e-9,err_msg=memory)
if __name__=='__main__':unittest.main()
