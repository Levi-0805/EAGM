"""Synthetic interface/split tests, not policy-performance evidence."""
import unittest,tempfile,copy
from pathlib import Path
import numpy as np,torch
from eagm2.visual_training import prepare_visual
from eagm2.contact import HOME

class VisualTrainingTests(unittest.TestCase):
    def test_train_only_normalization_and_encoded_history(self):
        torch.set_num_threads(1);rng=np.random.default_rng(123);eps=[];starts={}
        for split in ['train','validation','development_test']:
            x=np.tile(np.r_[HOME,np.zeros(14)],(4,1))+rng.normal(0,.001,(4,28))
            ep=dict(id=split+':0',split=split,x=x,y=x+.0001,c=np.tile(np.r_[HOME,HOME,0.],(4,1)),a=np.zeros((4,14)),prev=np.zeros((4,14)),r=np.zeros(4),rgb=rng.integers(0,256,(4,64,64,3),dtype=np.uint8),next_rgb=rng.integers(0,256,(4,64,64,3),dtype=np.uint8))
            eps.append(ep);starts[split]=[dict(episode_id=ep['id'],frame=1,x=x[1],xhist=x[:2],snapshot={'do_not_read':1})]
        data=dict(task='vision',episodes=eps,starts=starts)
        altered=copy.deepcopy(data);altered['episodes'][2]['y']+=.2
        with tempfile.TemporaryDirectory() as d1,tempfile.TemporaryDirectory() as d2:
            enc,p,r,report=prepare_visual(data,Path(d1),202001,'cpu',smoke=True)
            enc2,p2,r2,_=prepare_visual(altered,Path(d2),202001,'cpu',smoke=True)
            for k,v in p.state_dict().items():torch.testing.assert_close(v,p2.state_dict()[k],rtol=0,atol=0)
            for k,v in r.state_dict().items():torch.testing.assert_close(v,r2.state_dict()[k],rtol=0,atol=0)
            self.assertEqual(enc['episodes'][0]['x'].shape,(4,32))
            np.testing.assert_array_equal(enc['starts']['train'][0]['xhist'],enc['episodes'][0]['x'][:2])
            self.assertEqual(enc['starts']['train'][0]['snapshot'],{'do_not_read':1})
            self.assertEqual(data['episodes'][0]['x'].shape,(4,28))
            self.assertEqual(report['simulator_steps_added_by_prepare_visual'],0)
            self.assertTrue((Path(d1)/'checkpoints/visual_policy.pt').exists())
            self.assertTrue((Path(d1)/'checkpoints/visual_reward.pt').exists())
            z=torch.randn(32,requires_grad=True);rew=r(z,torch.zeros(14),torch.zeros(14));g=torch.autograd.grad(rew,z)[0]
            self.assertTrue(torch.isfinite(g).all())
if __name__=='__main__':unittest.main()
