import unittest
import numpy as np
from eagm2.data import collect
from eagm2.runtime import Budget
from tests.eagm2.test_extra_data import CountingEnv

class VisualEnv(CountingEnv):
    def render_rgb(self):return np.full((2,2,3),self.calls%255,dtype=np.uint8)

class PhaseStartTests(unittest.TestCase):
    def test_phase_delay_crossing_and_causal_history(self):
        for task in ('contact','vision'):
            data=collect(VisualEnv(),task,201001,Budget(),train_eps=9,val_eps=3,test_eps=3,length=128)
            starts=data['starts']['train'];episodes={e['id']:e for e in data['episodes']}
            self.assertEqual({(s['frame'],s['delay']) for s in starts},{(f,d) for f in (32,64,96) for d in range(3)})
            for s in starts:
                e=episodes[s['episode_id']];t=s['frame']
                for hist,key in [('xhist','x'),('chist','c'),('ahist','prev')]:
                    np.testing.assert_array_equal(s[hist],e[key][t-31:t+1]);self.assertEqual(len(s[hist]),32)
                np.testing.assert_array_equal(s['snapshot']['x'],s['x'])
                np.testing.assert_array_equal(s['refs'],e['refs'][t:])
            for split in ('validation','development_test'):
                self.assertEqual({s['frame'] for s in data['starts'][split]},{32})
    def test_short_episode_only_legal_snapshots(self):
        data=collect(CountingEnv(),'contact',201001,Budget(),train_eps=3,val_eps=0,test_eps=0,length=32)
        self.assertEqual([s['frame'] for s in data['starts']['train']],[32]*3)
    def test_failed_episode_records_completed_steps(self):
        budget=Budget();env=CountingEnv(fail_at=17)
        with self.assertRaisesRegex(RuntimeError,'synthetic infrastructure failure'):
            collect(env,'contact',201001,budget,train_eps=3,val_eps=0,test_eps=0)
        self.assertEqual(budget.steps['collection_train'],16)
        self.assertGreaterEqual(budget.seconds['collection_train'],0.)

    def test_tracking_protocol_preserved(self):
        data=collect(CountingEnv(),'tracking',200001,Budget(),train_eps=9,val_eps=0,test_eps=0,length=128)
        self.assertEqual({s['frame'] for s in data['starts']['train']},{32})

if __name__=='__main__':unittest.main(verbosity=2)
