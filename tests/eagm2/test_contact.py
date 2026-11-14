import unittest, numpy as np, torch
from eagm2.contact import ContactEnv, TorchContactReward

class ContactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.env=ContactEnv()
    @classmethod
    def tearDownClass(cls): cls.env.close()
    def test_exact_fk_and_reward(self):
        e=self.env;e.reset(seed=12,jitter=.01);fk=TorchContactReward(e)
        rng=np.random.default_rng(40)
        for _ in range(12):
            x=np.r_[e.home+rng.normal(0,.05,14),rng.normal(0,.02,14)]
            local,dot=fk.geometry(torch.tensor(x[:14]));g=e.geometry(x[:14])
            np.testing.assert_allclose(local.detach().numpy(),e.fixture_rotation.T@(g['tip']-e.mouth),atol=1e-12)
        x=e.observe();a=np.ones(14)*.001;prev=np.zeros(14)
        self.assertAlmostEqual(float(fk(torch.tensor(x),torch.tensor(a),torch.tensor(prev))),e.reward(a,prev),places=12)
    def test_fk_autograd_finite_difference(self):
        e=self.env;fk=TorchContactReward(e);q=torch.tensor(e.home+.03,requires_grad=True)
        J=torch.autograd.functional.jacobian(lambda z:fk.geometry(z)[0],q).detach().numpy()
        h=1e-5;fd=[]
        for i in range(14):
            dq=np.eye(14)[i]*h
            pp=e.geometry(q.detach().numpy()+dq)['tip'];pm=e.geometry(q.detach().numpy()-dq)['tip']
            fd.append(e.fixture_rotation.T@(pp-pm)/(2*h))
        np.testing.assert_allclose(J,np.asarray(fd).T,atol=1e-9,rtol=1e-7)
    def test_snapshot_replay_and_hash(self):
        e=self.env;e.reset(seed=14,delay=2)
        for _ in range(20):e.step(np.ones(14)*.001)
        s=e.snapshot();a=np.arange(14)*1e-4;x1=e.step(a);m1=e.metrics()
        e.restore(s);x2=e.step(a);m2=e.metrics()
        np.testing.assert_array_equal(x1,x2);self.assertEqual(m1,m2)
        bad=dict(s,model_sha256='wrong')
        with self.assertRaises(ValueError):e.restore(bad)
    def test_scripted_contact_success(self):
        e=self.env;e.reset(seed=0,jitter=0)
        for ref in e.reference(int(np.floor(10/e.control_dt))):
            x=e.observe();a=np.clip(.55*(ref-e.qtarget)+.1*(ref-x[:14]),-.03,.03);e.step(a)
        m=e.metrics();self.assertTrue(m['success'],str(m));self.assertGreater(m['contact_fraction'],.1);self.assertLess(m['max_penetration'],.001)
        self.assertGreaterEqual(m['depth'],.02);self.assertLess(m['position_error'],.002)

if __name__=='__main__':unittest.main()
