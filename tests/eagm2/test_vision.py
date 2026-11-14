import unittest,torch
from eagm2.vision import VisionPolicy,DeterministicPool2d

class VisionTests(unittest.TestCase):
    def setUp(self):torch.manual_seed(1);torch.set_num_threads(1)
    def test_pool_matches_adaptive_forward_and_gradient(self):
        for shape in [(8,8),(7,9),(1,3)]:
            x=torch.randn(2,3,*shape,dtype=torch.float64,requires_grad=True)
            a=DeterministicPool2d()(x)
            b=torch.nn.functional.adaptive_avg_pool2d(x,(2,2))
            torch.testing.assert_close(a,b)
            v=torch.randn_like(a)
            ga=torch.autograd.grad(a,x,v,retain_graph=True)[0]
            gb=torch.autograd.grad(b,x,v)[0]
            torch.testing.assert_close(ga,gb)
    def test_actual_rgb_dependency_and_adapter(self):
        p=VisionPolicy().freeze_base();c=torch.zeros(2,29);theta=torch.zeros(32,requires_grad=True)
        im=torch.zeros(2,3,64,64);a=p(im,c,theta);b=p(torch.ones_like(im),c,theta)
        self.assertGreater(float((a-b).abs().max().detach()),1e-6)
        grad=torch.autograd.grad(a.sum(),theta)[0];self.assertGreater(float(grad.abs().max()),1e-8)
        self.assertTrue(all(not v.requires_grad for v in p.parameters()))
    def test_deployment_has_no_privileged_argument(self):
        p=VisionPolicy();im=torch.rand(1,3,64,64);c=torch.zeros(1,29)
        with self.assertRaises(TypeError):p(im,c,state=torch.zeros(1,28))
        with self.assertRaises(TypeError):p(im,c,hidden_execution_state=torch.zeros(1,74))
        # Changing irrelevant fields in the caller's dictionary cannot reach input API.
        record={'rgb':im,'c_obs':c,'q':torch.ones(28)}
        a=p(record['rgb'],record['c_obs']);record['q']*=123
        torch.testing.assert_close(a,p(record['rgb'],record['c_obs']))
    def test_rgb_only(self):
        p=VisionPolicy(record_dim=0)
        self.assertEqual(p(torch.zeros(2,64,64,3,dtype=torch.uint8)).shape,(2,14))
    def test_latent_forward_equal(self):
        p=VisionPolicy();rgb=torch.rand(2,3,64,64);c=torch.rand(2,29);th=torch.rand(32)
        torch.testing.assert_close(p(rgb,c,th),p.action_from_latent(p.encode(rgb),c,th))

if __name__=='__main__':unittest.main()
