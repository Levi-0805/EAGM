import io
import unittest
import torch
from models.eagm import EAGMBase
from eagm2.legacy_adapter import LegacyWorldAdapter


def stats():
    return dict(xmean=torch.linspace(-.2,.2,28),xstd=torch.linspace(.1,1.,28),
                ymean=torch.linspace(-.01,.01,28),ystd=torch.linspace(.01,.1,28),
                umean=torch.linspace(-.2,.2,14),ustd=torch.linspace(.01,.3,14),
                lower=torch.full((14,),-2.),upper=torch.full((14,),2.),alpha=torch.tensor(.5))

class LegacyAdapterTests(unittest.TestCase):
    def test_exact_forward_input_and_parameter_gradients(self):
        torch.manual_seed(193)
        original=EAGMBase(stats()).double();adapter=LegacyWorldAdapter(stats()).double()
        adapter.legacy.load_state_dict(original.state_dict())
        # Nonzero residual weights exercise every legacy path, not just zero init.
        with torch.no_grad():
            original.delta[-1].weight.normal_(0,.02)
        adapter.legacy.load_state_dict(original.state_dict())
        x=torch.randn(4,28,dtype=torch.float64,requires_grad=True)
        c=torch.randn(4,29,dtype=torch.float64,requires_grad=True)
        a=torch.randn(4,14,dtype=torch.float64,requires_grad=True)*.02
        yo,uo=original(x,c,a)
        ya,ua,hidden=adapter.step(x,c,a,torch.randn(4,14,dtype=torch.float64))
        self.assertIsNone(hidden)
        torch.testing.assert_close(yo,ya,rtol=0,atol=0)
        torch.testing.assert_close(uo,ua,rtol=0,atol=0)
        go=torch.autograd.grad(yo.square().sum()+uo.square().sum(),(x,c,a)+tuple(original.parameters()),retain_graph=True)
        ga=torch.autograd.grad(ya.square().sum()+ua.square().sum(),(x,c,a)+tuple(adapter.parameters()))
        for g1,g2 in zip(go,ga):torch.testing.assert_close(g1,g2,rtol=0,atol=0)
        self.assertEqual(adapter.parameter_counts()['total'],sum(p.numel() for p in original.parameters()))

    def test_buffer_forwarding_and_checkpoint(self):
        adapter=LegacyWorldAdapter(stats()).to(dtype=torch.float64)
        for key in adapter._forwarded_stats:
            if hasattr(adapter.legacy,key):
                self.assertIs(getattr(adapter,key),getattr(adapter.legacy,key))
                self.assertEqual(getattr(adapter,key).dtype,torch.float64)
        out=io.BytesIO();torch.save(adapter.state_dict(),out);out.seek(0)
        other=LegacyWorldAdapter(stats()).double();other.load_state_dict(torch.load(out,weights_only=True))
        for k,v in adapter.state_dict().items():torch.testing.assert_close(v,other.state_dict()[k])
        self.assertEqual(adapter.memory,'none');self.assertEqual(adapter.prior,'legacy')
        with self.assertRaises(ValueError):LegacyWorldAdapter({**stats(),'xmean':torch.zeros(32)})
        with self.assertRaises(TypeError):adapter.step(torch.zeros(28),torch.zeros(29),torch.zeros(14),privileged_state=torch.zeros(74))

if __name__=='__main__':unittest.main(verbosity=2)
