import json
import unittest
import torch
from eagm2.training import make_model,train_critic,train_model


def data(dim=28):
 g=torch.Generator().manual_seed(42);n,h=8,3
 x=torch.randn(n,h,dim,generator=g)*.01
 c=torch.zeros(n,h,29)
 return dict(x=x,y=x+.001,c=c,a=torch.randn(n,h,14,generator=g)*.001,
             refs=torch.zeros(n,h,14),r=torch.randn(n,h,generator=g)*.1,
             u=torch.zeros(n,h,14),fprev=torch.zeros(n,h,14))


def test_named_init_shares_common_shapes_and_preserves_prior():
 a=make_model({},'delay_base',200001,'cpu');b=make_model({},'combined_base',200001,'cpu')
 ap=dict(a.named_parameters());bp=dict(b.named_parameters())
 common=[n for n in ap if n in bp and ap[n].shape==bp[n].shape]
 assert 'dynamics.0.weight' in common
 for n in common:assert torch.equal(ap[n],bp[n]),n
 assert not a.command_net[-1].weight.any()
 json.dumps(a.initialization_manifest)


def test_fqe_checkpoint_context_and_heldout_isolation():
 d=data();v=data();v['r']=v['r']+1000
 q,r=train_critic(d,200001,'cpu',updates=2,validation=v)
 q2,r2=train_critic(d,200001,'cpu',updates=2)
 assert r['context_dimension']==71 and r['validation']['used_for_selection'] is False
 for k,value in q.state_dict().items():assert torch.equal(value,q2.state_dict()[k]),k
 assert all(not p.requires_grad for p in q.parameters())
 assert 'target_state' in q.training_checkpoint
 json.dumps(r)
 for method in ['paml_adaptation','vagram_adaptation']:
  m=make_model({},method,200001,'cpu')
  out=train_model(m,method,d,[],200001,updates=1,batch_size=4,horizon=2,critic=q)
  assert out['log'][0]['objective_loss']>=0


class VisionActor(torch.nn.Module):
 def action_from_latent(self,z,c,theta):
  return .03*torch.tanh(z[...,:14]+theta[:14]+.1*theta[14:28])


def test_vision_32_adapter_context():
 d=data(32);vision=VisionActor()
 q,r=train_critic(d,202001,'cpu',updates=2,vision_policy=vision)
 assert r['context_dimension']==75 and r['actor_parameter_dimension']==32
 for method in ['paml_adaptation','vagram_adaptation']:
  m=make_model({},method,202001,'cpu',obs_dim=32)
  train_model(m,method,d,[],202001,updates=1,batch_size=4,horizon=2,critic=q,vision_policy=vision)


class CriticIntegrationTests(unittest.TestCase):
 test_named_init_shares_common_shapes_and_preserves_prior=staticmethod(test_named_init_shares_common_shapes_and_preserves_prior)
 test_fqe_checkpoint_context_and_heldout_isolation=staticmethod(test_fqe_checkpoint_context_and_heldout_isolation)
 test_vision_32_adapter_context=staticmethod(test_vision_32_adapter_context)
