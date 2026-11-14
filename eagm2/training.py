"""Common-data experimental trainer with explicit objective adaptations."""
import copy,time
import numpy as np
import torch
from torch import nn
from eagm2.runtime import measured,seed_all,stream_seed
from eagm2.models import ExecutionWorldModel,capacity_matched_width
from eagm2.prior import observable_step
from eagm2.rollout import imagined,tensor
from eagm2.gradients import calibration_loss
from eagm2.baselines import QCritic,paml_loss,vagram_loss
from training.rollout import actor
from training.reward import reward_torch

METHODS={
 'legacy_base':dict(memory='none',prior='legacy',loss='none'),
 'legacy_dir':dict(memory='none',prior='legacy',loss='legacy'),
 'ff_base':dict(memory='none',prior='original',loss='none'),
 'ff_dir':dict(memory='none',prior='original',loss='legacy'),
 'delay_base':dict(memory='none',prior='delay',loss='none'),
 'delay_dir':dict(memory='none',prior='delay',loss='legacy'),
 'recurrent_base':dict(memory='recurrent',prior='original',loss='none'),
 'recurrent_dir':dict(memory='recurrent',prior='original',loss='legacy'),
 'combined_base':dict(memory='recurrent',prior='delay',loss='none'),
 'combined_dir':dict(memory='recurrent',prior='delay',loss='legacy'),
 'capacity_mlp':dict(memory='none',prior='delay',loss='legacy',capacity=True),
 'capacity_gru':dict(memory='window',prior='delay',loss='legacy',capacity=True),
 'free_command':dict(memory='none',prior='free',loss='legacy'),
 'direction_only':dict(memory='none',prior='delay',loss='direction'),
 'norm_only':dict(memory='none',prior='delay',loss='norm'),
 'joint_calibration':dict(memory='none',prior='delay',loss='joint'),
 'paml_adaptation':dict(memory='none',prior='delay',loss='paml'),
 'vagram_adaptation':dict(memory='none',prior='delay',loss='vagram'),
}
def make_model(stats,name,seed,device,obs_dim=28):
 cfg=METHODS[name];seed_all(stream_seed(seed,'model_init'))
 if cfg['prior']=='legacy':
  if obs_dim!=28:raise ValueError('Frozen legacy model only supports physical state; not a visual baseline')
  from eagm2.legacy_adapter import LegacyWorldAdapter
  model=LegacyWorldAdapter(stats);model.initialization_manifest={'protocol':'frozen legacy EAGMBase constructor','seed':stream_seed(seed,'model_init')}
  return model.to(device)
 width=160
 if cfg.get('capacity'):
  ref=ExecutionWorldModel(stats,obs_dim=obs_dim,width=160,memory='recurrent',prior='delay')
  width,_=capacity_matched_width(stats,ref.parameter_counts()['trainable'],obs_dim=obs_dim,memory=cfg['memory'])
 model=ExecutionWorldModel(stats,obs_dim=obs_dim,width=width,memory=cfg['memory'],prior=cfg['prior'])
 # Named, shape-keyed streams make common modules independent of constructor order.
 manifest={}
 with torch.no_grad():
  for name,p in model.named_parameters():
   module=model.get_submodule(name.rsplit('.',1)[0])
   bound=(module.hidden_size if isinstance(module,nn.GRUCell) else (p.shape[-1] if p.ndim>1 else p.numel()))**-.5
   key=f"parameter/{name}/{tuple(p.shape)}";param_seed=stream_seed(seed,key)
   zero=cfg['prior']!='free' and name in ('command_net.2.weight','command_net.2.bias')
   if zero:p.zero_()
   else:p.uniform_(-bound,bound,generator=torch.Generator().manual_seed(param_seed))
   manifest[name]=dict(shape=list(p.shape),seed=param_seed,initialization='zero residual' if zero else 'uniform',bound=bound)
 model.initialization_manifest=dict(protocol='named parameter and shape keyed streams v1',root_seed=seed,parameters=manifest)
 return model.to(device)

class ContextQCritic(nn.Module):
 """Observable x,c,reference context; not a sufficient hidden-state belief."""
 def __init__(self,context_mean,context_std,reward_scale):
  super().__init__()
  self.register_buffer('context_mean',context_mean.detach().clone())
  self.register_buffer('context_std',context_std.detach().clone())
  self.register_buffer('reward_scale',reward_scale.detach().clone())
  self.network=QCritic(context_mean.numel(),14)
 def forward(self,context,action):
  return self.network((context-self.context_mean)/self.context_std,action/.03)*self.reward_scale

class NextContextCritic(nn.Module):
 def __init__(self,critic,c,ref):
  super().__init__();self.critic=critic;self.c=c;self.ref=ref
 def forward(self,state,action):
  return self.critic(torch.cat((state,self.c,self.ref),-1),action)

def _fixed_actor(theta,x,c,ref,vision_policy=None):
 return actor(theta,x,c[...,:14],ref) if vision_policy is None else vision_policy.action_from_latent(x,c,theta)

def train_critic(train,seed,device,updates=500,vision_policy=None,validation=None):
 seed_all(stream_seed(seed,'critic'))
 if train['x'].shape[1]<2:raise ValueError('critic requires aligned next c/reference')
 context=torch.cat((train['x'][:,0],train['c'][:,0],train['refs'][:,0]),-1)
 scale=train['r'][:,0].square().mean().sqrt().clamp_min(1e-3).detach()
 critic=ContextQCritic(context.mean(0),context.std(0,unbiased=False).clamp_min(.01),scale).to(device)
 target=copy.deepcopy(critic).eval()
 for p in target.parameters():p.requires_grad_(False)
 opt=torch.optim.Adam(critic.parameters(),lr=.001)
 gen=torch.Generator().manual_seed(stream_seed(seed,'critic_batch'));logs=[]
 theta=torch.zeros(28 if vision_policy is None else 32,device=device)
 gamma=.99;tau=.01
 def td(data,idx):
  x,c,ref=data['x'][idx,0],data['c'][idx,0],data['refs'][idx,0]
  nx,nc,nref=data['y'][idx,0],data['c'][idx,1],data['refs'][idx,1]
  with torch.no_grad():
   na=_fixed_actor(theta,nx,nc,nref,vision_policy)
   boot=target(torch.cat((nx,nc,nref),-1),na).reshape(-1)
   # Data are continuing trajectories; no invented terminal flags.
   expected=data['r'][idx,0].reshape(-1)+gamma*boot
  pred=critic(torch.cat((x,c,ref),-1),data['a'][idx,0]).reshape(-1)
  return pred,expected
 for it in range(updates):
  b=torch.randint(len(context),(128,),generator=gen).to(device)
  pred,expected=td(train,b);loss=((pred-expected)/scale).square().mean()
  if not torch.isfinite(loss):raise FloatingPointError('nonfinite FQE loss')
  opt.zero_grad();loss.backward();opt.step()
  with torch.no_grad():
   for tp,p in zip(target.parameters(),critic.parameters()):tp.lerp_(p,tau)
  if it%100==0 or it==updates-1:logs.append(dict(update=it+1,normalized_td_mse=loss.item()))
 with torch.no_grad():
  actual=_fixed_actor(theta,train['x'][:,0],train['c'][:,0],train['refs'][:,0],vision_policy)
  coverage=(actual-train['a'][:,0]).square().mean().sqrt().item()
  val=None
  if validation is not None:
   pred,expected=td(validation,torch.arange(len(validation['x']),device=device))
   val=dict(td_rmse=(pred-expected).square().mean().sqrt().item(),transitions=len(pred),used_for_selection=False)
 critic.training_checkpoint=dict(optimizer_state=opt.state_dict(),target_state=target.state_dict(),torch_rng=torch.get_rng_state(),batch_generator_state=gen.get_state(),cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])
 for p in critic.parameters():p.requires_grad_(False)
 report=dict(training_updates=updates,target='continuing discounted fixed theta0 FQE TD; frozen Polyak target',gamma=gamma,target_tau=tau,context=['x','observable_c','reference'],context_dimension=context.shape[-1],reward_scale=scale.item(),actor_parameter_dimension=theta.numel(),behavior_to_theta0_action_rmse=coverage,validation=val,log=logs,parameter_count=sum(p.numel() for p in critic.parameters()),checkpoint_attribute='training_checkpoint',limitations=['Off-policy behavior coverage is not guaranteed; action RMSE is only a descriptive diagnostic.','Observable context omits hidden Controller B state; FQE has state aliasing and is not an exact POMDP value.','Continuing discounted proxy, not exact finite horizon return; heldout TD residual is not true value error.','Actor fixed at theta0; does not evaluate all policy parameter positions.'],extra_simulator_queries=0)
 return critic,report
def train_model(model,name,train,labels,seed,*,updates=1000,batch_size=64,horizon=8,critic=None,reward_fn=None,vision_policy=None):
 cfg=METHODS[name];dev=next(model.parameters()).device
 opt=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=.001)
 gen=torch.Generator().manual_seed(stream_seed(seed,'paired_batch'));logs=[]
 with measured(dev) as resource:
  for it in range(updates):
   idx=torch.randint(train['x'].shape[0],(batch_size,),generator=gen).to(dev)
   b={k:v[idx] for k,v in train.items()}
   h=model.burnin(b['xhist'],b['chist'],b['ahist'],detach=False) if model.memory!='none' else None
   y,u,hn=model.step(b['x'][:,0],b['c'][:,0],b['a'][:,0],b['fprev'][:,0],h)
   dyn=(((y-b['y'][:,0])/model.ystd)**2).mean();cmd=(((u-b['u'][:,0])/model.ustd)**2).mean();loss=dyn+cmd
   # Open-loop multi-step fitting, same scheduled loss for every architecture.
   if it%10==0 and cfg['prior']!='legacy':
    xx=b['x'][:,0];cc=b['c'][:,0];fp=b['fprev'][:,0];hh=h;multi=loss*0
    for k in range(horizon):
     xx,_,hh=model.step(xx,cc,b['a'][:,k],fp,hh)
     multi=multi+(((xx-b['y'][:,k])/model.ystd)**2).mean()/horizon
     cc,fp=observable_step(cc,b['a'][:,k],model.lower,model.upper,float(model.alpha))
    loss=loss+.1*multi
   extra=loss*0
   if cfg['loss'] in ('legacy','direction','norm','joint') and it%10==0:
    lab=labels[(it//10)%len(labels)];th=tensor(lab['theta'],model).clone().requires_grad_(True)
    J=imagined(model,th,lab['start'],horizon,reward_fn=reward_fn,vision_policy=vision_policy)
    g=torch.autograd.grad(J,th,create_graph=True)[0]
    extra=calibration_loss(g,tensor(lab['gradient'],model),tensor(lab['mask'],model),mode=cfg['loss'])
    loss=loss+.1*extra
   elif cfg['loss'] in ('paml','vagram') and (cfg['loss']=='vagram' or it%10==0):
    if critic is None:raise ValueError('adaptation requires a fitted contextual critic')
    th=torch.zeros(28 if vision_policy is None else 32,device=dev,dtype=y.dtype,requires_grad=cfg['loss']=='paml')
    next_c=b['c'][:,1];next_ref=b['refs'][:,1]
    actor_fn=lambda state:_fixed_actor(th,state,next_c,next_ref,vision_policy)
    contextual=NextContextCritic(critic,next_c,next_ref)
    if cfg['loss']=='paml':extra=paml_loss(contextual,actor_fn,b['y'][:,0],y,[th])
    else:extra=vagram_loss(lambda state:contextual(state,actor_fn(state)),b['y'][:,0],y,state_scale=model.ystd)
    loss=loss+.1*extra
   if not torch.isfinite(loss):raise FloatingPointError(f'{name} nonfinite loss at {it}')
   opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5.);opt.step()
   if it%100==0 or it==updates-1:
    logs.append(dict(update=it+1,loss=loss.item(),dynamics_loss=dyn.item(),command_loss=cmd.item(),objective_loss=extra.item()))
    print(name,it+1,round(loss.item(),6),flush=True)
 return dict(log=logs,resource=resource,optimizer_state=opt.state_dict(),torch_rng=torch.get_rng_state(),batch_generator_state=gen.get_state(),batch_seed=stream_seed(seed,'paired_batch'),updates=updates)
def optimize(model,starts,horizon,updates=50,reward_fn=None,vision_policy=None):
 dev=next(model.parameters()).device;dim=28 if vision_policy is None else 32
 theta=torch.zeros(dim,device=dev,requires_grad=True);opt=torch.optim.Adam([theta],lr=.03);curves=[];traces={0:theta.detach().cpu().tolist()}
 with measured(dev) as resource:
  for it in range(updates):
   J=torch.stack([imagined(model,theta,s,horizon,reward_fn=reward_fn,vision_policy=vision_policy) for s in starts]).mean()
   opt.zero_grad();(-J).backward();opt.step()
   with torch.no_grad():theta.clamp_(-1,1)
   curves.append(float(J.detach()))
   if it+1 in (1,10,25,50,updates):traces[it+1]=theta.detach().cpu().tolist()
 return theta.detach().cpu().numpy(),dict(curve=curves,theta_traces=traces,resource=resource,optimizer_state=opt.state_dict(),torch_rng=torch.get_rng_state())
