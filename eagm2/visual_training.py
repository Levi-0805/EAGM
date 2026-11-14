"""Train-only visual BC and latent reward adaptation for exploratory runs.

A single-frame CNN is the explicit initial development adaptation. Privileged
state is used exclusively to construct training labels and held-out diagnostics;
the frozen imagined reward consumes latent observations, actions and previous
actions, never simulator state. Validation/test metrics do not select weights.
"""
from __future__ import annotations
from copy import deepcopy
import hashlib,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from eagm2.vision import VisionPolicy
from eagm2.contact import ContactEnv,TorchContactReward
from eagm2.runtime import measured,dump,stream_seed

class LatentContactReward(nn.Module):
    def __init__(self,zmean,zstd,tmean,tstd):
        super().__init__()
        for name,v in [('zmean',zmean),('zstd',zstd),('tmean',tmean),('tstd',tstd)]:self.register_buffer(name,torch.as_tensor(v,dtype=torch.float32))
        self.net=nn.Sequential(nn.Linear(len(zmean),96),nn.SiLU(),nn.Linear(96,96),nn.SiLU(),nn.Linear(96,5))
    def features(self,z):return self.net((z-self.zmean)/self.zstd)*self.tstd+self.tmean
    def forward(self,z,a,prev):
        g=self.features(z)
        return -100*g[...,:2].square().sum(-1)-20*(g[...,2]-.030).square()-.04*g[...,3].clamp_min(0)-.0005*g[...,4].clamp_min(0)-.05*(a-prev).square().mean(-1)

def _hash_data(data):
    h=hashlib.sha256()
    for e in data['episodes']:
        h.update(e['id'].encode());h.update(e['split'].encode())
        for key in ['rgb','next_rgb','x','y','c','a','r']:
            a=np.ascontiguousarray(e[key]);h.update(key.encode());h.update(str(a.shape).encode());h.update(str(a.dtype).encode());h.update(a.tobytes())
    return h.hexdigest()

def _encode(policy,images,device,batch_size=256):
    out=[]
    with torch.no_grad():
        for first in range(0,len(images),batch_size):
            rgb=torch.as_tensor(images[first:first+batch_size],device=device)
            out.append(policy.encode(rgb).cpu().numpy())
    return np.concatenate(out)

def prepare_visual(data,out,seed,device,smoke=False):
    out=Path(out);(out/'checkpoints').mkdir(exist_ok=True);(out/'training_logs').mkdir(exist_ok=True)
    if data.get('task')!='vision':raise ValueError('prepare_visual requires RGB vision dataset')
    train=[e for e in data['episodes'] if e['split']=='train']
    if not train:raise ValueError('No training episodes')
    groups={s:{e['id'] for e in data['episodes'] if e['split']==s} for s in ['train','validation','development_test']}
    if groups['train']&groups['validation'] or groups['train']&groups['development_test'] or groups['validation']&groups['development_test']:raise ValueError('Episode split collision')
    data_sha=_hash_data(data);init_seed=stream_seed(seed,'visual_initialization');batch_seed=stream_seed(seed,'visual_minibatches')
    torch.manual_seed(init_seed)
    policy=VisionPolicy().to(device);opt=torch.optim.Adam(list(policy.encoder.parameters())+list(policy.base.parameters()),lr=1e-3)
    rng=torch.Generator().manual_seed(batch_seed)
    images=torch.as_tensor(np.concatenate([e['rgb'] for e in train]),device=device)
    cs=torch.as_tensor(np.concatenate([e['c'] for e in train]),dtype=torch.float32,device=device)
    actions=torch.as_tensor(np.concatenate([e['a'] for e in train]),dtype=torch.float32,device=device)
    logs=[];updates=30 if smoke else 1000;batch=32 if smoke else 64
    with measured(device) as bc_resources:
        policy.train()
        for it in range(updates):
            b=torch.randint(len(images),(batch,),generator=rng).to(device)
            pred=policy(images[b],cs[b]);loss=((pred-actions[b])/.03).square().mean()
            opt.zero_grad(set_to_none=True);loss.backward();opt.step()
            if it%50==0 or it==updates-1:logs.append(dict(update=it+1,normalized_bc_mse=float(loss.detach())))
        policy.freeze_base()
    bc_checkpoint={'state_dict':policy.state_dict(),'optimizer':opt.state_dict(),'torch_rng':torch.get_rng_state(),'batch_generator_state':rng.get_state(),'cuda_rng':torch.cuda.get_rng_state_all() if str(device).startswith('cuda') else None,'config':dict(obs_dim=32,record_dim=29,adapter_dim=32,single_frame=True,updates=updates,seed=seed,init_seed=init_seed,batch_seed=batch_seed),'data_sha256':data_sha,'training_episode_ids':sorted(groups['train']),'selection':'fixed final update; no validation/test selection'}
    torch.save(bc_checkpoint,out/'checkpoints/visual_policy.pt');dump(out/'training_logs/visual_bc.json',dict(log=logs,resource=bc_resources))
    del images,cs,actions,opt
    encoded=dict(data);encoded['episodes']=[];encoded['starts']={s:[] for s in data['starts']}
    env=ContactEnv();exact=TorchContactReward(env).to(device).float();labels={};split_metrics={}
    with measured(device) as encoding_resources:
        for e in data['episodes']:
            ep=dict(e);ep['x']=_encode(policy,e['rgb'],device);ep['y']=_encode(policy,e['next_rgb'],device);encoded['episodes'].append(ep)
            # Geometry labels stay in this train/diagnostic scope, not the encoded world-model inputs.
            xnext=torch.as_tensor(e['y'],dtype=torch.float32,device=device)
            with torch.no_grad():
                local,dot=exact.geometry(xnext[...,:14]);target=torch.cat((local,(1-dot).unsqueeze(-1),xnext[...,14:].square().mean(-1,keepdim=True)),-1)
            labels[e['id']]=target.cpu().numpy()
    env.close();del exact
    byid={e['id']:e for e in encoded['episodes']}
    for split,starts in data['starts'].items():
        for s in starts:
            st=dict(s);ep=byid[s['episode_id']];t=s['frame'];n=len(s['xhist'])
            st['x']=ep['x'][t].copy();st['xhist']=ep['x'][t-n+1:t+1].copy()
            if len(st['xhist'])!=n:raise ValueError('History references frames before episode reset')
            encoded['starts'][split].append(st)
    tr=[e for e in encoded['episodes'] if e['split']=='train']
    Z=torch.as_tensor(np.concatenate([e['y'] for e in tr]),dtype=torch.float32,device=device)
    target=torch.as_tensor(np.concatenate([labels[e['id']] for e in tr]),dtype=torch.float32,device=device)
    zmean=Z.mean(0);zstd=Z.std(0).clamp_min(.01);tmean=target.mean(0)
    tstd=torch.maximum(target.std(0),torch.tensor([.001,.001,.002,.001,.001],device=device))
    torch.manual_seed(stream_seed(seed,'reward_initialization'))
    reward=LatentContactReward(zmean,zstd,tmean,tstd).to(device);ropt=torch.optim.Adam(reward.parameters(),lr=1e-3);rgen=torch.Generator().manual_seed(stream_seed(seed,'reward_minibatches'))
    reward_updates=50 if smoke else 1000;rlogs=[]
    with measured(device) as reward_resources:
        for it in range(reward_updates):
            b=torch.randint(len(Z),(batch,),generator=rgen).to(device)
            loss=((reward.features(Z[b])-target[b])/tstd).square().mean()
            ropt.zero_grad(set_to_none=True);loss.backward();ropt.step()
            if it%50==0 or it==reward_updates-1:rlogs.append(dict(update=it+1,normalized_feature_mse=float(loss.detach())))
        reward.eval()
        for p in reward.parameters():p.requires_grad_(False)
    torch.save(dict(state_dict=reward.state_dict(),optimizer=ropt.state_dict(),torch_rng=torch.get_rng_state(),batch_generator_state=rgen.get_state(),cuda_rng=torch.cuda.get_rng_state_all() if str(device).startswith('cuda') else None,config=dict(obs_dim=32,target_dim=5,updates=reward_updates,seed=seed),data_sha256=data_sha,training_episode_ids=sorted(groups['train']),selection='fixed final update; no validation/test selection'),out/'checkpoints/visual_reward.pt')
    dump(out/'training_logs/visual_reward.json',dict(log=rlogs,resource=reward_resources))
    with torch.no_grad():
        for split in groups:
            eps=[e for e in encoded['episodes'] if e['split']==split];zs=torch.as_tensor(np.concatenate([e['y'] for e in eps]),dtype=torch.float32,device=device)
            ytrue=np.concatenate([labels[e['id']] for e in eps]);yp=reward.features(zs).cpu().numpy()
            acts=torch.as_tensor(np.concatenate([e['a'] for e in eps]),dtype=torch.float32,device=device);prev=torch.as_tensor(np.concatenate([e['prev'] for e in eps]),dtype=torch.float32,device=device)
            rpred=reward(zs,acts,prev).cpu().numpy();rtrue=np.concatenate([e['r'] for e in eps])
            # BC metrics at pre-transition latent, using logged observable records only.
            zbefore=torch.as_tensor(np.concatenate([e['x'] for e in eps]),dtype=torch.float32,device=device);cs=torch.as_tensor(np.concatenate([e['c'] for e in eps]),dtype=torch.float32,device=device)
            pred=policy.action_from_latent(zbefore,cs)
            split_metrics[split]=dict(episodes=len(eps),frames=len(zs),bc_action_mae=float((pred-acts).abs().mean()),feature_rmse=np.sqrt(np.mean((yp-ytrue)**2,axis=0)).tolist(),reward_mae=float(np.mean(np.abs(rpred-rtrue))),reward_rmse=float(np.sqrt(np.mean((rpred-rtrue)**2))))
    torch.save(encoded,out/'visual_encoded_dataset.pt')
    report=dict(status='trained_exploratory',single_frame_deviation=True,policy='frozen CNN/base with trainable external 32D adapter; not full-network policy optimization',reward='train-only learned latent geometry/velocity head; exact physical reward used for simulator FD/evaluation',deployment_inputs=['RGB','observable target/filter/delay'],privileged_state='training/diagnostic labels and physical simulator snapshots only; never imagined reward inputs',target_features=['local_tip_x','local_tip_y','depth','one_minus_axis_dot','mean_joint_velocity_squared'],split_metrics=split_metrics,data_sha256=data_sha,training_episode_ids=sorted(groups['train']),selection='final fixed update; no held-out checkpoint selection',resources=dict(bc=bc_resources,encoding=encoding_resources,reward=reward_resources),simulator_steps_added_by_prepare_visual=0,demonstration_cost='already counted in collection_train by runner; no uncounted demonstration interaction',parameter_counts=dict(visual_policy=sum(p.numel() for p in policy.parameters()),reward=sum(p.numel() for p in reward.parameters())),normalization='train-only means/stds stored in reward state_dict')
    dump(out/'visual_preparation_manifest.json',report)
    return encoded,policy,reward,report
