"""New episode-isolated exploratory data; no historical data files are rewritten."""
from __future__ import annotations
import time
from unittest.mock import patch
import numpy as np
import torch
from eagm2.runtime import ROOT,stream_seed
from controllers.partial_unknown import ControllerBConfig,PartialUnknownArmEnv
from training.rollout import actor_raw_numpy
from training.reward import reward_numpy
CFG=ControllerBConfig(gain=1.05,extra_lag=1,rate_limit=.045,smooth_beta=.25)
def make_env(task='tracking',config=None):
    if task in ('contact','vision'):
        from eagm2.contact import ContactEnv
        return ContactEnv(config=config or CFG)
    with patch('controllers.compatible_arm_env.materialize_pilot_effective_xml',return_value=ROOT/'assets/pilot_effective_model.linux.xml'):
        return PartialUnknownArmEnv(config or CFG)
def collect(env,task,seed,budget,*,train_eps=18,val_eps=6,test_eps=30,length=128,warmup=32,history=32):
    episodes=[]
    for split,n in [('train',train_eps),('validation',val_eps),('development_test',test_eps)]:
        for ep in range(n):
            rng_seed=stream_seed(seed,f'{split}/episode/{ep}');rng=np.random.default_rng(rng_seed)
            delay=ep%3;begin=time.perf_counter();x=env.reset(seed=rng_seed,delay=delay,jitter=.01)
            total_length=length+warmup if split=='train' else max(warmup+64,96)
            if task=='tracking':
                tt=np.arange(total_length+512)*env.control_dt
                refs=x[:14][None,:]+.08*np.sin(2*np.pi*(.1+.02*(np.arange(14)%5))[None,:]*tt[:,None])
            else: refs=env.reference(total_length+512,start_frame=0)
            xs=[];cs=[];acts=[];ys=[];commands=[];rewards=[];fprevs=[];snapshots={};rgb=[];next_rgb=[]
            prev=np.zeros(14);fprev=env.observed_control_state()[14:28].copy()
            completed_steps=0
            try:
                for t in range(total_length):
                    c=env.observed_control_state().copy()
                    if t in (warmup,warmup+32,warmup+64) and t+32<=total_length:
                        snapshots[t]=env.snapshot()
                    xs.append(x.copy());cs.append(c);acts.append(prev.copy());fprevs.append(fprev.copy())
                    if task=='vision':rgb.append(env.render_rgb().copy())
                    action=np.clip(actor_raw_numpy(np.zeros(28),x,c[:14],refs[t])+rng.normal(0,.003,14),-.03,.03)
                    newx=env.step(action);completed_steps+=1
                    if task=='vision':next_rgb.append(env.render_rgb().copy())
                    reward=reward_numpy(newx,action,refs[t],prev) if task=='tracking' else env.reward(action,prev)
                    ys.append(newx.copy());commands.append(env.last_true_command.copy());rewards.append(reward)
                    fprev=c[14:28].copy();prev=action;x=newx
            finally:
                budget.add('collection_'+split,completed_steps,time.perf_counter()-begin)
            e=dict(id=f'{split}:{ep}',split=split,episode_seed=rng_seed,delay=delay,x=np.asarray(xs),c=np.asarray(cs),prev=np.asarray(acts),a=np.asarray(acts[1:]+[prev]),y=np.asarray(ys),u=np.asarray(commands),r=np.asarray(rewards),fprev=np.asarray(fprevs),refs=refs,snapshots=snapshots)
            if task=='vision':e.update(rgb=np.asarray(rgb),next_rgb=np.asarray(next_rgb))
            episodes.append(e)
    def starts(split):
        result=[]
        for e in episodes:
            if e['split']!=split:continue
            t=warmup
            if split=='train' and task in ('contact','vision'):
                legal=sorted(frame for frame in e['snapshots'] if frame>=history-1)
                if not legal:raise ValueError('No snapshot with a complete observable history')
                ordinal=len(result)
                # Rotate the phase offset every three episodes to cross phase with delay.
                t=legal[(ordinal+ordinal//3)%len(legal)]
            result.append(dict(episode_id=e['id'],frame=t,delay=e['delay'],x=e['x'][t],c=e['c'][t],fprev=e['fprev'][t],prev=e['prev'][t],refs=e['refs'][t:],snapshot=e['snapshots'][t],xhist=e['x'][max(0,t-history+1):t+1],chist=e['c'][max(0,t-history+1):t+1],ahist=e['prev'][max(0,t-history+1):t+1]))
        return result
    return dict(episodes=episodes,starts={split:starts(split) for split in ('train','validation','development_test')},lower=env.lower.copy(),upper=env.upper.copy(),alpha=env.alpha,control_dt=env.control_dt,task=task)
def stats(data):
    train=[e for e in data['episodes'] if e['split']=='train']
    x=np.concatenate([e['x'] for e in train]);y=np.concatenate([e['y'] for e in train]);u=np.concatenate([e['u'] for e in train]);delta=y-x
    return dict(xmean=x.mean(0),xstd=np.maximum(x.std(0),.01),umean=u.mean(0),ustd=np.maximum(u.std(0),.001),ymean=delta.mean(0),ystd=np.maximum(delta.std(0),.001),lower=data['lower'],upper=data['upper'],alpha=np.asarray(data['alpha']))
def sequences(data,split,*,horizon=8,history=32,device='cpu'):
    samples=[]
    for e in data['episodes']:
        if e['split']!=split:continue
        for t in range(history-1,len(e['x'])-horizon+1):
            samples.append({**{k:e[k][t:t+horizon] for k in ('x','y','c','a','prev','u','r','fprev','refs')},'xhist':e['x'][t-history+1:t+1],'chist':e['c'][t-history+1:t+1],'ahist':e['prev'][t-history+1:t+1]})
    return {k:torch.as_tensor(np.stack([s[k] for s in samples]),dtype=torch.float32,device=device) for k in samples[0]}
