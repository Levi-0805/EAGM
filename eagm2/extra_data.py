"""Extra training-only dynamics data under a hard simulator interaction cap.

RNGs below are named substreams of the authorized exploratory root, not new
research seeds. No validation/test episodes are collected or consulted.
"""
from __future__ import annotations
import time
from .data import collect
from .runtime import check_seed, stream_seed


class _CappedEnv:
    def __init__(self,env,cap):
        self._env=env;self.cap=cap;self.calls=0
    def __getattr__(self,name):
        return getattr(self._env,name)
    def step(self,action):
        if self.calls>=self.cap:
            raise RuntimeError('extra training simulator step cap reached')
        # Charge issued calls even if the simulator raises after integration;
        # never hide partially consumed failed collection attempts.
        self.calls+=1
        return self._env.step(action)


class _ExtraBudget:
    def __init__(self,budget):
        self.budget=budget;self.recorded=0;self.seconds=0.
    def add(self,kind,n,seconds=0):
        if kind!='collection_train':
            raise RuntimeError('extra-data collector attempted a held-out split')
        self.recorded+=int(n);self.seconds+=float(seconds)
        self.budget.add('collection_extra_train',n,seconds)


def collect_extra_training(env,task,seed,control_step_cap,budget,length=160,warmup=32,
                           *,history=32,horizon=None):
    """Return (training episodes, manifest), never exceeding control_step_cap.

    Full episodes cost length+warmup. A final shortened episode is permitted if
    enough room remains for both causal history and at least the task horizon.
    The shared collector additionally requires 32 post-warmup steps for its
    snapshot contract; any smaller remainder is explicitly left unused.
    Every episode uses id extra_train:N but split='train', so it can be appended
    to data['episodes'] without making any test/validation data trainable.
    """
    if task not in ('tracking','contact'):
        raise ValueError('extra training comparison currently supports tracking/contact only')
    check_seed(task,seed)
    if isinstance(control_step_cap,bool) or int(control_step_cap)!=control_step_cap or control_step_cap<0:
        raise ValueError('control_step_cap must be a nonnegative integer')
    if not all(isinstance(v,int) and not isinstance(v,bool) and v>0 for v in (length,warmup,history)):
        raise ValueError('length, warmup and history must be positive integers')
    horizon=(8 if task=='tracking' else 32) if horizon is None else horizon
    if not isinstance(horizon,int) or isinstance(horizon,bool) or horizon<=0:
        raise ValueError('horizon must be a positive integer')
    if warmup<history-1:
        raise ValueError('warmup cannot provide requested causal history')
    min_payload=max(horizon,32)
    if length<min_payload:
        raise ValueError('length must satisfy task horizon and collector snapshot requirements')
    cap=int(control_step_cap);per_episode=length+warmup
    full_count,remainder=divmod(cap,per_episode)
    groups=[]
    if full_count:groups.append(('full',full_count,length))
    if remainder>=warmup+min_payload:groups.append(('remainder',1,remainder-warmup))
    wrapped=_CappedEnv(env,cap);meter=_ExtraBudget(budget);episodes=[];streams=[]
    start=time.perf_counter()
    try:
        for group,n,payload in groups:
            stream_name='extra_train/'+group
            derived=stream_seed(seed,stream_name)
            batch=collect(wrapped,task,derived,meter,train_eps=n,val_eps=0,test_eps=0,
                          length=payload,warmup=warmup,history=history)
            for e in batch['episodes']:
                if e['split']!='train':raise RuntimeError('unexpected nontraining extra episode')
                original_id=e['id'];e['id']=f'extra_train:{len(episodes)}'
                e['extra_data_provenance']=dict(root_seed=seed,root_phase='development_exploratory',
                    substream_name=stream_name,substream_seed=derived,collector_episode_id=original_id)
                episodes.append(e)
            streams.append(dict(name=stream_name,derived_seed=derived,episodes=n,payload_steps=payload))
    finally:
        missing=wrapped.calls-meter.recorded
        if missing<0:raise RuntimeError('collector reported more steps than issued')
        if missing:
            budget.add('collection_extra_train_failed',missing,max(0,time.perf_counter()-start-meter.seconds))
    assert wrapped.calls<=cap
    if len({e['episode_seed'] for e in episodes})!=len(episodes):
        raise RuntimeError('extra episode RNG collision; do not use this collected data')
    manifest=dict(root_seed=seed,task=task,phase='development_exploratory',
        confirmation_authorized=False,stream_role='extra dynamics training substreams, not new research roots',
        control_step_cap=cap,issued_control_steps=wrapped.calls,unused_control_steps=cap-wrapped.calls,
        episode_count=len(episodes),warmup=warmup,history=history,horizon=horizon,
        wall_seconds=time.perf_counter()-start,streams=streams,
        episode_ids=[e['id'] for e in episodes],episode_seeds=[e['episode_seed'] for e in episodes],
        heldout_episodes_collected=0,partial_remainder_policy='collect only if warmup+max(horizon,32) fits; else unused',
        normalization_note='fit extra-data model normalization on common train plus these train episodes only')
    return episodes,manifest
