"""Sparse exploratory OOD postprocessing, never training or confirmation.
Run with --source runs/.../development_v1 --output reports/.../new_dir.
Optional --fd adds one start, gain shift only, at theta0 and frozen final theta.
"""
import argparse,dataclasses,hashlib,json,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from eagm2.runtime import Budget,check_seed,dump,stream_seed,seed_all
from eagm2.data import CFG,collect,make_env
from eagm2.models import ExecutionWorldModel
from eagm2.rollout import physical,gradient
from eagm2.gradients import finite_difference,gradient_metrics
METHODS=('ff_base','combined_dir','joint_calibration')
SHIFTS={'gain':{'gain':1.15},'smoothing':{'smooth_beta':.5},'rate':{'rate_limit':.025},'extra_lag':{'extra_lag':2}}

class CappedEnv:
    def __init__(self,env,counter,cap):self.env=env;self.counter=counter;self.cap=cap
    def __getattr__(self,key):return getattr(self.env,key)
    def step(self,action):
        if self.counter[0]>=self.cap:raise RuntimeError('OOD interaction cap reached')
        self.counter[0]+=1
        return self.env.step(action)

def load_source(source):
    source=Path(source).resolve()
    if source.name!='development_v1':raise ValueError('Only explicitly specified development_v1 supported')
    config=json.loads((source/'config.json').read_text());task=config['task'];seed=config['seed']
    if task not in ('tracking','contact'):raise ValueError('Vision reconstruction intentionally unsupported')
    if config.get('smoke'):raise ValueError('Refusing smoke run as development checkpoint')
    check_seed(task,seed)
    files=[source/'config.json',source/'source_hashes.json']
    models={};policies={}
    for method in METHODS:
        mp=source/'checkpoints'/f'{method}.pt';pp=source/'checkpoints'/f'{method}_policy.pt';files.extend((mp,pp))
        checkpoint=torch.load(mp,map_location='cpu',weights_only=False)
        if checkpoint['method']!=method or checkpoint['seed']!=seed:raise ValueError('Checkpoint provenance mismatch')
        model=ExecutionWorldModel(checkpoint['stats'],**checkpoint['model_config'])
        model.load_state_dict(checkpoint['state_dict'],strict=True);model.eval();models[method]=model
        theta=torch.load(pp,map_location='cpu',weights_only=False)['theta']
        theta=np.asarray(theta)
        if theta.shape!=(28,) or not np.isfinite(theta).all():raise ValueError('Invalid final theta')
        policies[method]=theta.copy()
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    return task,seed,models,policies,hashes

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--fd',action='store_true');parser.add_argument('--max-control-steps',type=int,default=50000);args=parser.parse_args()
    task,seed,models,policies,hashes=load_source(args.source)
    output=args.output.resolve()
    if not output.is_relative_to(ROOT/'reports/eagm2_dev'):raise ValueError('Output must be independent reports/eagm2_dev subdirectory')
    output.mkdir(parents=True,exist_ok=False);seed_all(seed);budget=Budget();counter=[0];started=time.perf_counter()
    dump(output/'source_hashes.json',hashes)
    dump(output/'config.json',dict(source=args.source,root_seed=seed,task=task,methods=METHODS,shifts=SHIFTS,fd=args.fd,fd_configuration='gain only',fd_start=0,fd_positions=['theta0','final'],fd_scales=[.5,1.],max_control_steps=args.max_control_steps,starts_per_config=3,training_updates=0,note='Sparse exploratory diagnosis only; not sufficient OOD evidence, not independent confirmation. Fixed methods/configurations before results. Fresh naturally evolved hidden states; no snapshot editing.'))
    implementation=[Path(__file__)]+list((ROOT/'eagm2').glob('*.py'))+list((ROOT/'controllers').glob('*.py'))
    dump(output/'diagnostic_source_hashes.json',{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in implementation})
    rows=[];fds=[]
    try:
        for name,change in SHIFTS.items():
            cfg=dataclasses.replace(CFG,**change)
            env=CappedEnv(make_env(task,cfg),counter,args.max_control_steps)
            derived=stream_seed(seed,'sparse_ood_v1/'+name)
            data=collect(env,task,derived,budget,train_eps=0,val_eps=3,test_eps=0)
            torch.save(data,output/f'{name}_observable_history_and_snapshots.pt')
            starts=data['starts']['validation'];horizon=16 if task=='tracking' else max(1,int(np.ceil(10/env.control_dt))-32)
            reward=None
            if task=='contact':
                from eagm2.contact import TorchContactReward
                reward=TorchContactReward(env).float()
            for i,start in enumerate(starts):
                baseline=physical(env,np.zeros(28),start,horizon,budget,'ood_returns',task=task)
                for method in METHODS:
                    final=physical(env,policies[method],start,horizon,budget,'ood_returns',task=task)
                    rows.append(dict(configuration=name,start=i,delay=start['delay'],method=method,theta0_return=baseline,final_return=final,paired_delta=final-baseline))
            dump(output/'per_start_returns.json',rows)
            if args.fd and name=='gain':
                start=starts[0];h=8 if task=='tracking' else 32
                for method in METHODS:
                    for position,theta in [('theta0',np.zeros(28)),('final',policies[method])]:
                        fd=finite_difference(lambda t:physical(env,t,start,h,budget,'ood_fd',task=task),theta,scales=(.5,1.))
                        pred=gradient(models[method],start,theta,h,reward_fn=reward)
                        fds.append(dict(configuration=name,start=0,method=method,position=position,fd=fd,predicted_gradient=pred,metrics=gradient_metrics(pred,fd['gradient'],fd['mask'])))
                        dump(output/'gradient_diagnostics.json',fds)
        for p,sha in hashes.items():
            if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=sha:raise RuntimeError('Source artifact changed')
        dump(output/'results.json',dict(status='completed',phase='sparse_exploratory_ood',rows=rows,gradients=fds,source_unchanged=True))
    except Exception:
        dump(output/'failure.json',dict(traceback=traceback.format_exc(),completed_rows=rows));raise
    finally:
        dump(output/'budget.json',dict(**budget.as_dict(),issued_control_steps=counter[0],wall_seconds=time.perf_counter()-started,training_updates=0))
if __name__=='__main__':main()
