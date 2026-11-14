"""Authorized development runner. Confirmation roots are hard-blocked."""
import os,sys
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import argparse,copy,time,traceback
import numpy as np,torch
from eagm2.runtime import *
from eagm2.data import make_env,collect,stats,sequences
from eagm2.training import METHODS,make_model,train_model,train_critic,optimize
from eagm2.gradients import finite_difference
from eagm2.rollout import physical
from eagm2.budget_controls import run_direct_controls
from eagm2.extra_data import collect_extra_training
from eagm2.evaluation import model_metrics,evaluate_gradients,evaluate_returns

def make_labels(env,starts,seed,task,horizon,budget,kind,vision_policy=None):
 rng=np.random.default_rng(stream_seed(seed,kind));labels=[];dim=28 if vision_policy is None else 32
 for i,s in enumerate(starts):
  theta=np.zeros(dim) if i%2==0 else rng.normal(0,.3,dim).clip(-.8,.8)
  fd=finite_difference(lambda th:physical(env,th,s,horizon,budget,kind,task=task,vision_policy=vision_policy),theta)
  labels.append(dict(start=s,**fd));print(kind,i+1,'queries',fd['queries'],flush=True)
 return labels

def selected_methods(task, requested=None):
 names=(requested.split(',') if requested is not None else
        [m for m in METHODS if task!='vision' or not m.startswith('legacy_')]
        +([] if task=='vision' else ['base_extra_data'])+['direct_fd','spsa'])
 if not names or len(set(names))!=len(names):raise ValueError('methods must be nonempty and unique')
 unknown=set(names)-set(METHODS)-{'base_extra_data','direct_fd','spsa'}
 if unknown:raise ValueError(f'unknown methods: {sorted(unknown)}')
 if task=='vision' and any(m.startswith('legacy_') or m=='base_extra_data' for m in names):
  raise ValueError('vision does not support frozen physical-state legacy or extra-data control')
 if bool('direct_fd' in names)!=bool('spsa' in names):
  raise ValueError('select direct_fd and spsa together for matched-budget comparison')
 return names

def main():
 p=argparse.ArgumentParser();p.add_argument('--task',choices=['tracking','contact','vision'],required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--attempt',required=True);p.add_argument('--device',default='cuda');p.add_argument('--updates',type=int,default=1000);p.add_argument('--policy-updates',type=int,default=50);p.add_argument('--methods',default=None);p.add_argument('--smoke',action='store_true');a=p.parse_args()
 methods=selected_methods(a.task,a.methods);check_seed(a.task,a.seed);seed_all(a.seed);out=new_run(a.task,a.seed,a.attempt);start=time.perf_counter();budget=Budget();env=None;status=[];active_method=None
 class Tee:
  def __init__(self,stream,path):self.stream=stream;self.file=open(path,'a',buffering=1)
  def write(self,text):self.stream.write(text);self.file.write(text);return len(text)
  def flush(self):self.stream.flush();self.file.flush()
 sys.stdout=Tee(sys.stdout,out/'training_logs/console.log');sys.stderr=Tee(sys.stderr,out/'training_logs/console.log')
 config=vars(a)|dict(resolved_methods=methods,phase='development_exploratory',confirmation_authorized=False,history=32,training_gradient_labels=2 if a.smoke else 12,diagnostic_gradient_labels=2 if a.smoke else 6,model_batch_size=16 if a.smoke else 64,selection='development only; no confirmatory interpretation')
 dump(out/'config.json',config);dump(out/'environment.json',environment());dump(out/'run_manifest.json',dict(status='running',config=config))
 try:
  if a.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA requested but unavailable; run host GPU authorized command; never silently label CPU as GPU')
  env=make_env(a.task);horizon=8 if a.task=='tracking' else 32
  data=collect(env,a.task,a.seed,budget,train_eps=3 if a.smoke else 18,val_eps=3 if a.smoke else 6,test_eps=3 if a.smoke else 30,length=96 if a.smoke else 160)
  torch.save(data,out/'dataset.pt');dump(out/'split_manifest.json',[dict(id=e['id'],split=e['split'],episode_seed=e['episode_seed'],delay=e['delay'],frames=len(e['x'])) for e in data['episodes']])
  reward_fn=None;vision_policy=None;vision_report=None
  if a.task=='contact':
   from eagm2.contact import TorchContactReward
   reward_fn=TorchContactReward(env).to(a.device).float()
  if a.task=='vision':
   from eagm2.visual_training import prepare_visual
   data,vision_policy,reward_fn,vision_report=prepare_visual(data,out,a.seed,a.device,smoke=a.smoke)
   dump(out/'vision_training.json',vision_report)
  statistic=stats(data);torch.save(statistic,out/'normalization.pt')
  tr=sequences(data,'train',horizon=horizon,device=a.device);val=sequences(data,'validation',horizon=horizon,device=a.device);te=sequences(data,'development_test',horizon=horizon,device=a.device)
  nlabel=2 if a.smoke else 12;ndiag=2 if a.smoke else 6
  labels=make_labels(env,data['starts']['train'][:nlabel],a.seed,a.task,horizon,budget,'labels_train',vision_policy)
  vallabs=make_labels(env,data['starts']['validation'][:ndiag],a.seed,a.task,horizon,budget,'labels_validation',vision_policy)
  testlabs=make_labels(env,data['starts']['development_test'][:ndiag],a.seed,a.task,horizon,budget,'labels_diagnostic',vision_policy)
  torch.save(dict(train=labels,validation=vallabs,diagnostic=testlabs),out/'gradient_labels.pt')
  train_collection_steps=budget.steps.get('collection_train',0)
  train_label_steps=budget.steps.get('labels_train',0)
  direct_cap=train_collection_steps+train_label_steps
  dump(out/'matched_budget_protocol.json',dict(unit='simulator_control_steps',
       common_train_collection_steps=train_collection_steps,training_fd_label_steps=train_label_steps,
       direct_control_cap_each=direct_cap,base_extra_data_increment_cap=train_label_steps,
       excluded=['validation collection','test collection','validation FD','diagnostic FD','final evaluation'],
       note='Direct controls do not consume or pay for unused model data; they receive the same total cap. Extra-data uses ff_base loss, same model update count and original policy starts.'))
  extra_tr=None;extra_statistic=None;extra_manifest=None
  if 'base_extra_data' in methods:
   active_method='base_extra_data_collection'
   extra_episodes,extra_manifest=collect_extra_training(env,a.task,a.seed,train_label_steps,budget,
                                                       length=96 if a.smoke else 160,horizon=horizon)
   torch.save(extra_episodes,out/'extra_training_episodes.pt');dump(out/'extra_training_manifest.json',extra_manifest)
   existing_seeds={e['episode_seed'] for e in data['episodes']}
   if existing_seeds & {e['episode_seed'] for e in extra_episodes}:raise RuntimeError('extra episode seed collides with common split')
   extra_data={**data,'episodes':data['episodes']+extra_episodes}
   extra_statistic=stats(extra_data);torch.save(extra_statistic,out/'extra_normalization.pt')
   extra_tr=sequences(extra_data,'train',horizon=horizon,device=a.device)
  critic=None;critic_report=None
  if any(METHODS[m]['loss'] in ('paml','vagram') for m in methods if m in METHODS):
   with measured(a.device) as cres:critic,critic_report=train_critic(tr,a.seed,a.device,updates=50 if a.smoke else 1000,validation=val,vision_policy=vision_policy)
   critic_report['resource']=cres;dump(out/'critic_training.json',critic_report);torch.save(dict(state_dict=critic.state_dict(),training_checkpoint=getattr(critic,'training_checkpoint',None)),out/'checkpoints/critic.pt')
  eval_h=16 if a.task=='tracking' else max(1,int(np.ceil(10/env.control_dt))-32)
  dim=28 if vision_policy is None else 32
  initial=evaluate_returns(env,np.zeros(dim),data['starts']['development_test'],eval_h,budget,a.task,vision_policy=vision_policy)
  dump(out/'initial_policy.json',initial)
  for name in [m for m in methods if m not in ('direct_fd','spsa')]:
   active_method=name;mbegin=time.perf_counter()
   train_name='ff_base' if name=='base_extra_data' else name
   method_tr=extra_tr if name=='base_extra_data' else tr
   method_stats=extra_statistic if name=='base_extra_data' else statistic
   model=make_model(method_stats,train_name,a.seed,a.device,obs_dim=method_tr['x'].shape[-1])
   count=model.parameter_counts();dump(out/name/'initialization_manifest.json',getattr(model,'initialization_manifest',{}));kwargs=dict(reward_fn=reward_fn,vision_policy=vision_policy)
   train_report=train_model(model,train_name,method_tr,labels,a.seed,updates=a.updates,batch_size=16 if a.smoke else 64,horizon=horizon,critic=critic,**kwargs)
   torch.save(dict(state_dict=model.state_dict(),stats=method_stats,method=name,implementation_method=train_name,model_config=dict(obs_dim=model.obs_dim,width=model.width,memory=model.memory,prior=model.prior),optimizer=train_report.pop('optimizer_state'),torch_rng=train_report.pop('torch_rng'),batch_generator_state=train_report.pop('batch_generator_state'),seed=a.seed),out/'checkpoints'/f'{name}.pt')
   dump(out/'training_logs'/f'{name}.json',train_report)
   grad=evaluate_gradients(model,testlabs,horizon,**kwargs)
   validation_grad=evaluate_gradients(model,vallabs,horizon,**kwargs)
   theta,pol=optimize(model,data['starts']['train'][:2 if a.smoke else 6],horizon,updates=a.policy_updates,**kwargs)
   torch.save(dict(theta=theta,optimizer=pol.pop('optimizer_state'),torch_rng=pol.pop('torch_rng'),cuda_rng=torch.cuda.get_rng_state_all() if a.device.startswith('cuda') else [],updates=a.policy_updates),out/'checkpoints'/f'{name}_policy.pt')
   result=dict(status='completed',phase='exploratory',method=name,seed=a.seed,parameter_counts=count,training=train_report,model_metrics=model_metrics(model,te,horizon),validation_model_metrics=model_metrics(model,val,horizon),gradient=grad,policy_optimization=pol,theta_final=theta,physical=evaluate_returns(env,theta,data['starts']['development_test'],eval_h,budget,a.task,vision_policy=vision_policy),wall_seconds=time.perf_counter()-mbegin,critic_adaptation=critic_report if METHODS[train_name]['loss'] in ('paml','vagram') else None)
   result['validation_gradient']=validation_grad
   result['extra_data_comparison']=extra_manifest if name=='base_extra_data' else None
   result['training_simulator_budget']=dict(collection=train_collection_steps,
       gradient_labels=train_label_steps if METHODS[train_name]['loss'] not in ('none','paml','vagram') else 0,
       extra_collection=extra_manifest['issued_control_steps'] if name=='base_extra_data' else 0,
       labels_generated_but_unused=METHODS[train_name]['loss'] in ('none','paml','vagram'))
   result['validation_physical']=evaluate_returns(env,theta,data['starts']['validation'],eval_h,budget,a.task,kind='validation_evaluation',vision_policy=vision_policy)
   result['delta_return_vs_initial']=result['physical']['mean_return']-initial['mean_return']
   result['wall_seconds']=time.perf_counter()-mbegin
   dump(out/name/'results.json',result);status.append(dict(method=name,status='completed',wall_seconds=result['wall_seconds']))
   dump(out/'experiment_status.json',status);dump(out/'budget.json',budget.as_dict())
   print('COMPLETED',name,'return',result['physical']['mean_return'],'cos',grad['mean_cosine'],flush=True)
  if 'direct_fd' in methods:
   active_method='direct_controls'
   direct_resources={}
   try:
    with measured(a.device) as direct_resources:
     controls=run_direct_controls(env,data['starts']['train'][:2 if a.smoke else 6],
                                 data['starts']['development_test'],initial,a.seed,a.task,horizon,
                                 eval_h,budget,out,direct_cap,vision_policy=vision_policy)
   finally:
    dump(out/'direct_controls_resource.json',direct_resources)
   for name,result in controls.items():
    status.append(dict(method=name,status='completed',wall_seconds=result['wall_seconds']))
    print('COMPLETED',name,'return',result['physical']['mean_return'],'training_steps',result['actual_training_control_steps'],flush=True)
   dump(out/'experiment_status.json',status);dump(out/'budget.json',budget.as_dict())
  dump(out/'results.json',dict(status='completed',phase='exploratory',task=a.task,seed=a.seed,methods=status,wall_seconds=time.perf_counter()-start,budget=budget.as_dict(),confirmation_authorized=False))
  dump(out/'run_manifest.json',dict(status='completed',config=config,wall_seconds=time.perf_counter()-start))
 except Exception:
  status.append(dict(method=active_method,status='failed',traceback=traceback.format_exc()))
  dump(out/'experiment_status.json',status);dump(out/'budget.json',budget.as_dict())
  dump(out/'failure.json',dict(traceback=traceback.format_exc(),wall_seconds=time.perf_counter()-start,budget=budget.as_dict()));dump(out/'run_manifest.json',dict(status='failed',config=config));raise
 finally:
  if env is not None and hasattr(env,'close'):env.close()
if __name__=='__main__':main()
