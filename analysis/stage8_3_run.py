"""Preregistered frozen-model optimizer/data factorial; total steps <=20000."""
import sys,json,time,hashlib,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'vendor/legacy_sim')]
import numpy as np,torch
from scripts.run_stage7c1_diagnosis import make_env
from tasks.reach import ReachConfig,collect_reach_episodes,FKSurrogate
from models.eagm import EAGMBase
from training.stage8_1_geometry import Geometry
from training.stage8_2_dynamics import Kernel,Budget,rollout
from analysis.stage8_2_run import physical
from training.eval_metrics import cosine
O=ROOT/'runs/stage8_3_alignment';P=ROOT/'runs/stage7c2_true_ee_current_sites_seed20'
def dump(n,v):(O/n).write_text(json.dumps(v,indent=2,default=lambda x:x.tolist() if hasattr(x,'tolist') else str(x)))
def update(g,kind,length):
 g=np.asarray(g,dtype=float)
 if kind=='raw':v=g
 elif kind=='adam_first':
  # First standard bias-corrected Adam ascent, fresh zero moments.
  v=g/(np.abs(g)+1e-8)
 else:raise ValueError(kind)
 if np.linalg.norm(v)<1e-15:raise ValueError('zero update; stop')
 return length*v/np.linalg.norm(v)
def main():
 O.mkdir(exist_ok=False);cfg=json.loads((ROOT/'configs/stage8_3_preregistered.json').read_text());dump('protocol.json',cfg);b=Budget(cfg['limit_control_steps']);t0=time.time();phases={};status='running'
 try:
  for p,h in cfg['source_sha256'].items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h
  torch.set_num_threads(1);data=torch.load(P/'data.pt',weights_only=False);train=data['train_starts'][:4];assert [s['episode_id'] for s in train]==cfg['train_episode_ids']
  val=[ep['rows'][0]['start'] for ep in data['episodes'] if ep['episode_id'] in cfg['validation_episode_ids']];assert len(val)==2
  collector=make_env();oldstep=collector.step
  def counted(a):b.tick();return oldstep(a)
  collector.step=counted;fresh=collect_reach_episodes(collector,ReachConfig(goal_mode='random',traj_shape='hold'),n_episodes=2,horizon=200,seed=cfg['fresh_test_seed']);new=[ep['rows'][0]['start'] for ep in fresh];phases['collection']=b.steps;torch.save(dict(episodes=fresh,starts=new),O/'fresh_test.pt')
  groups={'train':train,'validation':val,'fresh_test':new,'historical_test':data['test_starts']};dump('split_inventory.json',{g:[dict(episode=s['episode_id'],frame=s['frame'],collection_seed=83017 if g=='fresh_test' else 20) for s in ss] for g,ss in groups.items()})
  env=make_env();geo=Geometry(env.model);ck=torch.load(P/'dir_random.pt',weights_only=False);model=EAGMBase(ck['stats']);model.load_state_dict(ck['state']);model.eval();fk=FKSurrogate();fk.load_state_dict(torch.load(P/'fk.pt',weights_only=True));fk.eval()
  for p in list(model.parameters())+list(fk.parameters()):p.requires_grad_(False)
  ks={g:[Kernel(env,s['snapshot'],b) for s in ss] for g,ss in groups.items()};before=b.steps;grads=[]
  for i,(s,k) in enumerate(zip(train,ks['train'])):
   for name,c in [('learned','B'),('oracle','D')]:
    th=torch.zeros(28,requires_grad=True);j=rollout(model,th,s,c,fk,geo,k);g=torch.autograd.grad(j,th)[0].numpy();grads.append(dict(start=i,source=name,g=g,forward_value=float(j.detach())))
  phases['training_gradients']=b.steps-before;dump('training_gradients.json',grads);print('gradients',b.steps,flush=True)
  policies={}
  for K in [2,4]:
   for source in ['learned','oracle']:
    g=np.mean([r['g'] for r in grads if r['source']==source and r['start']<K],axis=0)
    for opt in ['raw','adam_first']:
     th=update(g,opt,cfg['update_l2']);assert np.max(abs(th))<1 and abs(np.linalg.norm(th)-cfg['update_l2'])<1e-12
     policies[f'K{K}_{source}_{opt}']=dict(K=K,source=source,optimizer=opt,g=g,theta=th,step_norm=float(np.linalg.norm(th)),theta_saturation=0.,rho_proj=1.)
  dump('policies.json',policies)
  before=b.steps;references=[]
  for group in ['train','validation','fresh_test']:
   for i,(s,k) in enumerate(zip(groups[group],ks[group])):
    epss=[.01] if group=='train' else [.005,.01];gg=[]
    for eps in epss:
     g=np.zeros(28)
     for j in range(28):
      p=np.zeros(28);m=p.copy();p[j]=eps;m[j]=-eps;g[j]=(physical(k,p,s,geo)[1][-1]-physical(k,m,s,geo)[1][-1])/(2*eps)
     gg.append(g)
    mask=None if len(gg)==1 else (np.ptp(gg,axis=0)<=np.maximum(1e-9,.1*np.abs(gg[-1]))).astype(int)
    references.append(dict(group=group,start=i,epsilons=epss,g=gg[-1],all_g=gg,stable_mask=mask,boundary_mask=[0]*28,fd_mode='central',scale_cosine=None if len(gg)==1 else cosine(*gg)))
  historical=json.loads((ROOT/'runs/stage8_2a_single_step/references.json').read_text())
  for i,r in enumerate(historical):references.append(dict(group='historical_test',start=i,g=r['gradients'][7],reused=True,source='runs/stage8_2a_single_step/references.json'))
  phases['reference_fd']=b.steps-before;dump('references.json',references)
  # Numerical label gate before return comparisons; do not select epsilon after inspection.
  gate=all(r['scale_cosine']>.999 and sum(r['stable_mask'])>=26 for r in references if r['group'] in ['validation','fresh_test']);dump('reference_gate.json',dict(passed=gate,rule='cosine>.999 and >=26/28 stable for all validation/fresh starts',test_selected=False))
  if not gate:status='stopped_unstable_reference';return
  before=b.steps;returns=[]
  for group,starts in groups.items():
   for i,(s,k) in enumerate(zip(starts,ks[group])):
    for key,th in [('theta0',np.zeros(28))]+[(n,v['theta']) for n,v in policies.items()]:
     rs=physical(k,th,s,geo,H=16)[1];returns.append(dict(group=group,start=i,policy=key,return_h8=rs[7],return_h16=rs[15]))
  phases['evaluation']=b.steps-before;dump('returns.json',returns);status='completed';print('DONE',b.steps,flush=True)
 except Exception:
  status='failed';(O/'failure.log').write_text(traceback.format_exc());raise
 finally:dump('budget.json',dict(status=status,control_steps=b.steps,cap=b.limit,phases=phases,wall_seconds=time.time()-t0,device='CPU',geometry_calls=locals().get('geo').calls if 'geo' in locals() else 0))
if __name__=='__main__':main()
