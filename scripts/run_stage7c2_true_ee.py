"""Seed20 protocol correction. Refuses overwrites; no multiseed launch."""
import sys, json, csv, hashlib, subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'vendor/legacy_sim')]
import numpy as np
import torch
def rank_correlation(a,b):
    def ranks(x):
        return np.array([np.sum(x<v)+(np.sum(x==v)-1)/2 for v in x])
    ra,rb=ranks(a),ranks(b)
    return float(np.corrcoef(ra,rb)[0,1]) if ra.std()>0 and rb.std()>0 else float("nan")
from scripts.run_stage7c1_diagnosis import make_env, pack_rows, model_grad
from tasks.reach import ReachConfig, collect_reach_episodes, fit_fk_surrogate, ee_from_q_mujoco, reach_reward_surrogate, reach_reward_true_mujoco
from training.dataset import make_split, fit_stats, pick_starts
from training.seeds import seed_everything
from training.rollout import set_reach_fk_surrogate, physical_return_true_ee, physical_return_surrogate
from training.gradients import sim_grad_true_ee_masked_detailed
from training.eval_metrics import gradient_quality
from training.stage6c_common import random_probe_thetas
from training.paired_train import train_pair, train_mc, clone_model
from training.policy_opt import policy_optimize
from models.eagm import EAGMBase, McWorld
from models.baselines import AuxCommandMLP
OUT=ROOT/'runs/stage7c2_true_ee_current_sites_seed20'
def dump(name,obj):
    (OUT/name).write_text(json.dumps(obj,indent=2,default=lambda x:x.tolist() if hasattr(x,'tolist') else str(x)))
def main():
    OUT.mkdir(exist_ok=False)
    seed_everything(20);env=make_env();steps=0;queries=0
    original_step=env.step
    def counted(a):
        nonlocal steps
        steps+=1
        return original_step(a)
    env.step=counted
    fk=fit_fk_surrogate(env,n_samples=4000,seed=20,steps=800);set_reach_fk_surrogate(fk)
    torch.save(fk.state_dict(),OUT/'fk.pt')
    eps=collect_reach_episodes(env,ReachConfig(goal_mode='random',traj_shape='hold'),n_episodes=36,horizon=200,seed=20)
    split=make_split(36,seed=20,n_train=24,n_val=6,n_test=6)
    rows=lambda ids:[r for ep in eps if ep['episode_id'] in set(ids) for r in ep['rows']]
    trr,ter=rows(split.train_eps),rows(split.test_eps)
    stats=fit_stats(trr,env_lower=env.lower,env_upper=env.upper);tr=pack_rows(trr);te=pack_rows(ter)
    starts=pick_starts(trr,delay=None,max_per_episode=1,max_total=12)
    test=pick_starts(ter,delay=None,max_per_episode=1,max_total=12)
    torch.save(dict(episodes=eps,split=split,train_starts=starts,test_starts=test,stats=stats),OUT/'data.pt')
    budget={'collection_steps':steps,'fk_fit_samples':4000,'fk_fit_steps':800}
    before=steps;labs=[]
    for st,th in zip(starts,random_probe_thetas(12,121)):
        g,m,n,meta=sim_grad_true_ee_masked_detailed(env,st,th,horizon=8)
        labs.append(dict(start=st,theta=th,g=g,mask=m,s0=float(np.median(abs(g[m>0.5]))) if m.any() else 1.,control_steps=n,fd_meta=meta))
    budget['train_label_steps']=steps-before;budget['train_label_queries']=sum(l['fd_meta']['simulator_queries'] for l in labs)
    assert budget['train_label_steps']==sum(l['control_steps'] for l in labs)
    torch.save(labs,OUT/'true_labels.pt')
    print('labels complete',budget,flush=True)
    seed_everything(27);mc=McWorld(stats)
    seed_everything(31);init=EAGMBase(stats);base=clone_model(init);dire=clone_model(init)
    assert all(torch.equal(a,b) for a,b in zip(base.parameters(),dire.parameters()))
    seed_everything(31);aux=AuxCommandMLP(stats)
    train_mc(mc,tr,3000,batch_seed=50)
    logs=train_pair(base,dire,tr,updates=3000,use_history=False,dir_labels=labs,batch_seed=50)
    train_pair(aux,None,tr,updates=3000,use_history=False,batch_seed=50)
    models={'mc':mc,'base':base,'dir_random':dire,'aux':aux}
    policies={};results={};per=[];grads=[]
    for name,model in models.items():
        torch.save(dict(state=model.state_dict(),stats=stats),OUT/(name+'.pt'))
        pol=policy_optimize(model,starts,use_history=False,is_mc=name=='mc',updates=50,lr=.03,seed=70)
        policies[name]=pol
        with torch.no_grad():
            if name=='mc':y=model(te['x'],te['c'],te['a']);cmd=None
            else:y,u=model(te['x'],te['c_obs'],te['a']);cmd=float((u-te['u_exec']).abs().mean())
            nmse=float((((y-te['y'])/model.ystd)**2).mean())
        for i,st in enumerate(test):
            record=dict(model=name,start=i,episode_id=st['episode_id'])
            for h in [4,16]:
                for tag,th in [('theta0',pol['theta0']),('final',pol['theta'])]:
                    for obj,fn in [('true',physical_return_true_ee),('surrogate',physical_return_surrogate)]:
                        record[f'{obj}_{tag}_h{h}']=fn(env,th,st,horizon=h)[0]
            per.append(record)
        results[name]=dict(nmse=nmse,command_mae=cmd,theta_sat=pol['final_theta_param_sat_frac'],rho_proj=pol['mean_rho_proj'])
        dump('policies.json',policies)
        print('policy complete',name,flush=True)
    before=steps
    # Common theta0 plus each model final, all held-out starts. Store coordinate arrays.
    for name,model in models.items():
        for tag,th in [('theta0',np.zeros(28)),('final',policies[name]['theta'])]:
            for i,st in enumerate(test):
                gt,mask,n,meta=sim_grad_true_ee_masked_detailed(env,st,th,horizon=8)
                gh,_=model_grad(model,st,th,is_mc=name=='mc')
                # Surrogate FD on exactly the same feasible stencils, independently scored.
                gs=[]
                for j in range(28):
                    p=np.array(th).copy();m=p.copy();p[j]=min(1,p[j]+.05);m[j]=max(-1,m[j]-.05)
                    gs.append((physical_return_surrogate(env,p,st,horizon=8)[0]-physical_return_surrogate(env,m,st,horizon=8)[0])/(p[j]-m[j]))
                q=gradient_quality(gh,gt,mask)
                q.update(magnitude_error=float(np.linalg.norm(gh-gt)/(np.linalg.norm(gt)+1e-12)))
                grads.append(dict(model=name,theta=tag,start=i,metrics=q,surrogate_metrics=gradient_quality(gh,np.array(gs),mask),true_vs_surrogate=gradient_quality(np.array(gs),gt,mask),g_model=gh,g_true=gt,g_surrogate=gs,mask=mask,fd=meta))
            dump('gradients.json',grads);print('gradient complete',name,tag,flush=True)
    budget['gradient_audit_steps']=steps-before
    audits=[]
    for st in test:
        env.restore(st['snapshot']);x=np.asarray(st['x']);a=np.zeros(14)
        left,right=ee_from_q_mujoco(env,x[:14]);ee=np.r_[left,right];pred=fk(torch.tensor(x[:14],dtype=torch.float32)).detach().numpy()
        rt=reach_reward_true_mujoco(env,x,a,a,st['ee_goals']);rs=reach_reward_surrogate(fk,x,a,a,st['ee_goals'])
        audits.append(dict(ee_error=(pred-ee).tolist(),ee_l2=float(np.linalg.norm(pred-ee)),abs_reward_error=abs(rs-rt),relative_reward_error=abs(rs-rt)/max(abs(rt),1e-12)))
    # Independent FK validation, without fitting or threshold changes.
    rng=np.random.default_rng(2020);errs=[]
    for _ in range(1000):
        q=rng.uniform(env.lower,env.upper);l,r=ee_from_q_mujoco(env,q)
        errs.append(fk(torch.tensor(q,dtype=torch.float32)).detach().numpy()-np.r_[l,r])
    audit=dict(fit_rmse=fk.fit_rmse,independent_rmse=float(np.sqrt(np.mean(np.square(errs)))),accept_rmse_m=.02,starts=audits)
    audit['fk_accept']=audit['fit_rmse']<=.02 and audit['independent_rmse']<=.02
    for name in models:
        rr=[r for r in per if r['model']==name];t=np.array([r['true_final_h16'] for r in rr]);s=np.array([r['surrogate_final_h16'] for r in rr]);z=np.array([r['true_theta0_h16'] for r in rr])
        results[name].update(true_return=float(t.mean()),surrogate_return=float(s.mean()),true_theta0=float(z.mean()),delta=float((t-z).mean()),fail_rate=float(((t-z)<0).mean()),rank_correlation=rank_correlation(t,s))
        for tag in ['theta0','final']:
            gg=[g['metrics'] for g in grads if g['model']==name and g['theta']==tag]
            results[name][tag+'_gradient']={k:float(np.nanmean([g[k] for g in gg])) for k in gg[0]}
    budget['total_control_steps']=steps
    budget['evaluation_control_steps']=steps-budget['collection_steps']-budget['train_label_steps']
    budget['note']='All env.step calls counted. FK/IK queries are kinematic evaluations, not control steps. All models share collection; only Dir consumes training FD labels.'
    dump('reward_audit.json',audit);dump('results.json',results);dump('budget.json',budget);dump('training.json',logs)
    with (OUT/'per_start.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(per[0]));w.writeheader();w.writerows(per)
    dump('manifest.json',dict(base_commit='bcdce42f7f93cda2dc3b9b090c241b0809660187',seed=20,torch=torch.__version__,files={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in list((ROOT/'training').glob('*.py'))+list((ROOT/'tasks').glob('*.py'))+list((ROOT/'models').glob('*.py'))+[Path(__file__)]}))
    print('DONE',results,flush=True)
if __name__=='__main__':main()
