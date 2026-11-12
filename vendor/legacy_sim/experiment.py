"""CPU OpenArm diagnostic. All numbers are measured; no hardware interfaces."""
from pathlib import Path
import argparse, copy, csv, json, time, hashlib, platform, sys
import numpy as np
import torch
from torch import nn
from sim_env import ArmEnv

ROOT=Path(__file__).resolve().parent
torch.set_num_threads(4)
DTYPE=torch.float32

def ref_path(center, seed, n, dt):
    rng=np.random.default_rng(7300+seed)
    amp=rng.uniform(.035,.10,14); freq=rng.uniform(.07,.22,14); phase=rng.uniform(-np.pi,np.pi,14)
    t=np.arange(n)*dt
    return center[None]+amp[None]*np.sin(2*np.pi*t[:,None]*freq[None]+phase[None])

def actor(theta, x, c, ref):
    # theta: 14 bounded residual biases + 14 residual error gains; all rad.
    return torch.clamp(.55*(ref-c[..., :14])+.01*theta[:14]+.08*theta[14:]*(ref-x[..., :14]),-.03,.03)

def c_step(c,a,lo,hi):
    target=torch.maximum(torch.minimum(c[..., :14]+a,hi),lo)
    filt=.5*target+.5*c[...,14:28]
    return torch.cat([target,filt,c[...,42:56],filt,c[...,56:57]],-1)

def reward(x,a,ref,prev):
    # Deliberately reduced diagnostic reward, not the full paper's task reward.
    return -((x[..., :14]-ref)**2).mean(-1)-.0005*(x[...,14:]**2).mean(-1)-.05*((a-prev)**2).mean(-1)

class World(nn.Module):
    def __init__(self,kind,stats,width=128):
        super().__init__(); self.kind=kind
        dim={'mx':42,'md':43,'mc':99,'mh':379}[kind]
        # Match parameter budgets with widths selected analytically, logged below.
        width={'mx':160,'md':160,'mc':139,'mh':83}[kind]
        self.net=nn.Sequential(nn.Linear(dim,width),nn.SiLU(),nn.Linear(width,width),nn.SiLU(),nn.Linear(width,28))
        for name,val in stats.items(): self.register_buffer(name,torch.tensor(val,dtype=DTYPE))
    def features(self,x,c,a,h):
        xx=(x-self.xmean)/self.xstd; aa=a/.03
        if self.kind=='mx': return torch.cat([xx,aa],-1)
        if self.kind=='md': return torch.cat([xx,aa,c[...,56:57]/2],-1)
        if self.kind=='mc': return torch.cat([xx,aa,(c-self.cmean)/self.cstd],-1)
        hx=(h[...,:28]-self.xmean)/self.xstd
        ha=h[...,28:]/.03
        return torch.cat([xx,aa,torch.cat([hx,ha],-1).flatten(-2),c[...,56:57]/2],-1)
    def forward(self,x,c,a,h):
        return x+self.net(self.features(x,c,a,h))*self.ystd+self.ymean

def tensor(x): return torch.as_tensor(np.array(x),dtype=DTYPE)
def history_next(h,x,a): return torch.cat([h[...,1:,:],torch.cat([x,a],-1).unsqueeze(-2)],-2)

def imagined(model, theta, start, lo, hi, horizon=8):
    x=tensor(start['x']); c=tensor(start['c']); h=tensor(start['h']); refs=tensor(start['refs'][:horizon]); prev=tensor(start['prev'])
    total=theta.sum()*0
    for k in range(horizon):
        a=actor(theta,x,c,refs[k]); newx=model(x,c,a,h)
        total=total+(.99**k)*reward(newx,a,refs[k],prev)
        h=history_next(h,x,a); x=newx; c=c_step(c,a,lo,hi); prev=a
    return total

def physical(env, theta, start, horizon=8):
    env.restore(start['snapshot']); total=0.; prev=start['prev'].copy(); sat=0
    for k in range(horizon):
        x=env.observe(); c=env.control_state(); ref=start['refs'][k]
        raw=.55*(ref-c[:14])+.01*theta[:14]+.08*theta[14:]*(ref-x[:14])
        sat+=np.count_nonzero(np.abs(raw)>=.03)
        a=np.clip(raw,-.03,.03); env.step(a); newx=env.observe()
        total+=.99**k*(-np.mean((newx[:14]-ref)**2)-.0005*np.mean(newx[14:]**2)-.05*np.mean((a-prev)**2))
        prev=a
    return float(total),sat/(14*horizon)

def collect(env,args,out):
    all_ep=[]; starts={'val':[],'test':[],'train':[]}; replay_error=0.
    for delay in [0,1,2]:
        for ep in range(args.trajectories):
            env.delay=delay; env.reset(seed=ep)
            center=env.observe()[:14].copy()
            refs=ref_path(center,ep,args.steps+16,env.control_dt)
            split='train' if ep<args.train_count else ('val' if ep<args.train_count+args.val_count else 'test')
            h=np.zeros((8,42)); h[:,:28]=env.observe(); prev=np.zeros(14)
            rows={k:[] for k in ['x','c','a','y','h']}
            rng=np.random.default_rng(10000+ep+delay*100)
            for k in range(args.steps):
                x=env.observe(); c=env.control_state()
                if k in [args.steps//3,2*args.steps//3]:
                    starts[split].append(dict(x=x.copy(),c=c.copy(),h=h.copy(),prev=prev.copy(),refs=refs[k:k+16].copy(),snapshot=env.snapshot(),delay=delay,trajectory=ep,frame=k))
                a=np.clip(.55*(refs[k]-c[:14])+rng.normal(0,.003,14),-.03,.03)
                env.step(a); y=env.observe()
                if not np.isfinite(y).all() or np.max(np.abs(y[14:]))>100: raise RuntimeError('Unstable collection')
                for name,value in [('x',x),('c',c),('a',a),('y',y),('h',h.copy())]: rows[name].append(value.copy())
                h=np.concatenate([h[1:],np.concatenate([x,a])[None]],0); prev=a
            rows={k:np.array(v,dtype=np.float32) for k,v in rows.items()}
            rows.update(split=split,delay=delay,trajectory=ep); all_ep.append(rows)
        print('Collected delay',delay,flush=True)
    for s in starts['test'][:3]:
        r1=physical(env,np.zeros(28),s)[0]; r2=physical(env,np.zeros(28),s)[0]
        replay_error=max(replay_error,abs(r1-r2))
    if replay_error>1e-10: raise RuntimeError(f'Replay mismatch: {replay_error}')
    arrays={}
    for split in ['train','val','test']:
        eps=[e for e in all_ep if e['split']==split]
        for name in ['x','c','a','y','h']: arrays[split+'_'+name]=np.concatenate([e[name] for e in eps])
    np.savez_compressed(out/'transitions.npz',**arrays)
    torch.save(starts,out/'starts.pt') # Only load snapshots generated by this script.
    return all_ep,starts,replay_error

def fit(episodes,kind,seed,args,out):
    torch.manual_seed(seed); rng=np.random.default_rng(seed)
    train=[e for e in episodes if e['split']=='train']; val=[e for e in episodes if e['split']=='val']
    x=np.concatenate([e['x'] for e in train]); c=np.concatenate([e['c'] for e in train]); delta=np.concatenate([e['y']-e['x'] for e in train])
    stats=dict(xmean=x.mean(0),xstd=np.maximum(x.std(0),.01),cmean=c.mean(0),cstd=np.maximum(c.std(0),.01),ymean=delta.mean(0),ystd=np.maximum(delta.std(0),.001))
    m=World(kind,stats); opt=torch.optim.Adam(m.parameters(),lr=.001)
    stacked={k:tensor(np.stack([e[k] for e in train])) for k in ['x','c','a','y','h']}
    valid={k:tensor(np.concatenate([e[k] for e in val])) for k in ['x','c','a','y','h']}
    logs=[]; best=float('inf'); best_state=None
    K=3
    for it in range(args.updates):
        e=torch.tensor(rng.integers(len(train),size=128)); t=torch.tensor(rng.integers(args.steps-K,size=128))
        xx=stacked['x'][e,t]; hh=stacked['h'][e,t]; loss=0
        for k in range(K):
            cc=stacked['c'][e,t+k]; aa=stacked['a'][e,t+k]; yy=stacked['y'][e,t+k]
            yp=m(xx,cc,aa,hh); loss=loss+(((yp-yy)/m.ystd)**2).mean()/K
            hh=history_next(hh,xx,aa); xx=yp
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(),5); opt.step()
        if (it+1)%100==0 or it==args.updates-1:
            with torch.no_grad(): vl=(((m(valid['x'],valid['c'],valid['a'],valid['h'])-valid['y'])/m.ystd)**2).mean().item()
            logs.append(dict(update=it+1,train=float(loss.detach()),validation=vl))
            if vl<best: best=vl; best_state=copy.deepcopy(m.state_dict())
    m.load_state_dict(best_state); m.eval()
    for p in m.parameters(): p.requires_grad_(False)
    torch.save({'state':m.state_dict(),'kind':kind,'stats':stats,'seed':seed},out/f'{kind}_seed{seed}.pt')
    (out/f'{kind}_seed{seed}_learning.json').write_text(json.dumps(logs,indent=2))
    print('Trained',kind,seed,'val',round(best,5),flush=True)
    return m,{'kind':kind,'seed':seed,'validation_normalized_mse':best,'parameters':sum(p.numel() for p in m.parameters())}

def gradient(m,s,lo,hi,theta=None):
    t=torch.zeros(28,requires_grad=True) if theta is None else tensor(theta).requires_grad_(True)
    j=imagined(m,t,s,lo,hi); g=torch.autograd.grad(j,t)[0].detach().numpy()
    if not np.isfinite(g).all(): raise RuntimeError('Nonfinite model gradient')
    return g

def diagnostic(env,models,starts,seed,args,out,lo,hi):
    bank=[]; rng=np.random.default_rng(seed+555)
    # Use all delays in validation direction construction; no final test selection.
    vv=starts['val']; selected=[vv[i] for i in np.linspace(0,len(vv)-1,6,dtype=int)]
    for kind in ['mx','mc']:
        for s in selected:
            g=gradient(models[kind],s,lo,hi); norm=np.linalg.norm(g)
            if norm<1e-12: raise RuntimeError('Degenerate model proposal')
            bank.append((kind,g/norm))
    for _ in range(4):
        g=rng.normal(size=28);bank.append(('random',g/np.linalg.norm(g)))
    np.savez(out/f'directions_seed{seed}.npz',directions=np.array([d for _,d in bank]),sources=np.array([a for a,_ in bank]))
    eps=.05; scales=[.5,1.,2.]
    # Numerical, training/validation-only scale; no threshold chosen on test values.
    val_d=[]
    for s in selected:
        for _,d in bank[:2]:
            val_d.append((physical(env,eps*d,s)[0]-physical(env,-eps*d,s)[0])/(2*eps))
    s0=max(float(np.median(np.abs(val_d))),1e-8); tau=max(1e-9,s0*.01)
    records=[]
    for si,s in enumerate(starts['test']):
        grads={kind:gradient(m,s,lo,hi) for kind,m in models.items()}
        for di,(origin,d) in enumerate(bank):
            vals=[]; sat=[]
            for scale in scales:
                ep=eps*scale; plus,p=physical(env,ep*d,s); minus,n=physical(env,-ep*d,s)
                vals.append((plus-minus)/(2*ep));sat.extend([p,n])
            v=vals[1]; stable=(max(vals)-min(vals))<=max(tau,.1*abs(v))
            resolved=stable and abs(v)>tau
            for kind,g in grads.items():
                u=float(g@d); signmod=0 if abs(u)<=tau else np.sign(u)
                records.append(dict(seed=seed,delay=s['delay'],trajectory=s['trajectory'],frame=s['frame'],direction=di,source=origin,model=kind,model_derivative=u,sim_derivative=v,fd_half=vals[0],fd_double=vals[2],tau=tau,scale=s0,stable=int(stable),resolved=int(resolved),agree=int(signmod==np.sign(v)) if resolved else '',abs_error=abs(u-v),normalized_error=abs(u-v)/s0,saturation=max(sat)))
        if (si+1)%12==0:print('Diagnostic',seed,si+1,'/',len(starts['test']),flush=True)
    return records

def policy_smoke(env,model,starts,lo,hi,out):
    # A small un-gated model-based policy-optimization run; no success assumed.
    train=starts['train'][::max(1,len(starts['train'])//6)][:6]
    test=starts['test'][::max(1,len(starts['test'])//6)][:6]
    theta=torch.zeros(28,requires_grad=True); opt=torch.optim.Adam([theta],lr=.03); rows=[]
    for it in range(21):
        if it%5==0:
            with torch.no_grad(): predicted=float(torch.stack([imagined(model,theta,s,lo,hi) for s in train]).mean())
            real=np.mean([physical(env,theta.detach().numpy(),s)[0] for s in test])
            rows.append(dict(update=it,model_train_return=predicted,heldout_sim_return=float(real),theta_norm=float(theta.detach().norm())))
        if it<20:
            loss=-torch.stack([imagined(model,theta,s,lo,hi) for s in train]).mean()
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_([theta],1);opt.step()
            with torch.no_grad():theta.clamp_(-1,1)
    np.save(out/'trained_actor_theta.npy',theta.detach().numpy())
    (out/'policy_training.json').write_text(json.dumps(rows,indent=2))
    return rows

def main():
    p=argparse.ArgumentParser();p.add_argument('--run-name',default='pilot_20261101');p.add_argument('--updates',type=int,default=1000);p.add_argument('--seeds',type=int,default=3);p.add_argument('--trajectories',type=int,default=24);p.add_argument('--steps',type=int,default=300);p.add_argument('--train-count',type=int,default=12);p.add_argument('--val-count',type=int,default=6)
    args=p.parse_args(); out=ROOT/'runs'/args.run_name
    if out.exists():raise SystemExit('Run directory already exists; use a fresh --run-name')
    out.mkdir(parents=True); began=time.time();env=ArmEnv()
    (out/'config.json').write_text(json.dumps(vars(args),indent=2));meta=env.metadata()
    meta.update(python=sys.version,torch=torch.__version__,numpy=np.__version__,platform=platform.platform(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out/'environment.json').write_text(json.dumps(meta,indent=2,default=str))
    (out/'effective_model.xml').write_text(env.effective_xml,encoding='utf-8')
    env.accounting_phase='collection_and_replay'
    episodes,starts,replay=collect(env,args,out)
    lo=tensor(env.lower);hi=tensor(env.upper)
    # Check torch mirror against the executed controller for all delay branches.
    c_error=0.
    env.accounting_phase='controller_mirror_audit'
    for s in starts['val']:
        env.restore(s['snapshot']);a=np.ones(14)*.005
        pred=c_step(tensor(env.control_state()),tensor(a),lo,hi).detach().numpy();env.step(a)
        c_error=max(c_error,float(np.max(np.abs(pred-env.control_state()))))
    if c_error>1e-6:raise RuntimeError(f'Controller mirror mismatch {c_error}')
    records=[];training=[];policy=None
    for seed in range(args.seeds):
        models={}
        for kind in ['mx','md','mc','mh']:
            models[kind],stat=fit(episodes,kind,seed,args,out);training.append(stat)
        env.accounting_phase='direction_diagnostic_with_validation'
        records.extend(diagnostic(env,models,starts,seed,args,out,lo,hi))
        if seed==0:
            env.accounting_phase='policy_smoke_eval'
            policy=policy_smoke(env,models['mc'],starts,lo,hi,out)
        with (out/'direction_probes.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    summary=[]
    for seed in range(args.seeds):
        for delay in range(3):
            for kind in ['mx','md','mc','mh']:
                rr=[r for r in records if r['seed']==seed and r['delay']==delay and r['model']==kind];res=[r for r in rr if r['resolved']]
                summary.append(dict(seed=seed,delay=delay,model=kind,probes=len(rr),resolved=len(res),agreement=float(np.mean([r['agree'] for r in res])) if res else None,directional_error=float(np.mean([r['normalized_error'] for r in res])) if res else None,raw_abs_error=float(np.mean([r['abs_error'] for r in rr])),saturation=max(r['saturation'] for r in rr)))
    report=dict(summary=summary,training=training,policy_training=policy,replay_max_abs_error=replay,controller_mirror_max_abs_error=c_error,total_control_steps=env.total_control_steps,control_steps_by_phase=env.control_steps_by_phase,total_physics_steps=env.total_control_steps*env.substeps,wall_seconds=time.time()-began,status='completed',warning='Preliminary descriptive experiment; no statistical significance; no hardware; no R5 training benefit claim')
    (out/'results.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
