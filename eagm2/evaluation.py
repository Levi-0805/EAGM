import numpy as np,torch
from eagm2.prior import observable_step
from eagm2.rollout import gradient,physical
from eagm2.gradients import gradient_metrics

def model_metrics(model,data,horizon=8):
 idx=torch.linspace(0,len(data['x'])-1,min(256,len(data['x'])),device=data['x'].device).round().long().unique()
 b={k:v[idx] for k,v in data.items()}
 with torch.no_grad():
  h=model.burnin(b['xhist'],b['chist'],b['ahist']) if model.memory!='none' else None
  x=b['x'][:,0];c=b['c'][:,0];fp=b['fprev'][:,0];errs=[];maes=[]
  for k in range(horizon):
   x,u,h=model.step(x,c,b['a'][:,k],fp,h)
   errs.append(float((((x-b['y'][:,k])/model.ystd)**2).mean()));maes.append(float((u-b['u'][:,k]).abs().mean()))
   c,fp=observable_step(c,b['a'][:,k],model.lower,model.upper,float(model.alpha))
 return dict(one_step_nmse=errs[0],multi_step_nmse=errs,command_mae=maes[0],multi_step_command_mae=maes,n_sequences=len(idx))
def evaluate_gradients(model,labels,horizon,**kwargs):
 result=[]
 for lab in labels:
  g=gradient(model,lab['start'],lab['theta'],horizon,**kwargs)
  result.append(dict(episode_id=lab['start']['episode_id'],delay=lab['start']['delay'],theta=lab['theta'],g_model=g,g_reference=lab['gradient'],mask=lab['mask'],metrics=gradient_metrics(g,lab['gradient']),stable_mask_metrics=gradient_metrics(g,lab['gradient'],lab['mask'])))
 cos=[r['metrics'].get('cosine') for r in result];cos=[v for v in cos if v is not None]
 return dict(per_probe=result,mean_cosine=float(np.mean(cos)) if cos else None,primary_metric='full_coordinate_cosine; stable mask metrics separate')
def evaluate_returns(env,theta,starts,horizon,budget,task,kind='evaluation',vision_policy=None):
 rows=[dict(episode_id=s['episode_id'],delay=s['delay'],**physical(env,theta,s,horizon,budget,kind,task=task,vision_policy=vision_policy,return_details=True)) for s in starts]
 vals=np.array([r['return_'] for r in rows]);d=dict(per_episode=rows,mean_return=float(vals.mean()),std_return=float(vals.std(ddof=1)) if len(vals)>1 else 0.)
 if task!='tracking':
  d.update(success_rate=float(np.mean([r['success'] for r in rows])),completion_time_capped=float(np.mean([r['completion_time'] for r in rows])),position_error=float(np.mean([r['position_error'] for r in rows])),contact_fraction=float(np.mean([r['contact_fraction'] for r in rows])))
 return d
