"""Engineering task feasibility, not development-model seed performance."""
import sys,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np,torch
from eagm2.contact import ContactEnv,TorchContactReward
from controllers.partial_unknown import ControllerBConfig
from PIL import Image
out=ROOT/'reports/eagm2_dev'
records=[]
configs=[('identity_d0',ControllerBConfig(),0),('B_d0',ControllerBConfig(gain=1.05,extra_lag=1,rate_limit=.045,smooth_beta=.25),0),('B_d1',ControllerBConfig(gain=1.05,extra_lag=1,rate_limit=.045,smooth_beta=.25),1),('B_d2',ControllerBConfig(gain=1.05,extra_lag=1,rate_limit=.045,smooth_beta=.25),2)]
for name,cfg,delay in configs:
 e=ContactEnv(cfg,delay=delay);e.reset(seed=0,delay=delay,jitter=0);start=time.perf_counter();rows=[];prev=np.zeros(14)
 for ref in e.reference(int(np.floor(10/e.control_dt))):
  x=e.observe();a=np.clip(.55*(ref-e.qtarget)+.1*(ref-x[:14]),-.03,.03);e.step(a)
  rows.append(dict(frame=e.frame,reward=e.reward(a,prev),**e.metrics()));prev=a
 record={'condition':name,'seed':0,'role':'engineering_scripted_teacher_only_not_learned_policy','metadata':e.metadata(),'metrics':e.metrics(),'wall_seconds':time.perf_counter()-start,'control_steps':len(rows),'physics_steps':len(rows)*e.substeps,'per_step':rows}
 records.append(record);print(name,e.metrics(),flush=True)
 if name=='B_d2':
  before=e.snapshot();rgb=e.render_rgb();np.testing.assert_array_equal(before['integration'],e.snapshot()['integration']);Image.fromarray(rgb).save(out/'tasks_contact_inserted.png')
 e.close()
(out/'tasks_feasibility.json').write_text(json.dumps({'status':'engineering_only','policy':'clip(.55*(ref-qtarget)+.1*(ref-q),-.03,.03)','results':records},indent=2)+'\n')
