import sys
sys.path[:0]=['.','vendor/legacy_sim']
import torch,numpy as np
from unittest.mock import patch
from scripts.run_stage7c1_diagnosis import make_env
from training.rollout import physical_return_true_ee,physical_return_surrogate,set_reach_fk_surrogate,physical_return_detailed
from tasks.reach import FKSurrogate
p='runs/stage7c2_true_ee_current_sites_seed20/'
d=torch.load(p+'data.pt',weights_only=False);st=d['test_starts'][0];env=make_env();th=np.zeros(28)
fk=FKSurrogate();fk.load_state_dict(torch.load(p+'fk.pt',weights_only=True));set_reach_fk_surrogate(fk)
legacy=physical_return_detailed(env,th,st,horizon=4)
s=physical_return_surrogate(env,th,st,horizon=4)[0]
with patch('training.rollout._REACH_FK',None):
 t=physical_return_true_ee(env,th,st,horizon=4)[0]
 t2=physical_return_true_ee(env,th,st,horizon=4)[0]
assert t==t2
assert np.isfinite(t)  # refreshed current-state FK need not equal legacy cached sites
assert abs(s-legacy['return_'])<1e-12
print('PASS: MuJoCo replay, legacy compatibility, true objective independent of FK',t,s)
# Ensure a deliberately stale cache cannot affect the new true reward.
from tasks.reach import reach_reward_true_mujoco, reach_reward_numpy, ee_from_q_mujoco
x=env.observe();a=np.zeros(14);prev=np.zeros(14)
left,right=ee_from_q_mujoco(env,x[:14])
expected=reach_reward_numpy(left,right,st['ee_goals'],x,a,prev)
env.data.site_xpos[:]=999.
q_before=env.data.qpos.copy();warm_before=env.data.qacc_warmstart.copy()
actual=reach_reward_true_mujoco(env,x,a,prev,st['ee_goals'])
assert abs(actual-expected)<1e-12
assert np.array_equal(q_before,env.data.qpos)
assert np.array_equal(warm_before,env.data.qacc_warmstart)
print('PASS: stale-site immunity and live warm-start preservation')
