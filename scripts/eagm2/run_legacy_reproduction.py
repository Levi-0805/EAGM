"""Run unmodified Base/Dir algorithms under original Stage5 protocol on new DEV seeds."""
import argparse,copy,time,torch
from unittest.mock import patch
from eagm2.runtime import *
from training.stage5_common import prepare_split_data,make_theta_bank,build_matched_labels,pack_rows,eval_suite
from training.dataset import pick_starts
from models.eagm import EAGMBase
from training.paired_train import train_pair

def main():
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--attempt',default='legacy_reproduction_v1');a=p.parse_args()
 out=new_run('tracking',a.seed,a.attempt);seed_all(a.seed);dump(out/'environment.json',environment());begin=time.perf_counter()
 try:
  with patch('controllers.compatible_arm_env.materialize_pilot_effective_xml',return_value=ROOT/'assets/pilot_effective_model.linux.xml'):
   data=prepare_split_data(a.seed)
  env=data['env'];tr=pack_rows(data['train_rows']);te=pack_rows(data['test_rows'])
  train=pick_starts(data['train_rows']);test=pick_starts(data['test_rows'])
  if not train or not test:raise RuntimeError("Original protocol produces empty delay=2 start pool; preserve failure and use separate stratified evaluation protocol")
  labels,steps=build_matched_labels(env,train,make_theta_bank(a.seed));labels_test,test_steps=build_matched_labels(env,test,[np.zeros(28)])
  base=EAGMBase(data['stats']);direction=copy.deepcopy(base)
  log=train_pair(base,direction,tr,updates=3000,use_history=False,dir_labels=labels,batch_seed=a.seed,device=torch.device('cpu'))
  results={}
  for name,model in [('legacy_base',base),('legacy_dir',direction)]:
   torch.save({'state_dict':model.state_dict(),'seed':a.seed,'stats':data['stats']},out/'checkpoints'/f'{name}.pt')
   results[name]=eval_suite(model,te,test,labels_test,train,env,use_history=False,is_mc=False,pol_seed=a.seed)
  torch.save({'labels_train':labels,'labels_diagnostic':labels_test,'split':data['split'],'stats':data['stats']},out/'labels_and_split.pt')
  dump(out/'training_logs/train.json',log)
  dump(out/'config.json',dict(phase='exploratory_legacy_reproduction',seed=a.seed,updates=3000,policy_updates=50,imagined_horizon=8,evaluation_horizon=16,original_protocol='configs/stage5_frozen.json',note='new development seed; not historical confirmation; original code unchanged'))
  dump(out/'results.json',dict(status='completed',phase='exploratory',results=results,wall_seconds=time.perf_counter()-begin,collection_control_steps=data['steps_main'],label_control_steps=steps,diagnostic_label_control_steps=test_steps))
 except Exception:
  dump(out/'failure.json',dict(traceback=traceback.format_exc(),wall_seconds=time.perf_counter()-begin));raise
if __name__=='__main__':main()
