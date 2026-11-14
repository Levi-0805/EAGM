"""CPU numerical validation only; no research training or seed reuse."""
import itertools
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from privileged_execution import PrivilegedExecution
from controllers.partial_unknown import PartialUnknownArmEnv,ControllerBConfig

METRICS={}


class PrivilegedExecutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.patch=patch('controllers.compatible_arm_env.materialize_pilot_effective_xml',return_value=ROOT/'assets/pilot_effective_model.linux.xml')
        cls.patch.start()
    @classmethod
    def tearDownClass(cls): cls.patch.stop()

    def compare_state(self,state,env):
        for name, actual in [('target',env.qtarget),('filtered',env.filtered),('queue',env.queue),
                             ('u_prev',env._u_prev),('u_smooth',env._u_smooth)]:
            np.testing.assert_allclose(getattr(state,name).detach().numpy(),actual,atol=2e-12,rtol=0)
        self.assertEqual(len(state.extra_queue),len(env._extra_queue))
        for pred,true in zip(state.extra_queue,env._extra_queue):
            np.testing.assert_allclose(pred.detach().numpy(),true,atol=2e-12,rtol=0)

    def test_cartesian_true_controller_and_midtrajectory_hidden_restore(self):
        rng=np.random.default_rng(880301); count=0; maximum=0.;configs=0
        for alpha,lag,gain,rate,beta in itertools.product((.2,.5,1.),(0,1,2),(.9,1.1),(.01,.08),(0.,.25,.8)):
            env=PartialUnknownArmEnv(ControllerBConfig(gain=gain,extra_lag=lag,rate_limit=rate,smooth_beta=beta),alpha=alpha)
            env.reset(seed=880301,delay=0,jitter=0)
            model,state=PrivilegedExecution.from_privileged_snapshot(env.snapshot(),env.lower,env.upper)
            for t in range(12):
                env.delay=t%3; action=rng.uniform(-.06,.06,14)
                if t==6:
                    model,state=PrivilegedExecution.from_privileged_snapshot(env.snapshot(),env.lower,env.upper)
                state,pred=model.step(state,torch.tensor(action),delay=env.delay)
                env.step(action)
                for key,true in [('nominal',env.last_nominal_command),('executed',env.last_true_command)]:
                    maximum=max(maximum,float(np.max(np.abs(pred[key].detach().numpy()-true))))
                    np.testing.assert_allclose(pred[key].detach().numpy(),true,atol=2e-12,rtol=0)
                self.compare_state(state,env);count+=1
            configs+=1
        METRICS['parity']=dict(configurations=configs,actual_control_steps=count,max_abs_error=maximum,
                               delays=[0,1,2],alphas=[.2,.5,1.],midtrajectory_reinitialization=True)

    def test_multistep_autograd_against_real_controller_fd(self):
        maximum=0.;probes=0;nonzero=0
        for alpha,d,lag,beta in itertools.product((.2,.5,1.),(0,1,2),(0,1,2),(0.,.25)):
            env=PartialUnknownArmEnv(ControllerBConfig(gain=1.,extra_lag=lag,rate_limit=.08,smooth_beta=beta),alpha=alpha)
            env.reset(seed=880302,delay=d,jitter=0)
            snapshot=env.snapshot()
            model,state=PrivilegedExecution.from_privileged_snapshot(snapshot,env.lower,env.upper)
            action=torch.full((14,),.001,dtype=torch.float64,requires_grad=True)
            # Only the first action varies. Horizon guarantees delayed influence reaches output.
            horizon=d+lag+3
            def predicted(first):
                current=state
                for k in range(horizon):
                    current,out=model.step(current,first if k==0 else torch.zeros_like(first))
                return out['executed']
            jac=torch.autograd.functional.jacobian(predicted,action).numpy()
            def true(first):
                env.restore(snapshot)
                for k in range(horizon):env.step(first if k==0 else np.zeros(14))
                return env.last_true_command.copy()
            eps=1e-6;eye=np.eye(14);a=action.detach().numpy()
            fd=np.stack([(true(a+eps*v)-true(a-eps*v))/(2*eps) for v in eye],axis=-1)
            maximum=max(maximum,float(np.max(np.abs(jac-fd))))
            np.testing.assert_allclose(jac,fd,atol=2e-9,rtol=2e-8)
            nonzero+=int(np.linalg.norm(jac)>0);probes+=1
        METRICS['gradient']=dict(probes=probes,nonzero_jacobians=nonzero,max_abs_error=maximum,
                                 fd_radius=1e-6,first_action_to_final_executed_command=True)

    def test_privileged_requirement_and_input_immutability(self):
        with self.assertRaises(ValueError): PrivilegedExecution.from_privileged_snapshot({},[-1]*14,[1]*14)
        env=PartialUnknownArmEnv();env.reset(seed=880303,delay=0,jitter=0)
        snapshot=env.snapshot();model,state=PrivilegedExecution.from_privileged_snapshot(snapshot,env.lower,env.upper)
        before=state.target.clone();model.step(state,torch.full((14,),.001,dtype=torch.float64))
        torch.testing.assert_close(state.target,before)
        np.testing.assert_array_equal(snapshot['qtarget'],before.numpy())


if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(PrivilegedExecutionTests))
    METRICS['passed']=result.wasSuccessful()
    (Path(__file__).parent/'privileged_execution_validation.json').write_text(json.dumps(METRICS,indent=2)+'\n')
    sys.exit(0 if result.wasSuccessful() else 1)
