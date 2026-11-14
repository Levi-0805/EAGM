import itertools
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from eagm2.prior import nominal_command, observable_step, previous_filter_from_invariant
from controllers.partial_unknown import PartialUnknownArmEnv, ControllerBConfig
from controllers.compatible_arm_env import materialize_pilot_effective_xml


class PriorControllerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory()
        cls.xml=materialize_pilot_effective_xml(Path(cls.temp.name)/'pilot.xml')
        cls.patch=patch('controllers.compatible_arm_env.materialize_pilot_effective_xml',return_value=cls.xml)
        cls.patch.start()
    @classmethod
    def tearDownClass(cls):
        cls.patch.stop();cls.temp.cleanup()

    def test_real_controller_full_cartesian(self):
        """True MuJoCo/controller steps, independent scalar B stack reference."""
        rng=np.random.default_rng(519)
        count=0;max_nominal=0.;max_exec=0.
        for alpha,gain,lag,rate,beta in itertools.product((.2,.5,1.),(.9,1.1),(0,1,2),(.01,.08),(0.,.25,.8)):
            env=PartialUnknownArmEnv(ControllerBConfig(gain=gain,extra_lag=lag,rate_limit=rate,smooth_beta=beta),alpha=alpha)
            env.reset(seed=519,jitter=0)
            f_prev=env.filtered.copy();prev=env.qtarget.copy();smooth=prev.copy();queue=[]
            for t in range(9):
                # Switching does not reset the queue; all d values exercised.
                env.delay=t%3
                c=env.observed_control_state().copy()
                a=rng.uniform(-.06,.06,14)
                u_nom=nominal_command(torch.tensor(c),torch.tensor(a),torch.tensor(f_prev),env.lower,env.upper,alpha).numpy()
                c_next,new_prev=observable_step(torch.tensor(c),torch.tensor(a),env.lower,env.upper,alpha)
                gained=gain*u_nom
                if lag:
                    queue.append(gained.copy())
                    delayed=prev.copy() if len(queue)<=lag else queue.pop(0)
                else:delayed=gained
                limited=prev+np.clip(delayed-prev,-rate,rate)
                if beta>0:smooth=beta*limited+(1-beta)*smooth;out=smooth.copy()
                else:out=limited
                out=np.clip(out,env.lower,env.upper);prev=out.copy()
                env.step(a)
                max_nominal=max(max_nominal,float(np.max(np.abs(u_nom-env.last_nominal_command))))
                max_exec=max(max_exec,float(np.max(np.abs(out-env.last_true_command))))
                np.testing.assert_allclose(u_nom,env.last_nominal_command,atol=2e-12,rtol=0)
                np.testing.assert_allclose(out,env.last_true_command,atol=2e-12,rtol=0)
                np.testing.assert_allclose(c_next.numpy(),env.observed_control_state(),atol=2e-12,rtol=0)
                np.testing.assert_allclose(env.data.ctrl[env.actuator_ids],out,atol=2e-12,rtol=0)
                f_prev=new_prev.numpy().copy();count+=1
        print(f'controller matrix: {count} actual control steps; nominal max={max_nominal:.3g}, executed-stack max={max_exec:.3g}')

    def test_joint_boundary_reset_and_inverse(self):
        for alpha in (.2,.5,1.):
            env=PartialUnknownArmEnv(alpha=alpha)
            env.reset(seed=520,pose=env.upper-.0001,jitter=0)
            old=env.filtered.copy()
            for d in (0,1,2):
                env.delay=d;c=torch.tensor(env.observed_control_state());a=torch.full((14,),.2,dtype=torch.float64)
                pred=nominal_command(c,a,torch.tensor(old),env.lower,env.upper,alpha)
                env.step(a.numpy());np.testing.assert_allclose(pred.numpy(),env.last_nominal_command,atol=2e-12)
                old=c[14:28].numpy()
                if alpha!=1:
                    inv=previous_filter_from_invariant(torch.tensor(env.observed_control_state()),alpha)
                    np.testing.assert_allclose(inv.numpy(),old,atol=2e-12)
        with self.assertRaises(ValueError):previous_filter_from_invariant(torch.zeros(29),1.)

    def test_interior_jacobian_and_invalid_delay(self):
        for d in (0,1,2):
            c=torch.zeros(29,dtype=torch.float64);c[28]=d;a=torch.zeros(14,dtype=torch.float64,requires_grad=True)
            fun=lambda a:nominal_command(c,a,torch.zeros(14,dtype=torch.float64),[-2.]*14,[2.]*14,.5)
            jac=torch.autograd.functional.jacobian(fun,a)
            torch.testing.assert_close(jac,(.5 if d==0 else 0)*torch.eye(14,dtype=torch.float64))
            eps=1e-5
            fd=torch.stack([(fun(a+eps*torch.eye(14,dtype=torch.float64)[j])-fun(a-eps*torch.eye(14,dtype=torch.float64)[j]))/(2*eps) for j in range(14)],-1)
            torch.testing.assert_close(jac,fd)
        c[28]=3
        with self.assertRaises(ValueError):fun(a)

if __name__=='__main__':unittest.main(verbosity=2)
