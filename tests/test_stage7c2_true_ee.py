import numpy as np
from unittest.mock import patch
from training.gradients import sim_grad_true_ee_masked_detailed

def test_true_fd_independent_of_surrogate_and_budget():
    calls=[]
    def objective(env,th,start,horizon):
        calls.append(np.asarray(th).copy())
        return float(np.sum(th)),{}
    with patch('training.rollout.physical_return_true_ee',objective), patch('training.rollout.physical_return',side_effect=AssertionError('legacy reward used')):
        for t in [np.zeros(28),np.ones(28),-np.ones(28)]:
            g,m,n,meta=sim_grad_true_ee_masked_detailed(None,{},t,horizon=8)
            assert np.allclose(g,1) and m.all()
            assert n==28*3*2*8 and meta['simulator_queries']==168
            assert len(meta['effective_epsilon'])==28
    assert all(np.max(abs(c))<=1 for c in calls)

if __name__=='__main__':
    test_true_fd_independent_of_surrogate_and_budget()
    print('PASS')
