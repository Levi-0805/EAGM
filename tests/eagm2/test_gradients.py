import unittest
import numpy as np
import torch
from eagm2.gradients import finite_difference, calibration_loss, gradient_metrics, spsa_gradient

class FiniteDifferenceTests(unittest.TestCase):
    def test_quadratic_all_stencils_exact_and_accounted(self):
        calls=[]
        def objective(x):
            calls.append(x.copy())
            return float(np.sum(x*x)+100)
        theta=np.array([0.,.98,1.,-1.])
        result=finite_difference(objective,theta,epsilon=.05)
        np.testing.assert_allclose(result['gradient'],2*theta,atol=2e-10)
        self.assertTrue(result['mask'].all())
        self.assertEqual(result['queries'],len(calls))
        self.assertEqual(result['queries'],len({x.tobytes() for x in calls}))
        self.assertTrue(all(np.max(np.abs(x))<=1 for x in calls))
        modes={d['mode'] for c in result['details'] for d in c}
        self.assertEqual(modes,{'symmetric_central','nonuniform_three_point','forward_three_point','backward_three_point'})
        np.testing.assert_array_equal(theta,[0,.98,1,-1])

    def test_exp_converges_and_position_is_not_clipped(self):
        f=lambda x:float(np.exp(x).sum())
        t=np.array([.98,1.,-1.])
        coarse=finite_difference(f,t,epsilon=.05)
        fine=finite_difference(f,t,epsilon=.005)
        self.assertLess(np.linalg.norm(fine['gradient']-np.exp(t)),np.linalg.norm(coarse['gradient']-np.exp(t))/20)
        with self.assertRaises(ValueError):finite_difference(f,[1.01])

    def test_stability_and_nonfinite_are_not_success(self):
        single=finite_difference(lambda x:x[0]**2,[.3],scales=(1,))
        self.assertEqual(single['mask'][0],0)
        bad=finite_difference(lambda x:float('nan'),[.3])
        self.assertEqual(bad['mask'][0],0)
        self.assertFalse(bad['stability_is_confidence_interval'])
        unstable=finite_difference(lambda x:float(np.sin(100*x[0])),[.1],epsilon=.05)
        self.assertEqual(unstable['mask'][0],0)

    def test_cache_accounting_and_scalar_bounds(self):
        f=lambda x:float(x@x)
        a=finite_difference(f,[1.,1.],cache=True)
        b=finite_difference(f,[1.,1.],cache=False)
        self.assertLess(a['queries'],b['queries'])
        np.testing.assert_allclose(a['gradient'],b['gradient'])
        with self.assertRaises(ValueError):finite_difference(f,[0],bounds=(0,0))

    def test_spsa_reproducible_and_query_count(self):
        f=lambda x:float(np.sum(x*x))
        a=spsa_gradient(f,[.2],epsilon=.01,directions=3,seed=42)
        np.testing.assert_allclose(a['gradient'],[.4],atol=1e-12)
        self.assertEqual(a['queries'],6)
        b=spsa_gradient(f,[.2],epsilon=.01,directions=3,seed=42)
        np.testing.assert_array_equal(a['estimates'],b['estimates'])
        with self.assertRaises(ValueError):spsa_gradient(f,[1.],bounds=(-1,1))

class CalibrationTests(unittest.TestCase):
    def test_legacy_equals_historical_formula(self):
        p=torch.tensor([.2,-.3,4.],dtype=torch.float64,requires_grad=True)
        t=torch.tensor([.1,-.1,8.],dtype=torch.float64)
        m=torch.tensor([1.,1.,0.],dtype=torch.float64)
        got=calibration_loss(p,t,m,mode='legacy',scale_floor=.01)
        error=(p[:2]-t[:2])/t[:2].abs()
        expected=torch.nn.functional.huber_loss(error,torch.zeros_like(error))
        torch.testing.assert_close(got,expected)

    def test_direction_scale_invariance_norm_detects_scale(self):
        t=torch.tensor([1.,-2.],dtype=torch.float64)
        p=2*t
        m=torch.ones_like(t)
        self.assertLess(abs(float(calibration_loss(p,t,m,mode='direction'))),1e-12)
        self.assertGreater(float(calibration_loss(p,t,m,mode='norm')),0.1)

    def test_higher_order_model_gradients_all_modes(self):
        for mode in ('legacy','direction','norm','joint'):
            weight=torch.tensor([.7,1.2],dtype=torch.float64,requires_grad=True)
            theta=torch.tensor([.2,-.4],dtype=torch.float64,requires_grad=True)
            objective=(weight*theta.square()).sum()
            model_grad=torch.autograd.grad(objective,theta,create_graph=True)[0]
            loss=calibration_loss(model_grad,torch.tensor([.2,-.5],dtype=torch.float64),torch.ones(2,dtype=torch.float64),mode=mode)
            gradient=torch.autograd.grad(loss,weight)[0]
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(float(gradient.norm()),0)

    def test_empty_mask_and_target_detached(self):
        p=torch.tensor([1.,2.],requires_grad=True)
        t=torch.tensor([1.,-1.],requires_grad=True)
        loss=calibration_loss(p,t,torch.zeros(2),mode='joint')
        torch.testing.assert_close(torch.autograd.grad(loss,p)[0],torch.zeros(2))
        loss=calibration_loss(p,t,torch.ones(2),mode='joint')
        self.assertIsNone(torch.autograd.grad(loss,t,allow_unused=True)[0])

class MetricsTests(unittest.TestCase):
    def test_class_balance_differs_from_accuracy(self):
        q=np.r_[np.ones(9),-1.]
        p=np.ones(10)
        result=gradient_metrics(p,q)
        self.assertAlmostEqual(result['sign_accuracy'],.9)
        self.assertAlmostEqual(result['balanced_sign_accuracy'],.5)

    def test_undefined_metrics_are_null(self):
        for p,q,m in [([0,0],[0,0],None),([1,2],[1,2],[0,0])]:
            r=gradient_metrics(p,q,m)
            self.assertIsNone(r['cosine'])
            self.assertIsNone(r['balanced_sign_accuracy'])
        self.assertIsNone(gradient_metrics([1,1],[1,1])['balanced_sign_accuracy'])
        r=gradient_metrics([float('nan'),1],[1,-1])
        self.assertEqual(r['n_nonfinite_selected'],1)
        self.assertEqual(r['finite_selected_coverage'],.5)

if __name__=='__main__':unittest.main()
