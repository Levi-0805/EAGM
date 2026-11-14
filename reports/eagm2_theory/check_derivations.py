"""Independent arithmetic checks, not empirical validation of theorem assumptions."""
import json
import numpy as np
from pathlib import Path
checks=[]
for a,b in ((.1,.2),(.01,.8),(.2,.2)):
 w=np.array([-b/(a*(a+b)),(b-a)/(a*b),a/(b*(a+b))]);x=np.array([-a,0,b])
 assert np.allclose([w@x**k for k in range(3)],[0,1,0])
 assert np.isclose(abs(w@x**3),a*b)
 checks.append(dict(test='nonuniform_cubic_bound_tight',a=a,b=b))
for h in (.01,.1,.5):
 w=np.array([-3,4,-1])/(2*h);x=np.array([0,h,2*h])
 assert np.isclose(abs(w@x**3),2*h*h)
 checks.append(dict(test='one_sided_cubic_bound_tight',h=h))
# Scalar same-coordinate linear dynamics, bilinear policy, quadratic reward.
z,zh=.5,.52;s,sh=0.,0.;theta=.3
L=.82+.21*.3;B=.21;e=.02;E=0.;s_bound=0.;bound=0.;g=gh=0.
for t in range(10):
 assert abs(zh-z)<=e+1e-12 and abs(sh-s)<=E+1e-12
 bound+=.99**t*(E+1.3*e*s_bound);g+=.99**t*z*s;gh+=.99**t*zh*sh
 znew=(.8+.2*theta)*z;snew=(.8+.2*theta)*s+.2*z
 zhnew=(.82+.21*theta)*zh+.01;shnew=(.82+.21*theta)*sh+.21*zh
 E=L*E+.023*s_bound+.01+.21*e;e=L*e+.04;s_bound=L*s_bound+B
 z,zh,s,sh=znew,zhnew,snew,shnew
assert abs(gh-g)<=bound
checks.append(dict(test='closed_loop_initial_error_recurrence',gradient_error=abs(gh-g),bound=bound))
Path(__file__).with_name('derivation_checks.json').write_text(json.dumps(dict(status='passed',checks=checks,note='Arithmetic consistency only; does not verify simulator smoothness or learned latent sufficiency.'),indent=2)+'\n')
print('7 independent arithmetic checks passed')
