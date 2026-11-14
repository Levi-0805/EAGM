import ast, numpy as np
from pathlib import Path
p=Path(__file__).resolve().parents[2] / 'controllers/partial_unknown.py'
tree=ast.parse(p.read_text()); cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='PartialUnknownArmEnv'); fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_apply_unknown'); ns={'np':np};exec(compile(ast.Module(body=[fn],type_ignores=[]),str(p),'exec'),ns)
class Stub: pass
for prev in [0.0,0.1]:
 s=Stub();s._active={'gain':1.,'extra_lag':0,'rate_limit':.03,'smooth_beta':0.};s._u_prev=np.full(14,prev);s._u_smooth=s._u_prev.copy();s.lower=np.full(14,-2.);s.upper=np.full(14,2.);s._extra_queue=[]
 out=ns['_apply_unknown'](s,np.full(14,.2))[0]
 print('extracted actual _apply_unknown: nominal=.2; u_prev=',prev,'output=',out)
 assert abs(out-(prev+.03))<1e-12
r=.2; f=.15; a=.03; alpha=.5; old=(f-alpha*r)/(1-alpha); new=alpha*(r+a)+(1-alpha)*f
print('algebraic FIFO check; pre-step r,f,queue=',r,f,[old,f]);print('U0=',new,'nominals d0,d1,d2=',[new,f,old]);print('interior action Jacobian U0=.5, nominal d1/d2=0')
print('Synthetic function/algebra checks only: no simulator trajectory or training results.')
