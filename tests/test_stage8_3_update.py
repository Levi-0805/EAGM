import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,torch
from analysis.stage8_3_run import update
rng=np.random.default_rng(83)
g=rng.normal(size=28)*1e-4;length=.03*np.sqrt(28)
x=torch.zeros(28,dtype=torch.double,requires_grad=True);opt=torch.optim.Adam([x],lr=.03);x.grad=-torch.tensor(g);opt.step();expected=x.detach().numpy();expected*=length/np.linalg.norm(expected)
assert np.allclose(update(g,'adam_first',length),expected,atol=1e-12)
assert np.isclose(np.linalg.norm(update(g,'raw',length)),length)
assert np.dot(update(g,'raw',length),g)>0
print('PASS: raw/Adam equal L2, first Adam agrees with torch; zero physical steps')
