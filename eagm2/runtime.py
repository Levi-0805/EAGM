"""Seed authorization, immutable outputs and measured resource accounting."""
from __future__ import annotations
import contextlib,hashlib,json,os,platform,random,subprocess,time,traceback
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
AUTH=json.loads((ROOT/'configs/eagm2/development_authorization.json').read_text())
def check_seed(task,seed):
    if seed not in AUTH['authorized_root_seeds'].get(task,[]):
        raise PermissionError(f'{task} seed {seed} is not authorized for development; confirmation locked')
def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
def stream_seed(root,name):
    return int.from_bytes(hashlib.sha256(f'eagm2:{root}:{name}'.encode()).digest()[:4],'little')
def clean(v):
    if isinstance(v,dict):return {str(k):clean(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [clean(x) for x in v]
    if isinstance(v,np.ndarray):return clean(v.tolist())
    if isinstance(v,np.generic):return clean(v.item())
    if isinstance(v,torch.Tensor):return clean(v.detach().cpu().numpy())
    if isinstance(v,float) and not np.isfinite(v):return None
    if isinstance(v,Path):return str(v)
    return v
def dump(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(clean(obj),indent=2,ensure_ascii=False,allow_nan=False)+'\n');tmp.replace(path)
def new_run(task,seed,attempt):
    check_seed(task,seed)
    p=ROOT/'runs/eagm2_development'/task/f'seed{seed}'/attempt
    p.mkdir(parents=True,exist_ok=False)
    for n in ('training_logs','checkpoints'): (p/n).mkdir()
    import tarfile
    sources=list((ROOT/'eagm2').glob('*.py'))+list((ROOT/'scripts/eagm2').glob('*.py'))+list((ROOT/'configs/eagm2').glob('*.json'))
    with tarfile.open(p/'source_snapshot.tar.gz','w:gz') as tar:
        for source in sources:tar.add(source,arcname=str(source.relative_to(ROOT)))
    dump(p/'source_hashes.json',{str(source.relative_to(ROOT)):hashlib.sha256(source.read_bytes()).hexdigest() for source in sources})
    return p
def environment():
    import mujoco
    gpu=subprocess.run(['nvidia-smi','--query-gpu=name,driver_version,memory.total,memory.free','--format=csv,noheader'],capture_output=True,text=True)
    return dict(python=platform.python_version(),numpy=np.__version__,torch=torch.__version__,mujoco=mujoco.__version__,cuda_available=torch.cuda.is_available(),cuda=torch.version.cuda,gpu=gpu.stdout.strip(),gpu_probe_error=gpu.stderr.strip(),git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),git_status=subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True),cpu_threads=torch.get_num_threads())
class Budget:
    def __init__(self):self.steps={};self.seconds={}
    def add(self,kind,n,seconds=0):self.steps[kind]=self.steps.get(kind,0)+int(n);self.seconds[kind]=self.seconds.get(kind,0)+float(seconds)
    def as_dict(self):return dict(control_steps=self.steps,simulator_wall_seconds=self.seconds,total_control_steps=sum(self.steps.values()))
@contextlib.contextmanager
def measured(device):
    cuda=str(device).startswith('cuda')
    if cuda:torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    result={};start=time.perf_counter()
    if cuda:begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);begin.record()
    try:yield result
    finally:
        if cuda:
            end.record();torch.cuda.synchronize();result.update(cuda_elapsed_seconds=begin.elapsed_time(end)/1000,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
        result['wall_seconds']=time.perf_counter()-start
        result['device']=str(device)
