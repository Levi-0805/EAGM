import sys
from pathlib import Path
ROOT = Path('/home/aaa/workspace/EAGM-2.0-full-validation')
sys.path.insert(0, str(ROOT))
import eagm2.runtime as runtime
ALLOWED = set()
for _task, _seeds in {'tracking': [200101, 200102, 200103, 200104, 200105, 200106, 200107, 200108, 200109, 200110, 200111, 200112, 200113, 200114, 200115], 'contact': [201101, 201102, 201103, 201104, 201105, 201106, 201107, 201108, 201109, 201110, 201111, 201112, 201113, 201114, 201115], 'vision': [202101, 202102, 202103, 202104, 202105, 202106, 202107, 202108, 202109, 202110, 202111, 202112, 202113, 202114, 202115]}.items():
    for _seed in _seeds:
        ALLOWED.add((_task, _seed))

def check_seed(task, seed):
    if (task, seed) not in ALLOWED:
        raise PermissionError(f'not a confirmation seed: {task} {seed}')

def new_run(task, seed, attempt):
    check_seed(task, seed)
    p = ROOT / 'runs' / 'eagm2_confirmation' / task / f'seed{seed}' / attempt
    p.mkdir(parents=True, exist_ok=False)
    for n in ('training_logs', 'checkpoints'):
        (p / n).mkdir()
    import tarfile, hashlib
    sources = (
        list((ROOT / 'eagm2').glob('*.py'))
        + list((ROOT / 'scripts' / 'eagm2').glob('*.py'))
        + list((ROOT / 'configs' / 'eagm2').glob('*.json'))
    )
    with tarfile.open(p / 'source_snapshot.tar.gz', 'w:gz') as tar:
        for source in sources:
            tar.add(source, arcname=str(source.relative_to(ROOT)))
    runtime.dump(
        p / 'source_hashes.json',
        {str(source.relative_to(ROOT)): hashlib.sha256(source.read_bytes()).hexdigest() for source in sources},
    )
    return p

runtime.check_seed = check_seed
runtime.new_run = new_run
sys.argv = [str(ROOT / 'scripts' / 'eagm2' / 'run_development.py')] + sys.argv[1:]
import runpy
runpy.run_path(str(ROOT / 'scripts' / 'eagm2' / 'run_development.py'), run_name='__main__')
