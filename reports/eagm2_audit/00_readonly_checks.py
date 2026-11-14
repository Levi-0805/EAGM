"""Read-only provenance and paired-return audit. Writes only stdout."""
from pathlib import Path
import csv, hashlib, itertools, json
ROOT = Path(__file__).resolve().parents[2]
manifest = json.loads((ROOT/'paper_eagm/data/source_manifest.json').read_text())
hashes = {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() == h for p,h in manifest.items()}
per_seed=[]
for seed in range(5,10):
    p=ROOT/f'runs/stage5a_confirm/seed{seed}/per_start.csv'
    rows=list(csv.DictReader(p.open()))
    grouped = {}
    for row in rows:
        key = (row['episode_id'], row['frame'], row['delay'])
        method = row['model']
        assert key not in grouped.setdefault(method, {}), 'duplicate paired start'
        grouped[method][key] = float(row['return_'])
    base, direction = grouped['eagm_base'], grouped['eagm_dir_matched']
    assert base.keys() == direction.keys(), 'unpaired starts'
    difference = sum(direction[k]-base[k] for k in base)/len(base)
    per_seed.append({'seed':seed,'paired_starts':len(base),'dir_minus_base':difference})
d = [s['dir_minus_base'] for s in per_seed]
observed = abs(sum(d))
p_exact = sum(abs(sum(s*x for s,x in zip(signs,d))) >= observed-1e-15
              for signs in itertools.product((-1,1),repeat=len(d)))/(2**len(d))
result = {'source_count':len(hashes),'all_source_hashes_match':all(hashes.values()),
          'stage5a_per_seed':per_seed,'stage5a_mean_difference':sum(d)/len(d),
          'two_sided_exact_sign_flip_p':p_exact,
          'note':'Descriptive historical audit; no new confirmation, no pooling of stages.'}
assert all(hashes.values()), 'source manifest mismatch'
assert abs(result['stage5a_mean_difference']-0.0005141552735794117)<1e-14
print(json.dumps(result,indent=2))
