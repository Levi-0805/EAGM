"""Restore GitHub-sized data parts, verifying SHA-256; never overwrite existing files."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
for item in json.loads((ROOT/'artifacts/research_data_parts/manifest.json').read_text()):
 target=ROOT/item['original']
 if target.exists():
  assert hashlib.sha256(target.read_bytes()).hexdigest()==item['sha256'],f'Existing file differs: {target}'
  print('Verified existing',item['original']);continue
 target.parent.mkdir(parents=True,exist_ok=True);temp=target.with_suffix('.restore_tmp')
 with temp.open('xb') as out:
  for part in item['parts']:
   content=(ROOT/part['path']).read_bytes();assert len(content)==part['size'] and hashlib.sha256(content).hexdigest()==part['sha256'];out.write(content)
 assert temp.stat().st_size==item['size'] and hashlib.sha256(temp.read_bytes()).hexdigest()==item['sha256']
 temp.rename(target);print('Restored',item['original'])
