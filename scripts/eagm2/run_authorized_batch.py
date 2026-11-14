"""Sequential, gated execution of exactly nine exploratory roots; no confirmation."""
from __future__ import annotations
import argparse
import ast
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
AUTHORIZED = {'tracking': (200001, 200002, 200003), 'contact': (201001, 201002, 201003), 'vision': (202001, 202002, 202003)}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hashes(root):
    files = set()
    for folder in ('eagm2', 'scripts/eagm2', 'configs/eagm2', 'controllers', 'training', 'tests/eagm2', 'assets'):
        base = root/folder
        if base.exists():
            files.update(p for p in base.rglob('*') if p.is_file() and '__pycache__' not in p.parts
                         and p.suffix in ('.py', '.json', '.xml'))
    return {str(p.relative_to(root)): digest(p) for p in sorted(files)}


def write_json(path, obj):
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(obj, indent=2, ensure_ascii=False)+'\n')
    temp.replace(path)


def valid_name(name):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', name):
        raise ValueError('attempt must be one safe directory name')
    return name


def validate_gates(root, smokes, tests_log):
    if set(smokes) != set(AUTHORIZED):
        raise ValueError('explicit successful smoke path required for each of three tasks')
    evidence = {}
    for task, path in smokes.items():
        path = path.resolve()
        config = json.loads((path/'config.json').read_text())
        manifest = json.loads((path/'run_manifest.json').read_text())
        result = json.loads((path/'results.json').read_text())
        if config.get('task') != task or config.get('seed') not in AUTHORIZED[task] or config.get('smoke') is not True:
            raise ValueError(f'{task}: invalid smoke identity')
        if manifest.get('status') != 'completed' or result.get('status') != 'completed' or (path/'failure.json').exists():
            raise ValueError(f'{task}: smoke did not successfully complete')
        evidence[task] = {'path': str(path), 'config_hash': digest(path/'config.json'),
                          'manifest_hash': digest(path/'run_manifest.json'), 'result_hash': digest(path/'results.json')}
    content = tests_log.read_text()
    if not re.search(r'^Ran \d+ tests? in ', content, re.M) or not re.search(r'^OK\s*$', content, re.M):
        raise ValueError('tests log must contain successful full verbose unittest output')
    if re.search(r'^(FAILED|ERROR:|FAIL:|Traceback)', content, re.M):
        raise ValueError('tests log includes failure')
    missing = []
    testfiles = sorted((root/'tests/eagm2').glob('test_*.py'))
    for file in testfiles:
        names = {node.name for node in ast.walk(ast.parse(file.read_text()))
                 if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith('test_')}
        for name in names:
            if not re.search(r'^'+re.escape(name)+r' \('+re.escape(file.stem)+r'\.', content, re.M):
                missing.append(file.stem+'.'+name)
    if not testfiles or missing:
        raise ValueError('test log does not cover full current suite: '+', '.join(missing))
    return dict(smokes=evidence, tests_log=str(tests_log.resolve()), tests_log_sha256=digest(tests_log))


def build_plan(root, python, attempt, updates=1000, policy_updates=50):
    valid_name(attempt)
    if updates <= 0 or not 0 < policy_updates <= 50:
        raise ValueError('positive updates required, policy updates capped at 50')
    return [dict(task=task, seed=seed, attempt=attempt, status='planned', worker_pid=None,
                 command=[str(python), str(root/'scripts/eagm2/run_development.py'), '--task', task,
                          '--seed', str(seed), '--attempt', attempt, '--device', 'cuda',
                          '--updates', str(updates), '--policy-updates', str(policy_updates)],
                 run_dir=str(root/'runs/eagm2_development'/task/f'seed{seed}'/attempt))
            for task, seeds in AUTHORIZED.items() for seed in seeds]


def persist(batch, rows):
    write_json(batch/'experiment_status.json', rows)
    with (batch/'experiment_status.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, ['task', 'seed', 'attempt', 'status', 'worker_pid', 'exit_code', 'wall_seconds'], extrasaction='ignore')
        writer.writeheader(); writer.writerows(rows)


def run_batch(root, batch, rows, frozen, *, python, dry_run=False):
    persist(batch, rows)
    if dry_run:
        return
    for row in rows:
        if source_hashes(root) != frozen:
            row['status'] = 'blocked_source_drift'; persist(batch, rows)
            raise RuntimeError('training/source files changed since batch preflight; stopping')
        if Path(row['run_dir']).exists():
            row['status'] = 'blocked_existing_attempt'; persist(batch, rows)
            raise FileExistsError(row['run_dir'])
        begun = time.perf_counter()
        row['status'] = 'running'
        with (batch/f"{row['task']}_{row['seed']}.console.log").open('x') as log:
            try:
                worker = subprocess.Popen(row['command'], cwd=root, stdout=log, stderr=subprocess.STDOUT)
            except Exception as exc:
                row.update(status='failed_to_launch', error=repr(exc))
                persist(batch, rows)
                raise
            row['worker_pid'] = worker.pid; persist(batch, rows)
            try:
                code = worker.wait()
            except BaseException:
                worker.terminate(); worker.wait()
                row.update(status='interrupted', exit_code=worker.returncode, wall_seconds=time.perf_counter()-begun)
                persist(batch, rows)
                raise
        row.update(exit_code=code, wall_seconds=time.perf_counter()-begun)
        manifest_path = Path(row['run_dir'])/'run_manifest.json'
        try:
            completed = manifest_path.exists() and json.loads(manifest_path.read_text()).get('status') == 'completed'
        except (OSError, ValueError) as exc:
            completed = False
            row['manifest_error'] = repr(exc)
        row['status'] = 'completed' if code == 0 and completed else 'failed'
        persist(batch, rows)
        if row['status'] != 'completed':
            raise RuntimeError(f"worker {row['task']}:{row['seed']} failed; later roots not launched")
    if source_hashes(root) != frozen:
        raise RuntimeError('source changed before analysis; stopping')
    command = [str(python), str(root/'scripts/eagm2/analyze_development.py'), '--attempt', rows[0]['attempt'],
               '--output-dir', str(batch/'analysis'), '--report-tag', 'authorized_batch']
    with (batch/'analysis.console.log').open('x') as log:
        result = subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT)
    write_json(batch/'analysis_status.json', dict(command=command, exit_code=result.returncode))
    if result.returncode:
        raise RuntimeError('analysis failed; raw experiments and analysis error log retained')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke', action='append', required=True, help='task=/absolute/successful/smoke/run')
    parser.add_argument('--tests-log', type=Path, required=True, help='full verbose unittest discovery success log')
    parser.add_argument('--batch-dir', type=Path, required=True)
    parser.add_argument('--attempt', default='development_v1')
    parser.add_argument('--updates', type=int, default=1000)
    parser.add_argument('--policy-updates', type=int, default=50)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    smokes = {}
    for spec in args.smoke:
        task, path = spec.split('=', 1)
        if task in smokes: raise ValueError('duplicate smoke task')
        smokes[task] = Path(path)
    evidence = validate_gates(ROOT, smokes, args.tests_log)
    auth = json.loads((ROOT/'configs/eagm2/development_authorization.json').read_text())
    if auth['authorized_root_seeds'] != {k:list(v) for k,v in AUTHORIZED.items()} or auth.get('confirmation_authorized') is not False:
        raise PermissionError('authorization does not match fixed nine exploratory roots')
    rows = build_plan(ROOT, sys.executable, args.attempt, args.updates, args.policy_updates)
    for row in rows:
        if Path(row['run_dir']).exists(): raise FileExistsError(row['run_dir'])
    args.batch_dir.mkdir(parents=True, exist_ok=False)
    frozen = source_hashes(ROOT)
    write_json(args.batch_dir/'source_hashes.json', frozen)
    write_json(args.batch_dir/'gates.json', evidence)
    write_json(args.batch_dir/'plan.json', rows)
    state = dict(status='planned' if args.dry_run else 'running', scheduler_pid=os.getpid(),
                 phase='exploratory_development', confirmation_authorized=False, dry_run=args.dry_run)
    write_json(args.batch_dir/'batch_manifest.json', state)
    try:
        run_batch(ROOT, args.batch_dir, rows, frozen, python=sys.executable, dry_run=args.dry_run)
        state['status'] = 'validated_dry_run' if args.dry_run else 'completed'
    except BaseException as exc:
        state.update(status='failed', error=repr(exc))
        raise
    finally:
        write_json(args.batch_dir/'batch_manifest.json', state)


if __name__ == '__main__':
    main()
