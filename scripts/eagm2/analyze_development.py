"""Read-only, descriptive audit of explicitly named authorized development attempts.

No automatic latest/best-attempt discovery, selection, confirmatory inference or
training imports. Missing/invalid outcomes remain null, with costs retained.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import itertools
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AUTHORIZED = {'tracking': (200001, 200002, 200003), 'contact': (201001, 201002, 201003), 'vision': (202001, 202002, 202003)}
FACTORIAL = {(0,0,0): 'ff_base', (0,0,1): 'ff_dir', (1,0,0): 'delay_base', (1,0,1): 'delay_dir', (0,1,0): 'recurrent_base', (0,1,1): 'recurrent_dir', (1,1,0): 'combined_base', (1,1,1): 'combined_dir'}
METRICS = {'return': 'physical.mean_return', 'success': 'physical.success_rate', 'delta_return': 'delta_return_vs_initial', 'cosine': 'gradient.mean_cosine', 'nmse': 'model_metrics.one_step_nmse', 'command_mae': 'model_metrics.command_mae', 'validation_nmse': 'validation_model_metrics.one_step_nmse', 'validation_return': 'validation_physical.mean_return', 'validation_success': 'validation_physical.success_rate', 'validation_cosine': 'validation_gradient.mean_cosine'}


def numeric(x):
    return x if isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) else None


def get(obj, key):
    for part in key.split('.'):
        obj = obj.get(part) if isinstance(obj, dict) else None
    return obj


def read(path, errors):
    if not path.exists():
        return {}
    try:
        obj = json.loads(path.read_text(), parse_constant=lambda _: None)
        if not isinstance(obj, dict):
            raise ValueError('expected JSON object')
        return obj
    except (ValueError, OSError) as exc:
        errors.append({'file': str(path), 'error': str(exc)})
        return {}


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row)) or ['status']
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False, allow_nan=False) if isinstance(v, (dict, list)) else v for k,v in row.items()})


def flatten_numbers(obj, prefix=''):
    if isinstance(obj, dict):
        for k,v in obj.items():
            yield from flatten_numbers(v, f'{prefix}.{k}' if prefix else k)
    elif numeric(obj) is not None:
        yield prefix, obj


def describe(values):
    good = [v for v in values if numeric(v) is not None]
    return dict(n_available=len(good), n_missing=len(values)-len(good), mean=statistics.mean(good) if good else None, median=statistics.median(good) if good else None, sd=statistics.stdev(good) if len(good)>1 else None, positive=sum(v>0 for v in good), zero=sum(v==0 for v in good), negative=sum(v<0 for v in good), leave_one_seed_out_means=[statistics.mean(good[:i]+good[i+1:]) for i in range(len(good))] if len(good)>1 else [])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--attempt', required=True, action='append', help='Exact attempt name, or task=attempt; repeat only for different tasks')
    p.add_argument('--tasks', default=','.join(AUTHORIZED))
    p.add_argument('--runs-root', type=Path, default=ROOT/'runs/eagm2_development')
    p.add_argument('--output-dir', type=Path, default=ROOT/'reports/eagm2_dev')
    p.add_argument('--report-tag', default='development')
    a = p.parse_args()
    tasks = a.tasks.split(',')
    if len(set(tasks)) != len(tasks) or any(t not in AUTHORIZED for t in tasks):
        p.error('tasks must be unique authorized task names')
    attempts = {}
    for spec in a.attempt:
        names, attempt = ([spec.split('=',1)[0]], spec.split('=',1)[1]) if '=' in spec else (tasks, spec)
        if not attempt or Path(attempt).name != attempt or attempt in ('.','..'):
            p.error('attempt must be one exact directory name')
        for task in names:
            if task not in tasks or task in attempts:
                p.error('one explicit attempt per requested task is required')
            attempts[task] = attempt
    if set(attempts) != set(tasks):
        p.error('provide an attempt for every requested task')
    if Path(a.report_tag).name != a.report_tag or a.report_tag in ('.','..'):
        p.error('invalid report tag')
    prefix = '06_' + a.report_tag
    a.output_dir.mkdir(parents=True, exist_ok=True)
    if list(a.output_dir.glob(prefix + '_*')):
        p.error('report tag already exists; choose a new tag to preserve the audit')
    rows, episodes, probes, costs, budgets, errors, artifacts, run_rows = [], [], [], [], [], [], [], []
    for task in tasks:
        attempt = attempts[task]
        for seed in AUTHORIZED[task]:
            run = a.runs_root/task/f'seed{seed}'/attempt
            ident = dict(task=task, seed=seed, attempt=attempt)
            config = read(run/'config.json', errors)
            manifest = read(run/'run_manifest.json', errors)
            failure = read(run/'failure.json', errors)
            total = read(run/'results.json', errors)
            budget = read(run/'budget.json', errors) or total.get('budget', {}) or failure.get('budget', {})
            status = manifest.get('status', total.get('status', 'missing'))
            excluded = bool(config.get('smoke')) or 'smoke' in attempt.lower() or 'invalid' in attempt.lower() or status in ('invalid', 'invalidated') or bool(manifest.get('invalid'))
            config_mismatch = (config.get('task', task) != task or config.get('seed', seed) != seed or config.get('attempt', attempt) != attempt)
            excluded = excluded or config_mismatch
            run_rows.append(ident | dict(status=status, excluded=excluded, exclusion_reason='smoke/invalid/config mismatch' if excluded else None, failure=failure, config=config))
            for name in ('config.json','environment.json','source_snapshot.tar.gz','source_hashes.json','split_manifest.json','budget.json','run_manifest.json'):
                file = run/name
                artifacts.append(ident | dict(artifact=name, exists=file.is_file(), sha256=hashlib.sha256(file.read_bytes()).hexdigest() if file.is_file() else None))
            for key,val in flatten_numbers(budget):
                budgets.append(ident | dict(scope='actual_attempt_shared_not_per_method', category=key, value=val, run_status=status, excluded=excluded))
            for scope, obj in [('attempt', total or manifest), ('failed_attempt', failure), ('shared_critic', read(run/'critic_training.json',errors).get('resource',{})), ('shared_vision', read(run/'vision_training.json',errors))]:
                for key,val in flatten_numbers(obj):
                    if any(s in key for s in ('wall_seconds','cuda_elapsed_seconds','peak_allocated_bytes','peak_reserved_bytes')):
                        costs.append(ident | dict(method=None, scope=scope, measure=key, value=val, run_status=status, excluded=excluded))
            methods = config.get('methods', '')
            methods = methods.split(',') if isinstance(methods, str) else list(methods)
            methods = sorted(set(filter(None, methods)) | {q.parent.name for q in run.glob('*/results.json')}) if run.exists() else []
            if not methods:
                rows.append(ident | dict(method=None, status='missing_no_method_manifest', included=False, **{m:None for m in METRICS}))
            for method in methods:
                result_path = run/method/'results.json'
                result = read(result_path, errors)
                direct = method in ('direct_fd', 'spsa')
                direct_complete = direct and result.get('stop_reason') in ('max_updates_reached', 'insufficient_budget_for_complete_iteration') and isinstance(result.get('physical'), dict)
                method_status = result.get('status', 'completed' if direct_complete else 'missing')
                valid = not excluded and method_status == 'completed' and result.get('seed',seed)==seed and result.get('method',method)==method
                row = ident | dict(method=method, status=method_status, run_status=status, included=valid, source=str(result_path), result_sha256=hashlib.sha256(result_path.read_bytes()).hexdigest() if result_path.is_file() else None)
                row.update({metric:numeric(get(result,key)) if valid else None for metric,key in METRICS.items()})
                row.update(trainable_parameters=numeric(get(result,'parameter_counts.trainable')), checkpoint_exists=(run/'checkpoints'/f'{method}.pt').is_file() or (run/method/'checkpoint.pt').is_file(), model_metrics_applicable=not direct, gradient_metrics_applicable=not direct, optimization_stop_reason=result.get('stop_reason'), optimization_updates=numeric(result.get('updates',get(result,'training.updates'))), physical_episode_count=len(get(result,'physical.per_episode') or []), probe_count=len(get(result,'gradient.per_probe') or []), undefined_cosine_count=sum(numeric(get(probe,'metrics.cosine')) is None for probe in get(result,'gradient.per_probe') or []), adaptation='deterministic-actor PAML adaptation; not original-paper reproduction' if method=='paml_adaptation' else 'VaGraM-inspired value-state-gradient weighting adaptation; not original-paper reproduction' if method=='vagram_adaptation' else None)
                rows.append(row)
                for split, physical_key, gradient_key in [('development_test','physical','gradient'), ('validation','validation_physical','validation_gradient')]:
                    for ep in get(result,physical_key+'.per_episode') or []:
                        episodes.append(ident | dict(method=method,included=valid,split=split,**ep))
                    for probe in get(result,gradient_key+'.per_probe') or []:
                        probes.append(ident | dict(method=method,included=valid,split=split,**probe))
                # Preserve direct optimizer failures even when no final results.json exists.
                for source in ('optimization.json','actual_simulator_accounting.json'):
                    supplemental=read(run/method/source,errors)
                    for key,val in flatten_numbers(supplemental):
                        if any(token in key for token in ('control_steps','queries','query_cap')) and not key.startswith('ledger.'):
                            budgets.append(ident | dict(method=method,scope=source,category=key,value=val,run_status=method_status,excluded=not valid))
                        if 'wall_seconds' in key:
                            costs.append(ident | dict(method=method,scope=source,measure=key,value=val,run_status=method_status,excluded=not valid))
                for key,val in flatten_numbers(result):
                    if any(token in key for token in ('control_steps','physics_substeps','objective_queries','query_cap')):
                        budgets.append(ident | dict(method=method,scope='method_results_do_not_sum_duplicates',category=key,value=val,run_status=method_status,excluded=not valid))
                for key,val in flatten_numbers(result):
                    if any(s in key for s in ('wall_seconds','cuda_elapsed_seconds','peak_allocated_bytes','peak_reserved_bytes')):
                        costs.append(ident | dict(method=method,scope='method_nested_do_not_sum',measure=key,value=val,run_status=status,excluded=not valid))
    # Differences use whole root-seed units, never probe/episode pseudo-replication.
    paired, factorial, capacity, summaries = [], [], [], []
    for task in tasks:
        relevant = [r for r in rows if r['task']==task and r['method']]
        methods = sorted({r['method'] for r in relevant})
        index = {(r['seed'],r['method']):r for r in relevant}
        for left,right in itertools.combinations(methods,2):
            for metric in METRICS:
                diffs=[]
                for seed in AUTHORIZED[task]:
                    x=index.get((seed,left),{}).get(metric);y=index.get((seed,right),{}).get(metric)
                    d=x-y if x is not None and y is not None else None
                    paired.append(dict(task=task,attempt=attempts[task],seed=seed,left=left,right=right,metric=metric,left_value=x,right_value=y,difference=d))
                    diffs.append(d)
                summaries.append(dict(task=task,attempt=attempts[task],analysis='paired_difference',contrast=f'{left} - {right}',metric=metric,**describe(diffs)))
        for metric in METRICS:
            effects = {name:[] for name in ('D','R','G','DxR')}
            for seed in AUTHORIZED[task]:
                vals={cell:index.get((seed,method),{}).get(metric) for cell,method in FACTORIAL.items()}
                missing=[FACTORIAL[cell] for cell,v in vals.items() if v is None]
                for effect in effects:
                    value=None
                    if not missing:
                        if effect=='DxR':
                            value=sum(vals[1,1,g]-vals[1,0,g]-vals[0,1,g]+vals[0,0,g] for g in (0,1))/2
                        else:
                            axis={'D':0,'R':1,'G':2}[effect]
                            value=sum((1 if cell[axis] else -1)*v for cell,v in vals.items())/4
                    factorial.append(dict(task=task,attempt=attempts[task],seed=seed,metric=metric,effect=effect,value=value,missing_cells=missing))
                    effects[effect].append(value)
            for effect,values in effects.items():
                summaries.append(dict(task=task,attempt=attempts[task],analysis='factorial',contrast=effect,metric=metric,**describe(values)))
        for seed in AUTHORIZED[task]:
            ref=index.get((seed,'combined_dir'),{})
            for name in ('capacity_mlp','capacity_gru'):
                item=index.get((seed,name),{})
                n=item.get('trainable_parameters'); nref=ref.get('trainable_parameters')
                rel=(n/nref-1) if n is not None and nref else None
                capacity.append(dict(task=task,attempt=attempts[task],seed=seed,method=name,reference='combined_dir',parameters=n,reference_parameters=nref,relative_difference=rel,within_five_percent=abs(rel)<=.05 if rel is not None else None,flops_available=False,cosine_difference=item.get('cosine')-ref.get('cosine') if item.get('cosine') is not None and ref.get('cosine') is not None else None,return_difference=item.get('return')-ref.get('return') if item.get('return') is not None and ref.get('return') is not None else None))
    tables=dict(seed_raw=rows,paired_differences=paired,factorial=factorial,capacity=capacity,descriptive=summaries,episodes=episodes,gradient_probes=probes,resources=costs,budgets=budgets,artifacts=artifacts)
    for name,table in tables.items():
        write_csv(a.output_dir/f'{prefix}_{name}.csv',table)
    write_json(a.output_dir/f'{prefix}_audit.json', dict(phase='development_exploratory',confirmation_authorized=False,attempts=attempts,authorized_roots={t:AUTHORIZED[t] for t in tasks},runs=run_rows,read_errors=errors,statistics='descriptive only; no p-values, significance, seed pooling, winner or checkpoint selection',tables=tables))
    write_json(a.output_dir/f'{prefix}_confirmation_draft.json',dict(status='not_authorized',confirmation_authorized=False,selected_candidate=None,primary_comparison=None,freeze_complete=False,required_before_authorization=['validation-only selection record or explicit no-go','full protocol and comparison family freeze','failure floor and exclusion rules','capacity/FLOPs and complete query budgets','environment/code/config/data hashes','task feasibility and input-isolation acceptance','second author approval'],confirmation_roots_accessed=False))
    included=sum(r['included'] for r in rows)
    report=f'''# Exploratory development audit\n\nExplicit attempts: `{json.dumps(attempts,ensure_ascii=False)}`. Included completed method-root runs: {included}/{len(rows)}. JSON read errors: {len(errors)}.\n\nOnly the nine authorized development roots can be read. No confirmation data are opened. This report neither selects a winner nor changes metrics. With at most three independent root seeds per task, all summaries are descriptive: no significance claims, p-values, confirmatory confidence intervals, or pooling of tasks. Test episodes and gradient probes do not increase the seed count.\n\nDirect FD/SPSA need no world-model checkpoint or gradient-fidelity metric; those fields are not applicable, not zero. Their optimizer stop reason and actual/reserved query counts are retained.\n\nMissing, failed, smoke and invalid runs remain visible; excluded outcomes are null. Complete methods in an otherwise failed attempt remain available unless the attempt is explicitly invalid. NaN/undefined metrics remain null; no unregistered failure floor is invented. Available-case means report their missing count and must not conceal incomplete evidence. Failure records and resource costs remain in the audit.\n\nThe raw table separates development-test outcomes from validation metrics. Missing validation physical returns or gradient cosine prevent the protocol's validation-only selection rule from being evaluated; development-test metrics cannot substitute. Protocol acceptance gates, contact reachability/full-episode success and vision isolation require separate actual logs. A positive local reward or cosine does not establish complete-task success.\n\nD/R/G use the eight ff/delay/recurrent/combined Base/Dir cells, excluding frozen legacy methods. Main effects average four differences; D×R averages the two difference-in-differences. Any missing cell leaves that seed's effect undefined. Only tracking full-coordinate cosine is the protocol's proposed confirmation mechanism endpoint; all present factorial estimates are exploratory. Capacity controls compare exact trainable counts against combined_dir and flag ±5%; missing FLOPs remain unverified. Return gains without cosine gains do not support gradient fidelity, and correlations would not establish mediation.\n\nPAML is a deterministic-actor/observable-interface adaptation, not an original-paper reproduction. VaGraM is an inspired value-state-gradient weighting adaptation; critic fit, representation assumptions, extra parameters and resource/query costs require disclosure. Privileged information is not a fair deployable comparison or a mathematical upper bound.\n\nResource CSV preserves actual logged scopes. Wall times are elapsed wall time; CUDA-event elapsed time is **not GPU-active kernel time**, utilization or hardware energy. Peak allocation/reservation are measured memory peaks, not additive usage. Nested stage/method/attempt times overlap and must not be summed. Shared critic/visual/data costs are separate; absent per-method attribution, physics-substep counts, FLOPs or failure-phase GPU timing are unknown, never zero. Budget CSV records actual available control-step categories; common-data comparisons do not imply equal total query budgets or cost-effectiveness. Unrequested attempts are not scanned, so historical failure totals require separate explicit analysis.\n\nConfirmation draft status: **not_authorized**. No candidate is populated and no confirmation run is launched.\n'''
    (a.output_dir/f'{prefix}_report.md').write_text(report)
    print(json.dumps(dict(report=str(a.output_dir/f'{prefix}_report.md'),included_method_roots=included,read_errors=len(errors))))


if __name__ == '__main__':
    main()
