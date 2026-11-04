"""Aggregate only completed episode metrics; summaries never enter their own mean."""
import json
from pathlib import Path

METRICS = ('task_success', 'task_progress', 'reward', 'num_steps', 'planner_steps',
           'planner_output_error', 'num_invalid_actions', 'num_invalid_action_ratio',
           'empty_plan', 'episode_elapsed_seconds')


def summarize(folder):
    results = []
    for path in sorted(Path(folder).rglob('*_final_res.json')):
        record = json.loads(path.read_text(encoding='utf-8'))
        if record.get('status') != 'completed':
            continue
        results.append(record)
    fingerprints = {r['fingerprint'] for r in results if 'fingerprint' in r}
    if len(fingerprints) > 1:
        raise ValueError('Directory contains multiple experiment configurations; summarize each run separately')
    identities = [(r['eval_set'], r['episode_id']) for r in results]
    if len(identities) != len(set(identities)):
        raise ValueError('Duplicate episode IDs in one evaluation condition')
    groups = {}
    for record in results:
        groups.setdefault(record['eval_set'], []).append(record)
    summary = {}
    for condition, records in groups.items():
        summary[condition] = {'episodes': len(records)}
        for metric in METRICS:
            values = [r[metric] for r in records if isinstance(r.get(metric), (int, float))]
            if values:
                summary[condition][metric] = sum(values) / len(values)
    base = {r['episode_id']: r for r in groups.get('base', [])}
    fuzzy = {r['episode_id']: r for r in groups.get('fuzzy', [])}
    pairs = sorted(base.keys() & fuzzy.keys())
    summary['paired'] = {'episodes': len(pairs), 'base_only': len(base.keys() - fuzzy.keys()),
                         'fuzzy_only': len(fuzzy.keys() - base.keys())}
    if pairs:
        for condition, records in [('base', base), ('fuzzy', fuzzy)]:
            summary['paired'][f'{condition}_success'] = sum(records[k]['task_success'] for k in pairs) / len(pairs)
            summary['paired'][f'{condition}_progress'] = sum(records[k]['task_progress'] for k in pairs) / len(pairs)
    return summary
