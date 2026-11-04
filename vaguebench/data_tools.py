"""The single merged annotation dataset and its paired runtime views."""
import json
from pathlib import Path, PurePosixPath

DATA_DIR = Path(__file__).parent / 'data'
DATA_FILE = DATA_DIR / 'dataset.json'


def load_dataset(path=None):
    return json.loads(Path(path or DATA_FILE).read_text(encoding='utf-8'))


def condition_tasks(rows, condition):
    if condition not in {'base', 'fuzzy'}:
        raise ValueError('Unknown evaluation condition')
    return [{'episode_id': row['episode_id'], 'task': row['task'],
             'repeat_idx': row['repeat_idx'], 'instruction': row[condition]} for row in rows]


def validate_data(data_file=None, data_root=None):
    rows = load_dataset(data_file)
    if not isinstance(rows, list) or not rows:
        raise ValueError('Dataset must be a nonempty merged list')
    seen = set()
    ids = set()
    root = Path(data_root).expanduser().resolve() if data_root else None
    for row in rows:
        path = PurePosixPath(row['task'])
        if path.is_absolute() or '..' in path.parts or '\\' in row['task']:
            raise ValueError('Nonportable task path')
        if type(row['repeat_idx']) is not int or row['repeat_idx'] < 0:
            raise ValueError('Invalid repeat_idx')
        identity = (row['task'], row['repeat_idx'])
        if identity in seen:
            raise ValueError('Duplicate task/repeat_idx')
        seen.add(identity)
        eid = row['episode_id']
        if not isinstance(eid, str) or not eid or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in eid) or eid in ids:
            raise ValueError('Invalid or duplicate episode_id')
        ids.add(eid)
        if any(not isinstance(row[field], str) or not row[field].strip() for field in ('base', 'fuzzy')):
            raise ValueError('Empty instruction')
        if root:
            task = (root / row['task']).resolve()
            if root not in task.parents or not task.is_file():
                raise ValueError(f'Missing or unsafe trajectory: {row["task"]}')
    return {'pairs': len(rows), 'instructions': 2 * len(rows)}
