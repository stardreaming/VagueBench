# Merged data format

The single public dataset is `vaguebench/data/dataset.json`. It is a list of 1,303 paired records flattened from `EmbodiedBench/v2-merged.json` in the original order. There are no separate split files or split-selection arguments.

```json
[
  {
    "episode_id": "vaguebench-00001",
    "task": "train/TASK/TRIAL/traj_data.json",
    "repeat_idx": 0,
    "base": "The original clear instruction.",
    "fuzzy": "The original vague instruction.",
    "type": [1]
  }
]
```

The example illustrates the schema. The full file preserves all original instruction strings, repeat indices, and metadata. Existing `type` values are retained without assigning category names or introducing category-based evaluation.

Only the trajectory root prefix is normalized: the part through `json_2.1.0/` is removed. The environment resolves `task` beneath the explicit `--data-root`. Both instruction conditions use the same original ALFRED trajectory and goal conditions. Each stable episode ID identifies the pair across outputs and resumed evaluations.

```bash
python -m vaguebench validate-data
python -m vaguebench validate-data --data-root /path/to/alfred/data/json_2.1.0
```

Validation checks relative task paths, nonempty instructions, unique episode IDs, and unique task/repeat identities. Supplying `--data-root` additionally checks trajectory existence. `--data-file` can select a different file in the same merged format for validation or evaluation.

Evaluation creates base/fuzzy runtime views in memory from this one file. It does not generate or require additional annotation files. The packaged distribution includes the complete merged annotations; simulator trajectory assets and experimental outputs are excluded.
