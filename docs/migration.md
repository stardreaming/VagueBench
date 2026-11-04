# Release candidate scope and review

Confirmed scope: ALFRED `base/fuzzy` evaluation plus the full annotations from `v2-merged.json`. Preserve existing labels without interpreting them. Other environments, ReAct, program prompting, defense prompting, experimental output directories, and historical analysis/selection scripts are excluded.

The source `EmbodiedBench` and existing `VagueBench` directories were read without changing their code. This candidate is prepared separately for review. No remote publication, git commit, or upload has been performed.

## Findings

- Existing VagueBench was largely copied from EmbodiedBench: the inspected text/config subset had 328 identical files. Its README was a placeholder and its model transport differed. There were 95 legacy `embodiedbench` imports in 26 Python files.
- The old entry point still used the EmbodiedBench package/config paths and referred to environments that were absent from VagueBench.
- The environment loaded its split path from a lowercase environment variable and its trajectories from machine-specific annotation paths.
- The active source evaluator used `defense_prompt`. This candidate selects the original `alfred_system_prompt` for the agreed baseline-only scope.
- Seven research launch scripts contained literal credential assignments, including commented assignments. They were not copied. This was a targeted source inspection, not a git-history audit.
- `merged.json` used `base/vague` and relative paths; `v2-merged.json` used per-partition records with metadata and absolute paths. After normalizing paths, both contain the same ordered task/repeat identities. Compared with `merged.json` and the existing `new_*.json` evaluation splits, v2 differs in 2 base texts and 39 fuzzy texts. The v2 text is authoritative for this release. All original source groups are flattened into one public dataset, as confirmed by the owner; there are no seven-way split files.

## Changes

- Independent `vaguebench` package and command line; no runtime dependency on the old EmbodiedBench checkout.
- Explicit data root, output directory, display, model provider, and endpoint options. API keys come only from environment variables.
- Retain ALFRED simulator/task/action logic, baseline examples, reward configuration, and the 120 navigation layout arrays required by graph loading. Retain the ALFRED license alongside these files.
- Optional SDK/local model imports, so data tools run without simulator or model dependencies. API runs do not import LMDeploy.
- Stable IDs for paired episodes; resume refuses incompatible configurations, code, annotations, prompts, or examples.
- Bound model transport retries and stop on unrecovered errors. Remove the evaluator's unbounded exception/sleep loop and interactive debugger in trace serialization.
- Avoid nonfinite means for episodes with no actions; keep the last task metrics when the next plan is empty or invalid.
- Summaries read completed episode files only and exclude existing summaries. Aggregate only one experiment configuration at a time and match base/fuzzy by episode ID.
- Add portable font fallback and configurable X display.

## Provenance

`migration_manifest.json` records the original v2 annotation SHA-256 and source-file hashes for migrated files. Instruction strings, repeat indices, and original metadata are retained exactly. Trajectory paths are normalized relative to the external ALFRED dataset root. The merged list preserves the original record order across source groups and ALFRED goal evaluation. Stable global episode IDs are added.

## Validation boundary

The candidate is checked with Python compilation, data/source consistency checks, standard-library unit tests, mocked evaluation/resume tests, package-content checks, and a targeted scan for literal credentials and machine-specific paths. Exact commands and results are recorded in `validation.md` after running them.

The initial preparation environment lacked Gym. A subsequent review on 2026-11-04 reused `embench2` and verified real scene restoration and a navigation action for both conditions; see `validation.md`. A complete model-driven episode and live model calls remain to be verified. No dependency installation or paid API execution was performed during these checks.

The root MIT license is copied from the existing VagueBench folder; this preparation does not independently assign a new license to upstream components or the annotations. Publication metadata, citation text, dataset hosting, and any paper-specific protocol remain for the owner to finalize.
