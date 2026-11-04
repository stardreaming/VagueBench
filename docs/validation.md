# Runtime review — 2026-11-04

Checked the current checkout with the existing `embench2` environment (Python 3.9.20). Its core packages match the evaluation pins: NumPy 1.23.5, Gym 0.23.0, AI2-THOR 2.1.0, Pillow 11.0.0, SciPy 1.13.1, NetworkX 3.2.1, OpenCV 4.10.0.84, and OpenAI 1.60.0.

| Check | Result |
| --- | --- |
| `python -m unittest discover -s tests -v` in `embench2` | All 15 tests passed, including four optional runtime regressions |
| Standard-library environment, same command | 11 tests passed, four optional runtime tests skipped |
| `python -m compileall -q vaguebench main.py tests` | Passed |
| Annotation and trajectory checks | All 1,303 paths exist in the ALFRED checkout's `data/json_2.1.0`; all JSON files parse, repeat indices exist, and corresponding layout arrays are bundled |
| Real `python -m vaguebench smoke-test ...` | Passed for `base` and `fuzzy`: first trajectory restored in FloorPlan11, `find a Pan` succeeded, PNG images and JSONL traces saved |
| Offline wheel build | Passed with `pip wheel --no-index --no-deps --no-build-isolation` |
| Existing environment's `pip check` | Reports two missing packages: `pycparser` (required by cffi) and `mdurl` (required by markdown-it-py); the runtime checks above passed without them |

The simulator check reused an existing display 1 and the cached `thor-201909061227-Linux64` executable, supplied explicitly with `--thor-executable`. The EmbodiedBench checkout's trajectory copy is incomplete for this dataset; use the ALFRED checkout's complete trajectory root. The existing shared Conda environment was not modified.

Fixed during this review:

- Clear the connector's sliced-object preference and placement retry counts when restoring each episode; previous episodes could otherwise affect later episodes.
- Handle absent name matches when navigating to an explicit object ID.
- Replace the removed Pillow `getsize` API in image annotation with `getbbox`.
- Correct the environment seed method's misspelled simulator call.
- Include the required JSON field template for `text`/`json_object` model responses and explicitly selected Anthropic backends.
- Reject evaluation resolutions below AI2-THOR's 300-pixel minimum before simulator startup, and clean up the simulator server when startup fails.
- Pin OpenCV and the historical Flask/Werkzeug dependencies to versions present in `embench2`; add `environment.yml` and an offline simulator smoke command.

These checks exercise real simulator startup and a navigation action, but do not demonstrate full model-driven task completion, live API/provider compatibility, or local LMDeploy inference. No model API requests or clean environment installation were performed. To repair the two unrelated missing packages in the shared environment, its owner can run `python -m pip install pycparser mdurl` after activating it.

# Earlier candidate validation — 2026-11-02

Checks run from the separate release candidate directory:

| Check | Result |
| --- | --- |
| `python -m vaguebench validate-data` | 1,303 pairs, 2,606 instructions, one merged data file |
| `python -m vaguebench validate-data --data-root <local ALFRED json_2.1.0>` | All referenced trajectory files exist under the supplied root |
| Comparison with original `v2-merged.json` | Every field matches for all 1,303 records after normalizing the trajectory root prefix, flattening groups, and adding episode IDs |
| `python -m unittest discover -s tests -v` | 10 tests passed |
| `python -m compileall -q vaguebench main.py tests` | Passed |
| Planner parser smoke | Valid actions, empty plans, out-of-range IDs, and boolean IDs handled as expected |
| Internal import scan | All `vaguebench` import targets resolve inside the candidate; no legacy package imports |
| Targeted source publication scan | No observed credential literals or author-specific local filesystem paths |
| Offline wheel build with `pip wheel --no-index --no-deps --no-build-isolation` | Passed |
| Wheel resource inspection | Canonical annotations, one merged dataset, 120 layout arrays, rewards, and ALFRED license present |
| Wheel extracted outside the checkout, `python -S -m vaguebench validate-data` | Passed with standard-library-only execution |

Tests cover paired runtime consistency, duplicate episode-ID rejection, unsafe/missing trajectory paths, invalid sampling configuration, empty-plan metrics, invalid-action limits, preservation of partial progress, mocked paired evaluation and resume, incompatible resume rejection, paired aggregation, and summary/mixed-run exclusion.

The unit and mocked evaluation checks do not demonstrate simulator compatibility or live model performance. No real Unity episode, Xorg startup, native provider request, or local model inference was run. A complete clean installation with the historical evaluation dependency set has not been verified. The targeted source scan does not cover repository history.
