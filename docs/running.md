# Running VagueBench

This guide covers installation, simulator setup, model connections, and evaluation. For annotation fields and trajectory conventions, see [the data guide](data.md).

## Installation

Use Linux with Python 3.9 and an X display for AI2-THOR 2.1.0. From the repository root:

```bash
conda env create -f environment.yml
conda activate vaguebench
```

Alternatively, use a Python virtual environment:

```bash
python3.9 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[eval]'
```

An existing compatible environment can also run `python -m vaguebench` directly from the checkout without installing the package. The environment name in `environment.yml` applies to newly created environments; it does not rename an existing installation.

The existing research environment has passed real scene restoration, navigation, and image/log output checks for both conditions. A complete clean simulator installation has not yet been verified. Evaluation dependencies pin OpenCV and the historical Flask/Werkzeug stack required by AI2-THOR. See [validation notes](validation.md) for the tested versions and remaining checks.

For native Claude access, also install `'.[anthropic]'`. The optional local LMDeploy backend uses `'.[local]'` together with the evaluation dependencies. API evaluation does not import LMDeploy or require local model weights.

Dataset validation and result aggregation require only Python's standard library.

## Data

All 1,303 annotated pairs are merged into one file: `vaguebench/data/dataset.json`. Each record contains both `base` and `fuzzy` text with a shared stable episode ID. Evaluation reads both conditions directly from this file; there are no partitioned split files.

Task paths are relative to an external ALFRED `json_2.1.0` directory. Simulator trajectory assets are obtained separately through the [ALFRED project](https://github.com/askforalfred/alfred).

```bash
python -m vaguebench validate-data
python -m vaguebench validate-data --data-root /path/to/alfred/data/json_2.1.0
```

See [the data format](data.md) for record fields and path conventions.

## Evaluation

Start an X display using your server setup. The retained helper can start display 1 on a compatible NVIDIA/Xorg host:

```bash
python -m vaguebench.envs.eb_alfred.scripts.startx 1
```

If display 1 is already running, reuse it. First check the simulator without a model or API credentials:

```bash
python -m vaguebench smoke-test \
  --data-root /path/to/alfred/data/json_2.1.0 \
  --x-display 1 \
  --output-dir outputs/simulator-smoke
```

This loads the first trajectory under both instructions, executes one navigation action, and saves images and logs. It checks simulator operation; it does not evaluate task completion. To reuse an existing AI2-THOR **2.1.0** Unity binary, add `--thor-executable /path/to/thor-201909061227-Linux64/thor-201909061227-Linux64` to `smoke-test` or `evaluate`. Without this option, AI2-THOR uses its default cache under the current user's home and may download the binary.

Set credentials in environment variables. `.env.example` lists the variable names; it is not loaded automatically.

```bash
export OPENAI_API_KEY='your_api_key_here'
python -m vaguebench evaluate \
  --data-root /path/to/alfred/data/json_2.1.0 \
  --model-name gpt-4o \
  --limit 1 \
  --output-dir outputs/smoke
```

Both `base` and `fuzzy` run by default. Remove `--limit 1` to evaluate all 1,303 pairs. The standard ALFRED prompt is the default. Defaults retain 10 examples, 500-pixel images, environment feedback, a maximum of 30 environment steps, and a maximum of 10 invalid actions.

For a separately hosted OpenAI-compatible model:

```bash
export OPENAI_API_KEY='your_endpoint_key_here'
export VAGUEBENCH_BASE_URL='http://localhost:8000/v1'
python -m vaguebench evaluate \
  --data-root /path/to/alfred/data/json_2.1.0 \
  --model-name YOUR_SERVED_MODEL --provider openai \
  --response-format json_object \
  --output-dir outputs/served_model
```

`--response-format` selects `json_schema` (default), `json_object`, or `text` according to the server's supported output format. Provider choices are `openai`, `anthropic`, `gemini`, and `dashscope`; `auto` recognizes Claude and Gemini names and otherwise selects OpenAI-compatible access. Credentials use `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, or `DASHSCOPE_API_KEY`. `--api-key-env` overrides the variable name. `--base-url` overrides the endpoint. Native provider and local model paths have not yet been exercised against live services in this candidate.

Resume an interrupted run by repeating the same command with `--resume`. Resume checks the configuration, source code, annotation data, prompt, and examples before reusing completed episodes. A failed episode stops the run and writes an error marker; retrying it requires `--resume`. No simulator or paid API calls are made by the data commands.

## Results and checks

```bash
python -m vaguebench summarize outputs/smoke
python -m unittest discover -s tests -v
```

Results include task success, goal-condition progress, reward, environment steps, planner steps, model parsing errors, invalid actions, empty plans, and elapsed time. Aggregate reports show completed episode counts and paired results over matching episode IDs. Missing counterparts are counted separately. Error markers and pre-existing summaries are excluded from means.

See [migration notes](migration.md) for the release scope, behavior changes, provenance, and remaining validation work.

