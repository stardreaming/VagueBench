# VagueBench

**Evaluating embodied agents under vague instructions.**

VagueBench studies how instruction clarity affects embodied task execution. Built on ALFRED, it pairs clear and vague instructions with the same task, scene, and goal conditions, enabling direct comparison of agent performance.

- **1,303 instruction pairs** covering 2,606 instructions.
- **Paired evaluation** across clear and vague conditions.
- **A unified workflow** for evaluation, resumable runs, and paired result analysis.

## Getting started

From the repository root:

```bash
conda env create -f environment.yml
conda activate vaguebench
python -m vaguebench validate-data
```

Simulator evaluation runs on Linux and requires ALFRED trajectory assets and an X display. Follow the [running guide](docs/running.md) to prepare the simulator and connect a model.

## Documentation

- [Running VagueBench](docs/running.md) — setup, evaluation, and results.
- [Dataset](docs/data.md) — annotations and paired task structure.
- [Validation](docs/validation.md) — verified behavior and current coverage.

## Acknowledgements

VagueBench builds on [EmbodiedBench](https://github.com/EmbodiedBench/EmbodiedBench), [ALFRED](https://github.com/askforalfred/alfred), and [AI2-THOR](https://github.com/allenai/ai2thor). See [third-party notices](THIRD_PARTY_NOTICES.md) for attribution and [LICENSE](LICENSE) for the repository license.
