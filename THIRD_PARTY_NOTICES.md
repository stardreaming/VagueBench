# Third-party provenance

- **EmbodiedBench**: the ALFRED environment wrapper, connector, planner, baseline prompt/examples, and related evaluation design derive from the existing EmbodiedBench research checkout. The source README identifies EmbodiedBench, ICML 2025, by Rui Yang and collaborators. See [the upstream repository](https://github.com/EmbodiedBench/EmbodiedBench).
- **ALFRED**: simulator environment/task/reward code, generation utilities, and navigation layouts are retained from the existing ALFRED subtree. Its original MIT license is included at `vaguebench/envs/eb_alfred/LICENSE`.
- **AI2-THOR**: the simulator is an external dependency and is not bundled as a simulator binary in this repository.

The root `LICENSE` is retained from the existing VagueBench repository. Upstream notices remain applicable to their respective files. Annotation provenance and the release scope are documented in `docs/migration.md`.
