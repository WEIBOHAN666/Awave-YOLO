# Contributing to AWave-YOLO

Thank you for improving AWave-YOLO. This document defines the minimum contribution process for code, experiments, and documentation.

## Scope
Contributions are welcome for:
- bug fixes and reproducibility improvements
- training/evaluation pipeline reliability
- benchmark extensions under the ASL dataset protocol
- documentation quality

## Issue Guidelines
When opening an issue, include:
- environment: OS, Python version, GPU/CUDA info
- exact command used
- relevant config values
- full traceback/log excerpt
- dataset directory layout (for path resolution issues)

For bug reports, provide a minimal reproducible example whenever possible.

## Pull Request Guidelines
Each PR should:
- target one clear change set (feature/fix/docs)
- describe motivation, implementation details, and expected impact
- list verification steps and command outputs used by the contributor
- avoid unrelated formatting-only changes in the same PR

## Code and Compatibility Rules
- Keep existing CLI interfaces backward compatible unless a breaking change is explicitly documented.
- Do not hardcode absolute local paths.
- Keep comments and variable names clear and concise.

## Testing and Verification
Before submitting a PR, run the following checks:

```bash
python prepare_data.py
python train.py
python eval.py
```

## Documentation Sync Requirement
If your PR changes behavior, paths, parameters, or outputs, update:
- `README.md`

## License
By contributing, you agree that your contributions are released under the MIT License in [LICENSE](./LICENSE).

## Dataset Compliance
Do not upload or redistribute restricted raw data in PRs.
Reference the official dataset link instead.
