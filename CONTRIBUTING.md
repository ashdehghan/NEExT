# Contributing to NEExT

Thanks for your interest in improving NEExT! Contributions of all kinds are welcome —
bug reports, documentation, and code.

## Reporting issues

Open an issue at [github.com/ashdehghan/NEExT/issues](https://github.com/ashdehghan/NEExT/issues).
For bugs, please include a minimal reproducible example, the NEExT version, and your
Python version.

## Development setup

```bash
# With uv (recommended)
uv venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
uv sync --extra dev --extra workbench-mcp

# Or with standard tooling
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -U pip
python3 -m pip install -e ".[dev,workbench-mcp]"
```

## Running tests

```bash
python3 -m pytest                  # full suite
python3 -m pytest --cov=NEExT      # with coverage
```

## Code style

Format and lint before opening a pull request (150-character line length):

```bash
python3 -m black .
python3 -m isort .
python3 -m ruff check .
```

## Pull requests

1. Fork the repo and create a branch for your change.
2. Keep each pull request focused on one thing.
3. Make sure tests pass and the formatters/linters are clean.
4. Describe what the change does and why in the PR description.

## License

By contributing, you agree that your contributions are licensed under the project's
[MIT License](LICENSE).
