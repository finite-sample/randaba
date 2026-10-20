# Development

From a checkout with Python 3.12 or newer and `uv` installed:

```bash
uv sync --locked --group docs
make ci
```

The checks cover linting, formatting, types, docstrings, tests with at least
95% coverage, documentation, distribution metadata, and tests against an
installed wheel. The documentation includes the README directly.

For local CI in standard Python Docker images:

```bash
make ci-docker
make ci-docker PYTHON_VERSION=3.14
```

Docker uses a temporary copy of the checkout. To work on the simulation
figures, also install the simulation dependencies with `uv sync --group sims`.
See [simulation reproduction](simulations.md) and the [release guide](releasing.md).
