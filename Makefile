PYTHON_VERSION ?= 3.12
UV ?= uv

.PHONY: sync lint test docs build wheel-test ci ci-docker

sync:
	$(UV) sync --locked --group docs

lint:
	$(UV) run --locked ruff check .
	$(UV) run --locked ruff format --check .
	$(UV) run --locked pyright
	$(UV) run --locked pydoclint src/

test:
	$(UV) run --locked pytest --cov=randaba --cov-report=term-missing

docs:
	$(UV) run --locked --group docs sphinx-build -W --keep-going -b html docs docs/_build/html

build:
	$(UV) build
	$(UV) run --locked twine check --strict dist/*

wheel-test: build
	@set -eu; \
	venv=$$(mktemp -d); \
	trap 'rm -rf "$$venv"' EXIT; \
	$(UV) venv "$$venv"; \
	$(UV) pip install --python "$$venv/bin/python" dist/*.whl --group test; \
	"$$venv/bin/python" -I -c 'import randaba; print(randaba.__file__); assert "site-packages" in randaba.__file__'; \
	"$$venv/bin/python" -I -m pytest --import-mode=importlib tests

ci:
	$(MAKE) sync lint test docs wheel-test

ci-docker:
	docker run --rm -v "$(CURDIR):/source:ro" -w /work \
		-e UV_PYTHON_DOWNLOADS=never python:$(PYTHON_VERSION)-bookworm \
		sh -ec 'tar -C /source --exclude=.venv --exclude=.git --exclude=dist --exclude=docs/_build --exclude=results --exclude=__pycache__ -cf - . | tar -xf -; python -m pip install --disable-pip-version-check uv; make ci'
