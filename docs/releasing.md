# Releasing randaba

The first public version is prepared as `0.1.0`. The package is not yet
published on PyPI. Both the distribution and Python import are `randaba`.
The repository and [documentation site](https://finite-sample.github.io/randaba/)
are public. The [release audit](release-audit.md) records the completed checks.

## First public release

1. The release-preparation branch is merged and the public launch is complete.
   Confirm the release commit has passing CI before tagging.
2. Run `make ci-docker` and `make ci-docker PYTHON_VERSION=3.14` on the release
   commit. The checks build a source distribution, build its wheel, validate
   metadata, and run tests against the installed wheel in an isolated environment.
3. Pages is configured to use GitHub Actions. Confirm the Docs workflow
   deploys successfully from `main`.
4. The GitHub Actions environment `pypi` exists. In PyPI, add a pending
   trusted publisher with these exact values:

   | Field | Value |
   |---|---|
   | PyPI project | `randaba` |
   | GitHub owner | `finite-sample` |
   | Repository | `randaba` |
   | Workflow | `release.yml` |
   | Environment | `pypi` |

   Follow the [PyPI pending-publisher instructions](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/).
   Publishing uses OIDC; no PyPI API key is needed.
5. Dispatch the Release workflow on `main` with `publish` left false for a
   build rehearsal. A branch run cannot publish to PyPI.
6. Move the changelog's Unreleased entries into a dated `0.1.0` section and
   leave an empty Unreleased section. Confirm `CITATION.cff`,
   `pyproject.toml`, and `uv.lock` agree on the version. Commit and merge
   these release notes, and wait for CI to pass on that commit.
7. Tag that commit `v0.1.0` and push the tag. **A matching tag push publishes
   to PyPI and creates a GitHub release.** Check both jobs: the GitHub release
   can exist even if the separate PyPI publishing job fails.
8. Verify `pip install randaba` in a clean environment, then make that the
   primary README installation command and remove the pre-release wording
   from this guide. Add the standard PyPI badge at that point so it reports
   the published version automatically:

   ```markdown
   [![PyPI](https://img.shields.io/pypi/v/randaba.svg)](https://pypi.org/project/randaba/)
   ```

## Later releases

Run `uv version X.Y.Z`, update `CITATION.cff` and the changelog, and run the
same local and CI checks. Push `vX.Y.Z` only after the release commit passes.
The release workflow rejects tags whose version differs from the built wheel.

If publication needs retrying, dispatch Release from the existing tag with
`publish` enabled. Already uploaded distributions are skipped. See
[PyPI's trusted-publishing documentation](https://docs.pypi.org/trusted-publishers/using-a-publisher/)
for the publishing mechanism.
