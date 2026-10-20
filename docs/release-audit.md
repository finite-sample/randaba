# Release audit

Audit date: October 19, 2026. Candidate version: `0.1.0`.

The release-preparation work was merged in
[PR #5](https://github.com/finite-sample/randaba/pull/5), at commit
`e31367aec2f0f1af3578353bc6f96d4068e31407`. The repository is public. The
distribution and import names are both `randaba`, and the license is MIT.

## Validation

| Check | Result |
|---|---|
| Native Python 3.13 and Docker Python 3.12/3.14 | 95 tests pass; 100% line coverage |
| Installed wheel on all three Python versions | 95 tests pass |
| Minimum runtime dependencies: NumPy 2.0.0, SciPy 1.13.0, Python 3.12 | 95 installed-wheel tests pass |
| Ruff lint/format, Pyright, Pydoclint, pre-commit | Pass |
| Sphinx with warnings treated as errors | Pass |
| Wheel built from source distribution; strict Twine metadata checks | Pass |
| Preen strict release audit | All 27 checks pass |
| actionlint and Zizmor | Pass |
| Gitleaks scan of all local Git history and release changes | No secrets detected |
| Research note built with latexmk | Pass |

The [merged-commit CI run](https://github.com/finite-sample/randaba/actions/runs/39368294599)
checks the package, wheel, dependency lock, and workflow security.
The [release rehearsal](https://github.com/finite-sample/randaba/actions/runs/39368318827)
passed with publishing disabled, producing the wheel and source distribution
through the release workflow.

The vulnerability audit cannot assess `py-canon` against PyPI advisories
because it is a Git-sourced documentation dependency. Its commit is pinned
in `uv.lock`. The full research simulation campaigns were not rerun; the
package tests include the checked-in R anticlust parity fixtures.

## Publication status

The GitHub repository is public. PyPI publication has not been attempted.
The GitHub environment `pypi` exists. Before pushing a release tag, configure
the matching PyPI pending trusted publisher and finalize the dated changelog.
See [Releasing randaba](releasing.md) for the exact publisher settings and
release procedure. A successful build rehearsal does not validate PyPI
credentials or reserve the package name.
