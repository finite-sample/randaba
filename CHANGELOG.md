# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Set the release version with `uv version X.Y.Z`, record the release here, and
tag the merged commit `vX.Y.Z`. Pushing the matching tag builds and publishes
that version.

## [Unreleased]

### Added

- Randomized assignment-based anticlustering for experiments with many arms.
- Complete, blocked, sequential covariate-adaptive, and rerandomized
  assignment, plus exchange anticlustering and greedy pair-switching.
- Arm contrasts and Monte Carlo randomization p-values under the sharp null.
- R anticlust parity fixtures, simulation scripts, and the research note.
- Input validation and a finite candidate budget for rerandomization.
- Local release checks with `make ci` and standard Python Docker images.

### Changed

- Use `randaba` consistently as the distribution and import name. The
  unpublished `balance_buys` package is replaced without a compatibility alias.
- Require 95% test coverage in CI and enable Pages deployment and dependency
  review automatically when the repository is public.

[Unreleased]: https://github.com/finite-sample/randaba/commits/main
