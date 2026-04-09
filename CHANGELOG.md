# Changelog

All notable changes to this project are documented in this file.

## [Unreleased]

### Added

- Git workflow guide: `docs/GIT_WORKFLOW.md`
- Experiment log template: `docs/EXPERIMENT_LOG_TEMPLATE.md`
- Pull request template: `.github/pull_request_template.md`
- Version-iteration management section in `README.md`

### Changed

- Added `.gitignore` rules to keep large datasets and generated artifacts out of version control.

---

## How to update this file

For each meaningful update:

1. Add one bullet under `Unreleased` (`Added` / `Changed` / `Fixed` / `Removed`).
2. Keep each bullet short and user-facing (what changed and why).
3. Before a release, rename `[Unreleased]` to a version tag (e.g., `[v0.1.0] - 2026-04-09`) and create a new empty `[Unreleased]` section.

