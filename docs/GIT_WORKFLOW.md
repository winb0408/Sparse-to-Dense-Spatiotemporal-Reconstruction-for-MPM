# Git Workflow for threePM

## Branch strategy

- `main`: stable branch, only merge tested changes.
- `dev`: daily development branch, all routine work lands here first.

## First-time setup (already done)

```bash
git checkout main
git checkout -b dev
```

## Daily iteration steps

1) Switch to development branch:

```bash
git checkout dev
```

2) Pull latest updates before coding:

```bash
git pull
```

3) After edits, commit with clear message:

```bash
git add .
git commit -m "feat: short summary of this update"
git push origin dev
```

4) When `dev` is stable, merge to `main`:

```bash
git checkout main
git pull
git merge dev
git push origin main
```

## Commit message format (recommended)

- `feat:` new feature
- `fix:` bug fix
- `refactor:` code cleanup without behavior change
- `docs:` document updates
- `exp:` experiment or training/eval configuration change

Example:

```text
exp: tune temporal_window from 3 to 5 for CIDC25 A1/F0
```

