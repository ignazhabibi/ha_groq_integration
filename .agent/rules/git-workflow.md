---
trigger: always_on
---

# Git & Development Workflow

Strict guidelines for version control and feature development in
`ha_groq_integration`.

## 1. Core Principle: Main Is Sacred

- Never commit directly to `main` unless the user explicitly asks for an
  emergency bypass and confirms the tradeoff.
- Treat pull requests as required for all changes, including documentation,
  `.agent/`, and CI workflow updates.
- `main` should always be green.
- Use short-lived feature branches.

## 2. Feature Branch Lifecycle

### Start

1. `git checkout main`
2. `git pull --ff-only`
3. `git checkout -b feature/<descriptive-name>`

### Develop

1. Change integration code under `custom_components/groq_cloud_conversation/`
   and tests under `tests/`.
2. Run focused tests while iterating.
3. Before commit, run the relevant quality gates:

```bash
.venv/bin/python -m pytest
.venv/bin/python -m ruff check .
.venv/bin/python -m ruff format --check .
.venv/bin/python -m mypy
```

4. Commit with `type: description`.
   Valid types: `feat`, `fix`, `refactor`, `docs`, `test`, `chore`.
   Do not require a scope by default.

### Submit

1. `git push -u origin HEAD`
2. Open a pull request into `main`.
3. Wait for the `quality-check` pipeline to pass.
4. Merge through GitHub.
5. Delete the branch after merge.

### Re-sync

1. `git checkout main`
2. `git pull --ff-only origin main`
3. `git branch -d <branch>`

## 3. Release Process

- Stable releases use tags like `vX.Y.Z` and the GitHub Actions workflow
  `.github/workflows/release.yml`.
- Pre-release validation uses tags like `vX.Y.Z-alpha.N`, `vX.Y.Z-beta.N`, or
  `vX.Y.Z-rc.N` and `.github/workflows/pre-release.yml`.
- Release workflows publish a Home Assistant integration archive, not an empty
  Python wheel.
- Keep the archive filename fixed and aligned with `hacs.json` `filename`, so
  HACS can download release assets predictably.
- Version changes must update both:
  - `custom_components/groq_cloud_conversation/manifest.json`
  - `pyproject.toml`
- The agent must stop for user confirmation before bumping versions, creating
  release branches, tagging, or pushing release tags.

## 4. Agent Role

- Allowed after normal confirmation: create branches, stage, commit, and push.
- Forbidden without explicit user instruction: merge PRs, bypass `main`, create
  release tags, or claim a release is live before the relevant GitHub Actions
  runs are green.
