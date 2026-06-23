---
description: lint, test, commit and push, then create Pull Request
---

# Feature Branch PR Workflow

This workflow enforces the "Main is Sacred" rule. It checks the current branch,
commits changes, pushes the branch, and helps create a pull request.

## 1. Safety Checks

The agent must verify:

1. Branch check:

```bash
git branch --show-current
```

If output is `main`, create or request a short-lived feature branch before
proceeding. Do not commit directly to `main`.

2. Lint, type-check, and test:

```bash
.venv/bin/python -m ruff format .
.venv/bin/python -m ruff check .
.venv/bin/python -m mypy
.venv/bin/python -m pytest
```

If tests or checks fail, abort the PR flow and fix the failure first.

3. Manifest JSON check:

```bash
.venv/bin/python -c "import json; json.load(open('custom_components/groq_cloud_conversation/manifest.json')); print('Manifest is valid JSON')"
```

## 2. Commit

1. Stage: `git add .`
2. Commit according to `.agent/rules/git-workflow.md`.
3. Use `type: description`, for example:
   - `feat: add stt model options`
   - `fix: handle malformed tool calls`
   - `test: add structured output coverage`

## 3. Push & PR

1. Push:

```bash
git push -u origin HEAD
```

2. Prepare PR title and body:
   - PR title should summarize the full branch in plain language.
   - PR body should use:
     - `What changed`
     - `Why`
     - `Impact`
     - `Validation`
   - Because this repository uses squash merge, the PR title should read well
     as the final commit title on `main`.

3. Create PR:
   - Prefer `gh pr create --title "<PR_TITLE>" --body-file <PR_BODY_FILE> --web`
     if `gh` is available.
   - If `gh` is missing or fails, display:
     `https://github.com/ignazhabibi/ha_groq_integration/pull/new/<BRANCH_NAME>`

## 4. Releases

For tagging and version bumps, use `.agent/workflows/release.md`.
