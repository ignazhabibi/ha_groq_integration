---
description: analyze changes, bump version, generate changelog, and tag release
---

# Release Workflow

This workflow creates a semantic release for the Groq Cloud Home Assistant
custom integration.

## Stable vs. Pre-release Tags

- Stable releases use tags like `vX.Y.Z`.
- Prerelease validation builds use `.github/workflows/pre-release.yml`.
- Supported prerelease tag formats are `vX.Y.Z-alpha.N`, `vX.Y.Z-beta.N`, and
  `vX.Y.Z-rc.N`.
- Hyphenated prerelease tags are intentionally excluded from the stable release
  publish step.

## 1. Pre-flight Checks

1. Ensure `main` is up to date:

```bash
git checkout main
git pull --ff-only origin main
```

2. `git status` must show a clean working tree.

3. Run the same quality gates as CI:

```bash
.venv/bin/python -m ruff check .
.venv/bin/python -m ruff format --check .
.venv/bin/python -m mypy
.venv/bin/python -m pytest -q
```

4. For dependency, CI, packaging, or metadata changes, validate a fresh install
   when practical:

```bash
python3.14 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install '.[dev]'
.venv/bin/python -m pytest -q
```

5. For release workflow changes, verify the generated asset is a Home Assistant
   integration archive containing `custom_components/groq_cloud_conversation/`,
   `manifest.json`, translations, brand assets, `hacs.json`, `README.md`, and
   `LICENSE`.
6. Confirm the generated archive filename matches `hacs.json` `filename`.

## 2. Analysis & Versioning

1. Read current versions from:
   - `custom_components/groq_cloud_conversation/manifest.json`
   - `pyproject.toml`
2. If they differ, call that out and align them in the release work.
3. Analyze commits since the latest tag:

```bash
git log --pretty=format:"%h %s" $(git describe --tags --abbrev=0)..HEAD
```

If no tags exist, use `git log`.

4. Determine bump:
   - MAJOR: breaking changes
   - MINOR: new features
   - PATCH: fixes, docs, chores
5. Tell the user:
   - current version,
   - whether version sources are aligned,
   - changes grouped by type,
   - proposed new version.
6. Wait for confirmation before editing files, creating a branch, tagging, or
   pushing release refs.

## 3. Message Hierarchy

- Commit messages: atomic and technical, `type: description`.
- PR titles and bodies: summarize the whole branch for reviewers.
- Release notes: summarize merged user-facing outcomes. Do not replay every raw
  commit subject.

## 4. Generate Changelog

Use this style:

```markdown
# Changelog

## Breaking Changes
<summary> (PR #xx, commit <hash>) (BC)

## New Features
<summary> (PR #xx, commit <hash>)

## Bug Fixes
<summary> (PR #xx, commit <hash>)

## Other Changes
<summary> (PR #xx, commit <hash>)
```

## 5. Execution

1. Create release branch:

```bash
git checkout -b release/v<NEW_VERSION>
```

2. Bump both version sources:
   - `custom_components/groq_cloud_conversation/manifest.json`
   - `pyproject.toml`

3. Commit:

```bash
git add custom_components/groq_cloud_conversation/manifest.json pyproject.toml
git commit -m "chore: bump version to <NEW_VERSION>"
```

4. Push release branch:

```bash
git push -u origin release/v<NEW_VERSION>
```

5. Open a PR into `main` and wait for `quality-check` to pass.

6. After merge, refresh local main:

```bash
git checkout main
git pull --ff-only origin main
```

7. Tag the merged main commit:

```bash
git -c core.commentChar=";" tag -a v<NEW_VERSION> -m "Release v<NEW_VERSION>" -m "<PASTE_CHANGELOG_HERE>"
git push origin v<NEW_VERSION>
```

## 6. Post-release

- Do not claim the release is live until:
  - the merged `main` push run is green, and
  - the tag run for `v<NEW_VERSION>` is green.
- For prerelease validation tags, check the matching `pre-release.yml` run.
- Perform a final documentation drift check:
  - `README.md`
  - `AGENTS.md`
  - `.agent/rules/tech-stack.md`
  - `.agent/rules/git-workflow.md`
  - `.agent/workflows/release.md`
- Confirm release workflow status:

```bash
gh run list --workflow release.yml --limit 5
```
