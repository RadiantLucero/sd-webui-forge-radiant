# Contributing

Forge Neo targets Python 3.13 and requires a platform-appropriate PyTorch installation and supported hardware for image-generation runtime checks. Follow the [installation instructions](README.md#installation) for a runnable environment. You do not need to install the full runtime just to make documentation changes or run Ruff.

For an overview of the startup, generation, and extension paths, see [docs/architecture.md](docs/architecture.md).

## Development workflow

This repository is a fork of `Haoming02/sd-webui-forge-classic`. Keep the local `neo` branch as a clean mirror of `upstream/neo`, and use the persistent `dev` branch for local development and commits.

```bash
git fetch upstream
git switch neo
git merge --ff-only upstream/neo
git switch dev
git rebase neo
```

This updates `neo` from the original and rebases local development commits onto it. Resolve any conflicts before continuing. If `dev` has already been pushed, rebasing changes its commit history; update the fork branch with `--force-with-lease` only when that is intended.

When you decide to publish, push `dev` to your fork and open a pull request directly from `RadiantLucero/sd-webui-forge-radiant:dev` to `Haoming02/sd-webui-forge-classic:neo`. No pull request within your own fork is needed.

```bash
git push -u origin dev
```

## Checks

Run Ruff on each changed Python file:

```bash
uvx --from ruff==0.17.4 ruff check path/to/file.py
```

The GitHub Actions workflow runs this check on Python files changed in a pull request. Existing Ruff findings elsewhere in the repository are tracked incrementally and are not a reason to reformat unrelated code.

There is no project-level automated test suite at present. For behavior changes, add focused tests when practical and describe any manual verification, including required GPU hardware, model files, or platform-specific dependencies. If you could not run a check, say so in the pull request.

## Pull requests

- Explain the problem and the intended change.
- Keep each pull request focused; avoid unrelated refactors.
- Include the checks you ran and their results.
- Call out behavior changes, compatibility impact, and hardware or model requirements.
