# Agent instructions

## Project overview

Forge Neo is a Python 3.13 application with a Gradio-based web UI and a GPU-dependent image-generation runtime.

- `launch.py`, `webui.py`, and `modules/` contain startup, application, and UI integration code.
- `backend/` and `modules_forge/` contain model execution, memory management, and Forge-specific backend code.
- `javascript/`, `html/`, `style.css`, and `script.js` implement the browser UI.
- `extensions-builtin/` contains built-in extensions; check local conventions and licenses before changing bundled or third-party code.
- `requirements.txt` lists runtime dependencies. `pyproject.toml` sets the supported Python version and Ruff rules.
- See [docs/architecture.md](docs/architecture.md) for the startup flow, generation path, and code navigation map.

## Working rules

- Keep changes focused and preserve existing behavior unless the task explicitly asks to change it.
- Follow the surrounding code's conventions; avoid unrelated formatting or cleanup.
- Do not add model weights, generated output, local configuration, or virtual environments to Git.
- Do not assume the application can start in a CPU-only or dependency-free environment. Model loading and inference require platform-specific dependencies and supported hardware.
- There is currently no project-level automated test suite. Add focused tests for new behavior when it can be isolated from the GPU runtime.
- If a check cannot be run, state that clearly and explain why; do not report an unrun check as passing.

## Validation

For a Python change, run Ruff on the changed Python file(s):

```bash
uvx --from ruff==0.17.4 ruff check path/to/file.py
```

The repository currently has existing Ruff findings, so the CI workflow checks changed Python files rather than requiring the entire legacy codebase to be clean. When working in a file with existing findings, distinguish pre-existing issues from findings introduced by the change.

For UI or runtime changes, run an appropriate manual smoke test when the required environment is available, and report any GPU or model requirements.
