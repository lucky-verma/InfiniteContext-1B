# AGENTS.md

## Cursor Cloud specific instructions

### Project Status

This project is in **planning phase** (~5% complete). All documentation describes the *target* architecture. There is no application to "start" or "serve" yet — Phase 1 implementation has not begun.

### Development Environment

- **Python 3.12.3** on the VM (meets the 3.10+ requirement from `memory-bank/techContext.md`).
- Dev tools (`black`, `ruff`, `mypy`, `pytest`, `pytest-cov`) are installed via `pip3 install`.
- Tool binaries are in `~/.local/bin` — the VM's `~/.bashrc` adds this to `PATH`.

### Available Commands

| Task | Command |
|------|---------|
| Lint (ruff) | `ruff check .` |
| Format check | `black --check .` |
| Type check | `mypy <file_or_dir>` |
| Run tests | `pytest tests/ -v` |

### Key Files

- `pyproject.toml` — project metadata, tool configuration (ruff, black, mypy, pytest).
- `CLAUDE.md` — planned commands, architecture overview, and technology stack.
- `memory-bank/` — 6 planning docs; read `techContext.md` for dependencies and `activeContext.md` for current focus.

### Gotchas

- No `requirements.txt` or `setup.py` — dependencies are defined in `pyproject.toml` `[project.optional-dependencies]`.
- The planned directory structure (`infra/`, `kernels/`, `training/`, `serving/`) is scaffolded but contains no implementation code yet.
- GPU-dependent work (Triton kernels, FSDP training, vLLM serving) requires NVIDIA hardware not present on this Cloud VM. CPU-only development and testing is the expected workflow here.
