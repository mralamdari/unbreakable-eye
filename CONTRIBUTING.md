# Contributing to Unbreakable Eye

First off, thanks for taking the time to contribute! 🎉

## Code of Conduct

This project adheres to the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to [mralamdari2000@gmail.com](mailto:mralamdari2000@gmail.com).

## How Can I Contribute?

### 🐛 Reporting Bugs

Before submitting a bug report:
- Check the [issues](https://github.com/mralamdari/unbreakable-eye/issues) to avoid duplicates
- Use the **Bug Report** template when creating a new issue
- Include your environment (OS, Python version, hardware)
- Attach relevant logs and error messages
- If possible, include a minimal reproduction

### 💡 Suggesting Features

- Use the **Feature Request** template
- Explain the use case — *why* is this valuable?
- If it's a new AI backend, link to the model paper/repo
- If it's a deployment target, mention the hardware

### 🔧 Pull Requests

1. **Fork** the repository
2. **Create a branch**: `git checkout -b feat/my-feature` or `fix/my-bug`
3. **Make your changes**
4. **Run tests**: `pytest tests/unit/ -v`
5. **Lint**: `ruff check .`
6. **Type check**: `mypy src/`
7. **Commit** with a descriptive message:
   - `feat: add support for ...`
   - `fix: resolve crash when ...`
   - `docs: update README for ...`
   - `refactor: simplify ...`
   - `ci: fix test pipeline`
8. **Push** and open a Pull Request

### 🧪 Testing Guidelines

- All new features must include unit tests
- Run the full test suite before opening a PR: `pytest tests/unit/ -v`
- Integration tests are in `tests/integration/` — these need hardware and are not run in CI
- If you add a new detector backend, add a corresponding integration smoke test

### 🏗️ Architecture Notes

- **Layer isolation**: `src/core/` (infra) → `src/vision/` (models) → `src/engine/` (pipeline) → `src/web/` (API). Keep imports flowing one way.
- **Detector factory**: Add new backends via `src/vision/detectors/` and register in `src/vision/factory.py`.
- **Config**: All env vars go in `src/core/config.py` (Pydantic Settings). New Telegram config goes in `src/telegram/config.py`.

### 🤔 Questions?

Open a [Discussion](https://github.com/mralamdari/unbreakable-eye/discussions) or tag `@mralamdari` in an issue.

---

Happy building! 🚀
