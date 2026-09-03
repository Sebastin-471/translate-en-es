# Contributing to translate-en-es

Thank you for your interest in contributing! This document outlines the process and standards for contributing to this project.

## Code of Conduct
This project follows the [Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/version/2/1/code_of_conduct/). By participating, you agree to uphold this code.

## Getting Started

### Prerequisites
- Python 3.11+
- Git
- (Optional) CUDA 12.1+ for GPU acceleration

### Development Setup
```bash
# Clone the repository
git clone https://github.com/your-org/translate-en-es.git
cd translate-en-es

# Create virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\Activate.ps1
# Linux/macOS:
source .venv/bin/activate

# Install in development mode with all dev dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

## Development Workflow

### 1. Create a Branch
```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

### 2. Make Changes
- Follow the existing code style (enforced by Ruff)
- Add tests for new functionality
- Update documentation (ADRs for architectural changes)

### 3. Run Quality Checks
```bash
# Lint and format
ruff check src tests
ruff format src tests

# Type check
mypy src

# Run tests
pytest tests/unit -v
pytest tests/integration -v

# Security scan (optional)
pip-audit
osv-scanner .
```

### 4. Commit
```bash
git add .
git commit -m "feat: add amazing feature"
```
Follow [Conventional Commits](https://www.conventionalcommits.org/):
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation only
- `refactor:` Code restructuring
- `test:` Test additions/changes
- `chore:` Maintenance tasks

### 5. Push and Create PR
```bash
git push origin feature/your-feature-name
```
Open a Pull Request against `main`.

## Code Standards

### Python Style
- **Ruff** for linting and formatting (line length: 100)
- **MyPy** strict mode for type checking
- **Structlog** for structured logging
- **Dataclasses** with `slots=True` for data objects
- **Protocols** for interfaces (structural subtyping)

### Architecture Principles
- **Hexagonal Architecture**: Core domain pure; infrastructure at edges
- **Dependency Injection**: Composition root (`app.py`) only place importing infrastructure
- **Async First**: `asyncio` for concurrency; `asyncio.Queue` for inter-stage communication
- **Thread Safety**: Hotkeys/Tray use async queue dispatch, not `run_coroutine_threadsafe`

### Testing
- **Unit Tests**: Mock all infrastructure; test core logic in isolation
- **Integration Tests**: Full pipeline with mock engines
- **GPU Tests**: Marked `@pytest.mark.gpu`; run on self-hosted runners
- **Coverage Target**: ≥80% for core modules

### Configuration
- All settings in `config/base.yaml` + environment overlays
- Environment variables: `TRANSLATOR_<SECTION>__<KEY>`
- Hot-reload supported for development

## Pull Request Checklist
- [ ] Tests pass locally
- [ ] Ruff/MyPy pass
- [ ] Coverage maintained or improved
- [ ] Documentation updated (ADR for architectural changes)
- [ ] No breaking changes without version bump
- [ ] Conventional commit messages

## Release Process
1. Version bump in `pyproject.toml`
2. Update `CHANGELOG.md`
3. Tag release: `git tag v0.x.x`
4. GitHub Actions builds and publishes to PyPI
5. MSIX installer built for Windows

## Architecture Decision Records (ADRs)
For significant architectural changes, create an ADR in `docs/adr/`:
1. Copy template from `docs/adr/README.md`
2. Number sequentially (001, 002, ...)
3. Include Context, Decision, Consequences
4. Link from PR description

## Questions?
- Open a GitHub Discussion
- Check existing Issues and PRs
- Review ADRs for design rationale