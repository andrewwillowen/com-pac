# Copilot Instructions

## Project Overview

`com-pac` is a Python CLI tool that calculates the **Center of Mass** and **Principal Axes** for molecules, given Cartesian coordinates and isotopologue information. It outputs rotational constants, dipole moments, and principal axes coordinates.

## Tech Stack

- **Language**: Python 3.10+
- **Build tool**: [Hatch](https://hatch.pypa.io/)
- **Key dependencies**: `numpy`, `pandas`, `mendeleev`
- **Code formatter**: [Black](https://github.com/psf/black)
- **Linter**: [Ruff](https://docs.astral.sh/ruff/)
- **Testing**: `pytest`, `hypothesis`

## Project Structure

```
src/com_pac/   # Main package source code
tests/         # Unit and end-to-end tests
docs/          # Documentation source
```

## Build & Test Commands

```bash
# Install hatch
pip install hatch

# Run tests
hatch test

# Format code (check only)
hatch run hatch-static-analysis:format-check

# Format code (apply fixes)
hatch run hatch-static-analysis:format-fix

# Lint code (check only)
hatch run hatch-static-analysis:lint-check

# Lint code (apply fixes)
hatch run hatch-static-analysis:lint-fix
```

## Coding Standards

- Follow [PEP 8](https://peps.python.org/pep-0008/) style conventions.
- Use **Black** for code formatting — do not manually adjust formatting that Black controls.
- Use **Ruff** for linting — resolve all lint warnings before submitting.
- Add docstrings to all public functions and classes.
- Prefer `pathlib.Path` over string-based file paths.
- Keep functions focused and testable; avoid overly long functions.

## Commit Message Format

All commit messages must follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

**Common types:**

| Type       | When to use                                         |
|------------|-----------------------------------------------------|
| `feat`     | A new feature                                       |
| `fix`      | A bug fix                                           |
| `docs`     | Documentation changes only                          |
| `style`    | Formatting, whitespace, import ordering, etc. (no logic change) |
| `refactor` | Code change that neither fixes a bug nor adds a feature |
| `test`     | Adding or updating tests                            |
| `chore`    | Maintenance tasks (build, CI, dependencies, etc.)   |
| `perf`     | Performance improvements                            |

**Examples:**

```
feat(parser): support case-insensitive section headings
fix(diagonalize): correct eigenvalue sorting for near-zero inertias
docs: update README with installation instructions
chore: upgrade numpy to latest compatible version
test: add hypothesis tests for parser edge cases
```

## Pull Request Guidance

- Reference the relevant issue number in the PR description (e.g., `Closes #42`).
- Ensure all tests pass (`hatch test`) before requesting review.
- Keep PRs focused — one logical change per PR.
