# Troubleshooting

This document contains solutions to common issues you might encounter while installing or using this project.

## Build error: `ModuleNotFoundError: No module named 'pkg_resources'`

When installing packages (such as `flatdict` for `isaaclab`), you might encounter a build error:

```text
  × Failed to build `flatdict==4.0.1`
  ...
  ModuleNotFoundError: No module named 'pkg_resources'
```

### Cause
Recent versions of `setuptools` (v71+) no longer inject `pkg_resources` into isolated build environments by default.

### Solution
Bypass build isolation for the failing package using the `--no-build-isolation-package` flag. For example:

```bash
pip install -e ".[isaaclab]" --no-build-isolation-package flatdict
```
