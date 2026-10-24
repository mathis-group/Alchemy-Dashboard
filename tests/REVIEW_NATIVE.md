# Reviewing the native simulation form

This draft depends on [Modern-AlChemy #31](https://github.com/mathis-group/Modern-AlChemy/pull/31).
Both normal dependency declarations in `pyproject.toml` still use upstream
`mathis-group/Modern-AlChemy@main`. That revision does not provide the recursive
Python API. A normal dependency sync is not yet a complete install for this form.

The following **opt-in review environment** explicitly installs the contributor
commit recorded in `tests/native-review.txt`. Review that commit before running it;
it is unmerged native code. The override does not change production dependencies.
Use Python 3.10+, a Rust toolchain, and uv:

```sh
uv venv .venv-review
uv pip install --python .venv-review/bin/python -e . --overrides tests/native-review.txt
.venv-review/bin/python -m unittest discover -s tests -v
.venv-review/bin/python -m alchemy_dashboard.main
```

On Windows, use `.venv-review/Scripts/python.exe`. Run the interpreter directly:
`uv run` can synchronize the normal upstream-main dependency again. After native
acceptance/merge, maintainers must choose the upstream dependency revision and
remove or update this temporary review override before treating the default install
as usable.

For browser checks, install Playwright in the review environment and its Chromium
browser, or set `CHROMIUM_PATH` to an existing Chromium executable:

```sh
uv pip install --python .venv-review/bin/python playwright
.venv-review/bin/python -m playwright install chromium
.venv-review/bin/python tests/browser_simulation.py -v
```

The browser tests enable Chromium's sandbox. If the environment cannot launch it,
report that browser check as blocked; do not disable the sandbox. Optionally set
`BROWSER_ARTIFACTS` to a directory to retain screenshots, server logs, the disposable
SQLite database, and application-file hashes. Tests use byte-identical application
files in a temporary directory so they cannot overwrite a local experiment database.

The form protects against overlapping submissions on the same page. This is not a
server-side idempotency key for multiple tabs or request replay. Native computation
and input validation precede persistence. Existing database helpers commit each
write separately, so a database failure during persistence can leave partial data;
these changes do not provide transaction-wide rollback.
