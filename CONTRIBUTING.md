# Contributing

## Installation 📦️

Set up a virtual Python environment, version 3.12* or 3.13*.

Install the project with its development dependencies.

```console
$ pip install -e .[dev]
```

Set up the pre-commit git hook (for normalizing formatting) using the `pre-commit` library, which is automatically installed in the above command.

```console
$ pre-commit install
pre-commit installed at .git/hooks/pre-commit
```

## Testing 🧪

Testing can be run with `pytest`, which is installed as a development dependency. Thanks to the project's configuration ([`pyproject.toml`](./pyproject.toml)), `pytest` will search for test modules in the directories [`src/`](./src/) and [`tests/`](./tests/), and it will run [doctests](https://docs.python.org/3/library/doctest.html) if you write them in Python modules.

```console
$ pytest
=========================== test session starts ============================
platform linux -- Python 3.12.3, pytest-8.3.5, pluggy-1.6.0
rootdir: /###/simetree
configfile: pyproject.toml
testpaths: tests, src
collected 15 items

tests/e2e/pymc_test.py .s                                                                                                        [ 13%]
tests/e2e/sbi_test.py .

```

- `tests/`

    - `e2e/` : End-to-End tests, meant to test the full workflow of certain model configurations.

    - `unit/` : Unit tests, meant to efficiently test code units' functionality.

    - `mock/` : Directory for storing mock data, meant to let us skip time-consuming methods / functions that prepare the tested methods / functions in unit tests.


## Formatting ✨

Formatting is normalized with the `isort` and `ruff` libraries, which are installed in the development dependency group, and continuously integrated with a `pre-commit` git hook.

You can run `isort` and `ruff` whenever you want by calling the installed libraries' CLIs.

```console
$ isort src/
Fixing /###/simetree/src/utils/evaluation.py
Fixing /###/simetree/src/models/__init__.py
```

They will also be run automatically whenever you make a `git commit`.

```console
$ git commit -m "my commit"
ruff check...............................................................Passed
ruff format..............................................................Passed
isort....................................................................Passed
uv-lock..................................................................Passed
[master f82454b] my commit
 5 files changed, 23 insertions(+), 7 deletions(-)
```
