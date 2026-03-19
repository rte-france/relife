<h1 align="center">
  <img src="doc/source/_static/small_relife.gif" />
</h1></br>

An open-source Python library for optimizing large-scale infrastructure investment decisions.

- **Documentation:** http://opensource.rte-france.com/relife/
- **Source code:** https://github.com/rte-france/relife
- **Contributing:** http://opensource.rte-france.com/relife/developper/index.html
- **Bug reports:** https://github.com/rte-france/relife/issues

Some explanations might be missing until we finish the documentation properly, so do not hesitate to open an issue.

## Installation

**From PyPi**

```bash
$ python -m pip install relife
```

**From source**

```bash
$ git clone https://github.com/rte-france/relife.git
$ cd relife
$ python -m pip install .
```

**For developpers**

The project has two dependency groups, `dev` and `doc`.

If you wish to work on the codebase, install ReLife with the packages included in the `dev` group.
Using the editable mode (`-e`) is recommanded.

```bash
$ git clone https://github.com/rte-france/relife.git
$ cd relife
$ python -m pip install -e . --group dev
```

If you wish to work on the documentation, install ReLife with the packages included in the `doc` group.

```bash
$ git clone https://github.com/rte-france/relife.git
$ cd relife
$ python -m pip install . --group doc
```

## Development tools

We use [ruff](https://docs.astral.sh/ruff/) as linter and formatter.
We currently use [basedpyright](https://docs.basedpyright.com/latest/) as Python type checker and LSP server. In a near future, we may move to [ty](https://docs.astral.sh/ty/).

## Citing

```
@misc{relife,
    author = {T. Guillon},
    title = {ReLife: a Python package for asset management based on reliability theory and lifetime data analysis.},
    year = {2022},
    journal = {GitHub},
    howpublished = {\url{https://github.com/rte-france/relife}},
}
```

## Credits

- The documentation uses [pydata-sphinx-theme](https://github.com/pydata/pydata-sphinx-theme) project. Original license
  is [here](doc/LICENSE.txt)
- Some parts of the documentation are highly inspired by [Scikit-learn](https://scikit-learn.org/stable/), [Scipy](https://scipy.org/) and [NumPy](https://numpy.org/)
