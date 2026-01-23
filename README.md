<h1 align="center">
  <img src="doc/source/_static/small_relife.gif" />
</h1></br>

An open-source Python library for asset management

- **Documentation:** http://opensource.rte-france.com/relife/
- **Source code:** https://github.com/rte-france/relife
- **Contributing:** http://opensource.rte-france.com/relife/developper/index.html
- **Bug reports:** https://github.com/rte-france/relife/issues

Some explanations might be missing until we finish the documentation properly, so do not hesitate to open an issue.

## Installation

**From PyPi**

```bash
$ pip install relife
```

**From source**

```bash
$ git clone https://github.com/rte-france/relife.git
$ cd relife
$ pip install .
```

**For developpers**

```bash
$ git clone https://github.com/rte-france/relife.git
$ cd relife
$ pip install -e ".[dev]"
```

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
