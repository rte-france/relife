<p align="center">
  <img src="doc/source/_static/small_relife.gif" />
</p>

## Documentation

It is available at https://rte-france.github.io/relife/.
Some explanations might be missing until we finish it properly, so do not hesitate to open an issue.

## Installation (PyPi)

```bash
pip install relife
```
**Developper installation**

Just change the pip installation command by (notice the usage of the editable mode) :

```
pip install -e ".[dev]"
```

## Installation (source)

To install the current codebase 

```bash
git clone https://github.com/rte-france/relife.git
cd relife
pip install .
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

The documentation uses [pydata-sphinx-theme](https://github.com/pydata/pydata-sphinx-theme) project. Original license
is [here](doc/LICENSE.txt)
