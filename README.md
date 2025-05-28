<p align="center">
  <img src="doc/source/_static/small_relife.gif" />
</p>

:warning: **Read this:** The repository is currently undergoing active code refactoring. A future release will be available soon.

## Installation (source)

The project package uploaded to PyPi has not changed. It means that the latest distribution package uploaded to PyPi is version 1.0.0.
The current codebase has changed and will be used for the next release v2.0.0. If ``pip install relife`` you will install version 1.0.0.
To install the current codebase 

```bash
source <path_to_your_venv>/bin/activate
mkdir -p <your_desired_path>/relife
git clone https://github.com/rte-france/relife.git
pip install .
```

**Developper installation**

Just change the pip installation command by (notice the usage of the editable mode) :

```
pip install -e ".[dev]"
```

## Documentation

It is available at https://rte-france.github.io/relife/.
Some explanations might be missing until we finish it properly, so do not hesitate to open an issue.

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
