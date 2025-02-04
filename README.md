# ReLife

:warning: **Read this:** The repository is currently undergoing active code refactoring. A future release will be available soon.

## Why?

The first version of ReLife was not well maintained and had limited contribution possibilities. We do not expect this new release to be perfect, but we are doing our best.

The future version will include cleaner documentation and better tutorials. The code will also adopt a more structured OOP style while adhering to typing theory as much as possible. We will also explain our use of certain design patterns.

If you plan to use ReLife extensively, we would greatly appreciate your feedback. Do not hesitate to open an issue.

## Installation (stable)

The project package uploaded to PyPi has not changed, meaning the latest version (v1.0.0) published is the same.

```bash
pip install relife
```

## Installation (v2.0.0)

The new version of ReLife can be built from source. You'll need to clone this repository and install ReLife with pip in
your python environment (**we highly encourage to install ReLife in a python virtual environment**)

```
source <path_to_your_venv>/bin/activate
mkdir -p <your_desired_path>/relife
git clone https://github.com/rte-france/relife.git
pip install . # or alternatively pip install ".[dev]" adds required dependencies to contribute to ReLife
```

## Documentation

:warning: The documentation has been rebuilt and is quite similar to the old one.

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
The logo was made by [Freepik](https://www.freepik.com) from [Flaticon](<https://www.flaticon.com>)