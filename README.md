# Installation (conda)

```bash
git clone --single-branch --branch refactoring https://github.com/rte-france/relife.git
cd relife
```

```bash
conda create --name relife2 python=3.11
conda activate relife2
```

For users :

```bash
#in your environment
pip install .
```

For contributors, install the library in edidatable mode with extra dependencies labelled as ```test```, ```format```
and ```doc``` :

```bash
#in your environment
pip install -e .[test,format,doc]
```

# Contributions

Create a new branch locally

```commandline
git branch mybranch
git checkout mybranch
```

Run nox before commits

```bash
nox
```

It will download all necessary python librairies called by each nox session. To avoid that behaviour add the following
arguments to the command:

```bash
nox -rR
```

Note that ```nox``` command must be run at least one time to install each venv used by nox sessions.

If some reported errors are judged useless, ignore them or edit ```pyproject.toml``` file to ignore them each
time ```nox``` will be called.

Commit and when you're ready to share your branch with the world, push it on the remote ```origin``` :

```commandline
git push origin mybranch
```

At the end, the maintainer will fetch the branch and merge into the main branch if the work has been done correctly.

# Build Sphinx doc

```bash
#in your environment
sphinx-build -M html sphinx_docs/source/ sphinx_docs/build/ -Ea
python -m http.server --directory ./sphinx_docs/build/html/
```

Then go to : http://localhost:8000
