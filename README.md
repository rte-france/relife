# Installation

For users :

```bash
#in your environment
pip install .
```

For contributors :

```bash
#in your environment
pip install -e .[test,format,doc]
```

# Build Sphinx doc (temporary)

```bash
#in your environment
sphinx-build -M html sphinx_docs/source/ sphinx_docs/build/ -Ea
python -m http.server --directory ./sphinx_docs/build/html/
```

Then go to : http://localhost:8000
