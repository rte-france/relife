# Installation

```bash
#in your environment
pip install [-e] .
```

# Build Sphinx doc (temporary)

```bash
#in your environment
pip install sphinx
pip install myst-parser
pip install sphinx-book-theme
pip install sphinx-copybutton
pip install sphinx-design
sphinx-build -M html sphinx_docs/source/ sphinx_docs/build/
python -m http.server --directory ./sphinx_docs/build/html/ -E
```

Then go to : http://localhost:8000

# Minimal documentation

- [Quick start](./docs/quick_start.md)
- [ReLife in details](./docs/details.md)
- [How to contribute to ReLife ?](HOW_TO_CONTRIBUTE.md)