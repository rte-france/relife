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
sphinx-build -M html sphinx_docs/source/ sphinx_docs/build/
python -m http.server
```

Then go to : http://localhost:8000/sphinx_docs/build/html/index.html

# Minimal documentation

- [Quick start](./docs/quick_start.md)
- [ReLife in details](./docs/details.md)
- [How to contribute to ReLife ?](HOW_TO_CONTRIBUTE.md)