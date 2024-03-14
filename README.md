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
sphinx-build -M html sphinx_docs/source/ sphinx_docs/build/ -Ea
python -m http.server --directory ./sphinx_docs/build/html/
```

Then go to : http://localhost:8000
