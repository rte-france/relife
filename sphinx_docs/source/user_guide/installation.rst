Installation
============

Step 1 : build from source
--------------------------

.. code-block:: console

    git clone --single-branch --branch refactoring https://github.com/rte-france/relife.git
    cd relife
    

Step 2 : Create virtualenv
--------------------------

Unix/macOS
^^^^^^^^^^

Option 1 : conda
~~~~~~~~~~~~~~~~

.. code-block:: console

    conda create --name relife2 python=3.9
    conda activate relife2


Step 3 : install python module
------------------------------

.. code-block:: console

    pip install .


Step 4 (optional) : build documentation
---------------------------------------

.. code-block:: console

    pip install sphinx
    pip install myst-parser
    pip install sphinx-book-theme
    pip install sphinx-copybutton
    pip install sphinx-design
    sphinx-build -M html sphinx_docs/source/ sphinx_docs/build/ -Ea
    python -m http.server --directory ./sphinx_docs/build/html/

Then go to : `http://localhost:8000 <http://localhost:8000>`_