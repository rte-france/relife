Installation
============

ReLife is a python package and it is uploaded on the `Python Package Index (PyPi) <https://pypi.org/>`_. To install ReLife locally make
sure to have Python **3.11 (or newer)** installed on your machine.

**Optional but recommended :** create and activate a virtual environment.

For Linux users :

.. code-block::

    /usr/bin/python3.** -m venv <venv_location>/relife
    source <venv_location>/relife/bin/activate

For Windows users :

.. code-block::

    py -3.** -m venv <venv_location>\relife
    .\<venv_location>\relife\Scripts\activate

Install ReLife with `pip <https://packaging.python.org/en/latest/key_projects/#pip>`_ :

.. code-block::

    python -m pip install relife

**From source :** to install ReLife from source, go to `relife repository <https://github.com/rte-france/relife>`_. Clone the codebase and install ReLife with `pip <https://packaging.python.org/en/latest/key_projects/#pip>`_.

.. code-block::

    git clone https://github.com/rte-france/relife.git
    cd relife
    python -m pip install .

As a contributor, you'll need optional development dependencies (type checker, documentation builder, etc.). Use this command instead :

.. code-block::

    python -m pip install -e ".[dev]"