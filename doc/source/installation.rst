Installation
============

ReLife is a Python package. It is uploaded on the `Python Package Index (PyPi) <https://pypi.org/>`_. Before you install ReLife, make
sure Python **3.11 (or newer)** is installed.

Install ReLife with `pip <https://packaging.python.org/en/latest/key_projects/#pip>`_ :

.. code-block::

    $ python -m pip install relife

**Optional but recommended :** create and activate a virtual environment before.

For Linux users :

.. code-block::

    $ /usr/bin/python3.** -m venv <venv_location>/relife
    $ source <venv_location>/relife/bin/activate

For Windows users :

.. code-block::

    $ py -3.** -m venv <venv_location>\relife
    $ .\<venv_location>\relife\Scripts\activate

**From source :** to install ReLife from source, go to the `ReLife repository <https://github.com/rte-france/relife>`_. Clone the codebase and install ReLife with `pip <https://packaging.python.org/en/latest/key_projects/#pip>`_.

.. code-block::

    $ git clone https://github.com/rte-france/relife.git
    $ cd relife
    $ python -m pip install .

For contributors, optional development dependencies (type checker, documentation builder, etc.) are available. Use this command instead :

.. code-block::

    $ python -m pip install -e ".[dev]"