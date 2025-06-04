Installation
------------

.. contents::
    :local:

ReLife is a python package and it is uploaded on the `Python Package Index (PyPi) <https://pypi.org/>`_. To install ReLife locally make
sure :

* Python **3.11 (or newer)** is installed on you're machine. For Linux users with a Debian based distribution, an appropriate version of Python is already installed if your OS is up-to-date. For Windows users, please go to `python.org <https://www.python.org/>`_, read the documentation and install the appropriate Python version.
* You created a new python virtual environment (e.g, relife). For Linux users, you may need to install ``python3.**-venv`` (where ``**`` corresponds to your python version). If you don't know what a Python virtual environment is, please go to `docs.python.org <https://docs.python.org/>`_, read and learn. You're about to use a Python package so you must know the very basics of Python, including virtual environments.

**Linux users (Debian based)**

Create and activate the virtual environment :

.. code-block::

    /usr/bin/python3.** -m venv <venv_location>/relife
    source <venv_location>/relife/bin/activate

Install ReLife with `pip <https://packaging.python.org/en/latest/key_projects/#pip>`_ :

.. code-block::

    pip install relife

**Windows users**

Create and activate the virtual environment :

.. code-block::

    py -3.** -m venv <venv_location>\relife
    .\<venv_location>\relife\Scripts\activate

Install ReLife with `pip <https://packaging.python.org/en/latest/key_projects/#pip>`_:

.. code-block::

    pip install relife

Installation from source
------------------------

To install ReLife from source, go to `relife repository <https://github.com/rte-france/relife>`_. Clone the codebase and install ReLife with `pip <https://packaging.python.org/en/latest/key_projects/#pip>`_.

.. code-block::

    git clone https://github.com/rte-france/relife.git
    cd relife
    pip install .

As a contributor, you may want to install optional dependencies (type checking, documentation builder, etc.). You can use this command instead :

.. code-block::

    pip install -e ".[dev]"