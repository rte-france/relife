Developement environment
========================

Advices about development environments, set ups to do before contributing to ReLife


Build system
~~~~~~~~~~~~

According to this https://packaging.python.org/en/latest/guides/tool-recommendations/#build-backends, we chose
Setuptools to be our build system. Poetry is another popular project but does support C/C++ backend natively and
does not use standard project.toml template.


CI
~~~

Our CI tool is Nox.

.. list-table::
   :widths: 25 75

   * - ``nox -l``
     - List available nox sessions
   * - ``nox``
     - Run all nox sessions at once and install specific venv in ``.nox/``
   * - ``nox -rR``
     - Same as ``nox`` but without installing venv. It assumes that ``.nox/`` was already created and filled
   * - ``nox -t NAME``
     - Run only the session called ``NAME``
