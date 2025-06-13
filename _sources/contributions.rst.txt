Contributors guide
==================

.. contents::
    :local:

Documentation style
-------------------

The following documentation is built with `PyData Sphinx Theme <https://pydata-sphinx-theme.readthedocs.io/en/stable/>`_ and uses the
NumPy documentation style. Here are important points to have in mind if you want to **contribute to the documentation** :

* Read the `NumPy documentation style guide <https://numpydoc.readthedocs.io/en/latest/format.html>`_
* Classes documentation .rst files are generated using `Jinja2 <https://jinja.palletsprojects.com/en/stable/>`_ template engine. The template is written in ``default_class_templates.rst`` of the ``doc/source/_templates``. A guide to Jinja2 templating is given in `Sphinx autosummary documentation <https://www.sphinx-doc.org/en/master/usage/extensions/autosummary.html>`_. Basically, one can catch usefull variables like ``methods`` and create nice autosummary tables inside an ``autoclass``. We chose this style because it creates very clean and comprehension interface documentation of the class.
* Take a special care to attributes class documentation. Sphinx does not handle attribute instances easily, especially when they are **inherited**. One must reference them manually in the object class under the ``Attributes`` field.  As it is mentionned in `NumPy documentation style guide <https://numpydoc.readthedocs.io/en/latest/format.html>`_, property methods (getter and/or setter) can be listed there. Their attached docstring will be loaded automatically. One more thing, some IDE (like PyCharm) may raise warnings about unreferenced variables. It is a bug... ignore it or disable them at the statement level.