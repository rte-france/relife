For developpers
===============

.. note::

    The content of this section is under continuous improvement as the project gathers more contributors. It is highly inspired by `Scikit-Learn <https://scikit-learn.org/dev/developers/contributing.html#ways-to-contribute>`_ and `Scipy <https://scipy.github.io/devdocs/dev/index.html>`_ contributing documentation.
    We highly encourage new comers to read these documentations if ours do not answer their questions.

Local repository set up
-----------------------

Clone ReLife repository.

.. code-block::

    $ git clone git@github.com:YourLogin/rte-france/relife.git
    $ cd relife

We highly encourage you to use a python virtual environment. In your environment, intall ReLife in editable mode from the source with the developpers dependencies.

.. code-block::

    $ pip install -e ".[dev]"

Add the ``upstream`` remote to synchronize your local repo to the ReLife repo.

.. code-block::

    $ git remote add upstream git@github.com:rte-france/relife.git


Development workflow
--------------------

The development is done a specific branch called ``develop/<relife-version>``. From this development branch, open a new branch to develop your feature :

.. code-block::

    $ git checkout develop/<relife-version>
    $ git checkout -b my-new-feature

Modify, stag and commit changes. Where you're done, push the changes to your forked repo

.. code-block::

    $ git push origin my-new-feature

To be sure that you didn't miss any updates on `develop/<relife-version>`, regularly ``merge`` the ``upstream``.

.. code-block::

    $ git checkout develop/<relife-version>
    $ git fetch upstream
    $ git merge upstream/develop/<relife-version>
    $ git checkout my-new-feature
    $ git merge develop/<relife-version>


Please follow and use these standard acronyms to start your commit messages :

.. code-block::

    BUG: bug fix
    DEP: deprecate something, or remove a deprecated object
    DEV: development tool or utility
    DOC: documentation
    ENH: enhancement
    MAINT: maintenance commit (refactoring, typos, etc.)
    REV: revert an earlier commit
    STY: style fix (PEP8, reformat, etc.)
    TYP: typing
    TEST: addition or modification of tests
    REL: related to releasing ReLife
    CI: related to CI


Opening a pull request
----------------------

If you want to submit your work, open a pull request to merge your ``origin/my-new-feature`` branch to ``upstream/develop/<relife-version>``.
Before a PR can be merged, it needs to be approved by 1 core developer.
An incomplete contribution -- where you expect to do more work before receiving
a full review -- should be marked as a `draft pull request
<https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/changing-the-stage-of-a-pull-request>`_
and changed to "ready for review" when it matures.

In order to ease the reviewing process, we recommend that your contribution
complies with the following rules before marking a PR as "ready for review". The
**bolded** ones are especially important:

1. **Give your pull request a helpful title** that summarizes what your
contribution does. This title will often become the commit message once
merged so it should summarize your contribution for posterity. In some
cases "Fix <ISSUE TITLE>" is enough. "Fix #<ISSUE NUMBER>" is never a
good title.

2. **Do not include unecessary/unjustified commits** that don't have direct relations
with the issue. Do not include unecessary lines rewrittings or modifications of code style
unless it is clearly motivated. Also, remove personal comments in your code.

3. **Make sure to address the whole issue**. Don't propose code that resolves the issue partly. If case
you have any doubt, discuss it directly in the corresponding issue post.

4. **Make sure your code passes the tests**. The whole test suite can be run
with ``pytest``, but it is usually not recommended since it takes a long
time. It is often enough to only run the test related to your changes:
for example, if you changed something in ``relife/lifetime_model/conditional_model.py``, running the following commands will usually be enough:

    - ``pytest relife/lifetime_model/tests/test_conditional_model.py`` to run the tests specific to the file
    - ``pytest relife/lifetime_model`` to test the whole ``relife.lifetime_model`` module

5. **Make sure your code is properly commented and documented**, and **make
sure the documentation renders properly**. To build the documentation, please
refer to our Documentation style guidelines. Run Sphinx locally before.

6. Typing your code is not necessary but make sure it is clear enough. The typing is handled by
the core team of ReLife. More specifically, it is a work in progress that will be done through
stubfiles gradually.

About conception
----------------

If you want to adress conception problem, you're welcome. **But these issues must be carefully motivated and well justified**. More precisely, we
won't accept any modifications that would be too subjective, e.g. *because you think it is more readable*.

.. warning::

    We are aware that overall ReLife code base *design* can be improved. Especially, we are currently having a special care on typing and are *stubifying* the code base.
    This work is done in addition to feature enhancements and progress at its own pace. It will be tested against mypy and it complements development principles that we study.
    At the end, we expect this static type checking will get the overall code base quality in the right direction.


About the documentation
-----------------------

To build the documentation locally run the following commands :

.. code-block::

    $ cd doc
    $ make html

To run a local documentation server and consult your documentation version :

.. code-block::

    $ cd doc
    $ python -m http.server -d build/html/

Then go to `http://localhost:8000 <http://localhost:8000>`_

.. warning::

    Using Sphinx for the first time is frustrating. It is easy to get tons of errors as they are cumulative. Read the Sphinx
    documentation and **be carefull** about unwanted blank lines or missing spaces in reST directives.

The documentation is built with `PyData Sphinx Theme <https://pydata-sphinx-theme.readthedocs.io/en/stable/>`_ and uses the
NumPy documentation style. Here are important points to have in mind if you want to **contribute to the documentation** :

* Read the `NumPy documentation style guide <https://numpydoc.readthedocs.io/en/latest/format.html>`_
* Classes documentation .rst files are generated using `Jinja2 <https://jinja.palletsprojects.com/en/stable/>`_ template engine. The template is written in ``class_template.rst`` of the ``doc/source/_templates``. A guide to Jinja2 templating is given in `Sphinx autosummary documentation <https://www.sphinx-doc.org/en/master/usage/extensions/autosummary.html>`_. Basically, one can catch usefull variables like ``methods`` and create nice autosummary tables inside an ``autoclass``. We chose this style because it creates very clean and comprehension interface documentation of the class.
* Take a special care to attributes class documentation. Sphinx does not handle attribute instances easily, especially when they are **inherited**. One must reference them manually in the object class under the ``Attributes`` field of the docstring.  As it is mentionned in `NumPy documentation style guide <https://numpydoc.readthedocs.io/en/latest/format.html>`_, property methods (getter and/or setter) can be listed there. Their attached docstring will be loaded automatically. One more thing, some IDE (like PyCharm) may raise warnings about unreferenced variables. It is a bug... ignore or disable it at the statement level.
