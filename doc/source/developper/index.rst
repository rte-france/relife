For developpers
===============

.. note::

    The content of this section is under continuous improvement as the project gathers
    more contributors. It is highly inspired by `Scikit-Learn <https://scikit-learn.org/dev/developers/contributing.html#ways-to-contribute>`_ 
    and `Scipy <https://scipy.github.io/devdocs/dev/index.html>`_ contributing
    documentation. We highly encourage new comers to read these documentations if ours
    do not answer their questions.

Local set up
------------

1. Fork ReLife repository

First, you need to `create an account <https://github.com/join>`_ on GitHub
(if you do not already have one) and fork the project repository by clicking on the
‘Fork’ button near the top of the page. This creates a copy of the code under your 
Github account. For more details on how to fork a repository see 
`this guide <https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo>`_.
It explains how to set up a local clone of your forked git repository.

2. Set up a local clone of your fork

Clone your fork of the ReLife repo from your GitHub account to your
local disk:

.. code-block::

  $ git clone https://github.com/<YourLogin>/relife.git  # add --depth 1 if your connection is slow

and change into that directory:

.. code-block::

  $ cd relife

.. _upstream:

Next, add the ``upstream`` remote. This saves a reference to the main ReLife
repository, which you can use to keep your repository synchronized with the latest
changes (you'll need this later in the :ref:`development_workflow`):

.. code-block::

  $ git remote add upstream https://github.com/scikit-learn/scikit-learn.git

Check that the `upstream` and `origin` remote aliases are configured correctly
by running:

.. code-block::

  $ git remote -v

This should display:

.. code-block:: text

  origin    https://github.com/YourLogin/scikit-learn.git (fetch)
  origin    https://github.com/YourLogin/scikit-learn.git (push)
  upstream  https://github.com/scikit-learn/scikit-learn.git (fetch)
  upstream  https://github.com/scikit-learn/scikit-learn.git (push)

3. ReLife local installation

Make sure Python ``3.11+`` is installed on your machine. Using this Python, create
a `Python virtual environment <https://docs.python.org/3/library/venv.html>`_ with
the name of your choice. **Activate your virtual environment**. Intall ReLife in
`editable mode <https://setuptools.pypa.io/en/latest/userguide/development_mode.html>`_
from the source code with the developpers dependencies.

.. code-block::

    $ cd relife
    $ pip install -e ".[dev]"


Additional Python installations
-------------------------------

The last command above should have installed the following dependencies:

- `Black <https://github.com/psf/black>`_ and `Isort <https://github.com/PyCQA/isort>`_.
  These tools allow you to automatically `format your code <https://en.wikipedia.org/wiki/Pretty-printing#Formatting_of_program_source_code>`_.
  Ensure that your IDE is configured to call them to format on save.
- `Flake8 <https://github.com/PyCQA/flake8>`_ and `Pylint <https://github.com/PyCQA/pylint>`_.
  These programs are Python `linters <https://en.wikipedia.org/wiki/Lint_(software)>`_.
  Ensure that your IDE captures diagnostics from these tools while you are coding.
  Although their default configurations can be quite aggressive, we refrain from
  providing generic configurations as it may overlook critical errors in some cases.
  Aim to resolve all warnings; if necessary, disable warnings locally by adding
  specific configurations for these tools on your machine. Do not commit these configurations,
  as they remain personal to your environment.
- `Pyright <https://github.com/microsoft/pyright>`_. This is a `static type checker <https://en.wikipedia.org/wiki/Type_system#Type_checking>`_.
  Again, ensure that your IDE communicates with the Pyright language server (`LSP <https://en.wikipedia.org/wiki/Language_Server_Protocol>`_)
  to receive feedback on your type annotations. Type checking can be challenging and may
  not be desired at first, so consider flagging your modules with ``#pyright: basic`` to start.
  Once you feel comfortable, gradually enhance your type annotations by removing ``#pyright: basic``
  and enabling strict mode in Pyright configurations.
- After successfully passing all Pyright analyses, use `Mypy <https://github.com/python/mypy>`_
  to validate or supplement the diagnostics provided by Pyright.

.. _development_workflow:

Development workflow
--------------------

The next steps describe the process of modifying code and submitting a PR:

1. Synchronize your ``main`` branch with the ``upstream/main`` branch,
   more details on `GitHub Docs <https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/syncing-a-fork>`_:

.. code-block:: bash

  $ git checkout main
  $ git fetch upstream
  $ git merge upstream/main

2. Create a feature branch to hold your development changes:

.. code-block:: bash

  $ git checkout -b my_feature

and start making changes. Always use a feature branch. It's good
practice to never work on the ``main`` branch!

3. Develop the feature on your feature branch on your computer, using Git to
   do the version control. When you're done editing, add changed files using
   ``git add`` and then ``git commit``:

.. code-block:: bash

  $ git add modified_files
  $ git commit

You will be prompted to enter a commit message. Please, as much as possible,
start your message with the dedicated :ref:`commit_markers`. Your message must
look like this

.. code-block:: text
  
  <commit_marker>: commit title

  commit description

Then push the changes to your GitHub account with:

.. code-block:: bash

   $ git push -u origin my_feature

4. Before opening a pull request, please verify that your work meets our :ref:`pull_requests_checklist`.

5. Follow `these <https://help.github.com/articles/creating-a-pull-request-from-a-fork>`_
   instructions to create a pull request from your fork. This will send a
   notification to potential reviewers.

It is often helpful to keep your local feature branch synchronized with the
latest changes of the main ReLife repository:

.. code-block:: bash

    $ git fetch upstream
    $ git merge upstream/main

Subsequently, you might need to solve the conflicts. You can refer to the
`Git documentation related to resolving merge conflict using the command
line
<https://help.github.com/articles/resolving-a-merge-conflict-using-the-command-line/>`_
or the `Git documentation itself (Basic Merge Conflicts) <https://git-scm.com/book/en/v2/Git-Branching-Basic-Branching-and-Merging>`_

.. note::

   One very helping tool to manage git command is `Lazygit <https://github.com/jesseduffield/lazygit>`_.
   It comes with a very user-friendly TUI and preconfigured set of usefull commands to manage
   commits and branches.

.. _commit_markers:

Commit message markers
----------------------

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

.. _pull_requests_checklist:

Pull request checklist
----------------------

Before a pull request can be merged, it needs to be approved by 1 core developer.
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

3. **Make sure to address the whole issue**. Don't propose code that resolves the issue partly.
If case you have any doubt, discuss it directly in the corresponding issue post.

4. **Make sure your code passes the tests**. The whole test suite can be run
with ``pytest``, but it is usually not recommended since it takes a long
time. It is often enough to only run the tests related to your changes:
for example, if you changed something in ``relife/lifetime_model/_conditional_model.py``,
running the following commands will usually be enough:

- ``pytest relife/lifetime_model/tests/test_conditional_model.py`` to run the tests specific to the file
- ``pytest relife/lifetime_model`` to test the whole ``relife.lifetime_model`` module

5. **Make sure your code is properly commented and documented**, and **make
sure the documentation renders properly**. To build the documentation locally, please
refer to :ref:`build_the_doc`.

6. Typing your code is not necessary at first, but make sure it is logic. The typing can be handled by
the core team of ReLife. It is still a work in progress.

About conception
----------------

If you want to adress conception problem, you're welcome.
**But these issues must be carefully motivated and well justified**. More precisely, we
won't accept any modifications that would be too subjective, 
e.g. *only because you think it is more readable*.

.. warning::

    We are aware that overall ReLife code base *design* can be improved. Especially, we
    are currently having a special care on typing and are *stubifying* the code base.
    This work is done in addition to feature enhancements and progress at its own pace.
    It will be tested against mypy and it complements development principles that we study.
    At the end, we expect this static type checking will get the overall code base
    quality in the right direction.


.. _build_the_doc:

Build the documentation
-----------------------

To build the documentation locally run the following commands :

.. code-block::

    $ cd doc
    $ make html

To run a local documentation server and read your built documentation :

.. code-block::

    $ cd doc
    $ python -m http.server -d build/html/

Then go to `http://localhost:8000 <http://localhost:8000>`_

.. warning::

    Using Sphinx for the first time is frustrating. It is easy to get tons of errors as
    they are cumulative. Read the Sphinx documentation and **be carefull** about
    unwanted blank lines or missing spaces in reST directives.

The documentation is built with `PyData Sphinx Theme <https://pydata-sphinx-theme.readthedocs.io/en/stable/>`_ and uses the
NumPy documentation style. Here are important points to have in mind if you want to **contribute to the documentation** :

* Read the `NumPy documentation style guide <https://numpydoc.readthedocs.io/en/latest/format.html>`_
* Classes documentation .rst files are generated using `Jinja2 <https://jinja.palletsprojects.com/en/stable/>`_ template engine.
  The template is written in ``class_template.rst`` of the ``doc/source/_templates``.
  A guide to Jinja2 templating is given in
  `Sphinx autosummary documentation <https://www.sphinx-doc.org/en/master/usage/extensions/autosummary.html>`_.
  Basically, one can catch usefull variables like ``methods`` and create nice autosummary
  tables inside an ``autoclass``. We chose this style because it creates very clean and
  comprehension interface documentation of the class.
* Take a special care to attributes class documentation. Sphinx does not handle
  attribute instances easily, especially when they are **inherited**. One must reference
  them manually in the object class under the ``Attributes`` field of the docstring.
  As it is mentionned in `NumPy documentation style guide <https://numpydoc.readthedocs.io/en/latest/format.html>`_,
  property methods (getter and/or setter) can be listed there. Their attached docstring
  will be loaded automatically. One more thing, some IDE (like PyCharm) may raise
  warnings about unreferenced variables. It is a bug... ignore or disable it at the statement level.
