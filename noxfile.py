import nox


@nox.session(tags=["reformat"])
def black(session):
    session.install("black")
    session.run("black", "noxfile.py")
    session.run("black", "relife2/.")


@nox.session(tags=["reformat"])
def isort(session):
    session.install("isort")
    session.run("isort", "relife2/.")


@nox.session(tags=["lint"])
def lint(session):
    requirements = nox.project.load_toml("pyproject.toml")["project"]["dependencies"]
    session.install(*requirements)
    session.install("pylint")
    if session.posargs:
        args = session.posargs
    else:
        args = ()
    session.run("pylint", "--extension-pkg-whitelist=numpy", "relife2/.", *args)


@nox.session(tags=["test"])
def tests(session):
    requirements = nox.project.load_toml("pyproject.toml")["project"]["dependencies"]
    session.install(*requirements)
    session.install(".")  # install relife2
    session.install("pytest")
    if session.posargs:
        args = session.posargs
    else:
        args = ()
    session.run("pytest", "tests/", *args)


@nox.session(tags=["type"])
def mypy(session):
    requirements = nox.project.load_toml("pyproject.toml")["project"]["dependencies"]
    session.install(*requirements)
    session.install("mypy")
    if session.posargs:
        args = session.posargs
    else:
        args = ()
    session.run("mypy", *args, "relife2/.")
