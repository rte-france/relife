import nox


@nox.session(tags=["style", "fix"])
def black(session):
    session.install("black")
    session.run("black", "noxfile.py")
    session.run("black", "relife2/.")


@nox.session(tags=["style", "fix"])
def isort(session):
    session.install("isort")
    session.run("isort", "relife2/.")


@nox.session(tags=["style"])
def lint(session):
    session.install("pycodestyle")
    session.run("pycodestyle", "relife2/.")


@nox.session(tags=["style"])
def mypy(session):
    session.install("mypy")
    session.run("mypy", "relife2/.")
