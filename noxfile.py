import nox


@nox.session(tags=["style", "reformat"])
def black(session):
    session.install("black")
    session.run("black", "noxfile.py")
    session.run("black", "relife2/.")


@nox.session(tags=["style", "reformat"])
def isort(session):
    session.install("isort")
    session.run("isort", "relife2/.")


@nox.session(tags=["style"])
def lint(session):
    requirements = nox.project.load_toml("pyproject.toml")["project"]["dependencies"]
    session.install(*requirements)
    session.install("pylint")
    session.run("pylint", "--extension-pkg-whitelist=numpy", "relife2/.")


@nox.session(tags=["style"])
def mypy(session):
    requirements = nox.project.load_toml("pyproject.toml")["project"]["dependencies"]
    session.install(*requirements)
    session.install("mypy")
    session.run("mypy", "relife2/.")
