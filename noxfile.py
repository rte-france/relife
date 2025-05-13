import nox


@nox.session(tags=["reformat"])
def black(session):
    session.install("black")
    session.run("black", "noxfile.py")
    session.run("black", "relife/.")


@nox.session(tags=["reformat"])
def isort(session):
    session.install("isort")
    session.run("isort", "relife/.")


@nox.session(tags=["lint"])
def lint(session):
    requirements = nox.project.load_toml("pyproject.toml")["project"]["dependencies"]
    session.install(*requirements)
    session.install("pylint")
    if session.posargs:
        args = session.posargs
    else:
        args = ()
    session.run("pylint", "--extension-pkg-whitelist=numpy", "src/relife/.", *args)


@nox.session(tags=["test"], python=["3.9", "3.10", "3.11", "3.12", "3.13"])
def test(session):
    requirements = nox.project.load_toml("pyproject.toml")["project"]["dependencies"]
    session.install(*requirements)
    session.install(".")  # install relife2
    session.install("pytest")
    if session.posargs:
        args = session.posargs
    else:
        args = ()
    session.run("pytest", "test/", *args)


@nox.session(tags=["parametrized_test"])
@nox.parametrize("numpy", ["2.0.0", "2.0.1", "2.0.2", "2.1.0", "2.1.1", "2.1.2", "2.1.3", "2.2.0"])
@nox.parametrize("scipy", ["1.13.0", "1.13.1", "1.14.0", "1.14.1", "1.15.0"])
@nox.parametrize("matplotlib", ["3.9", "3.10"])
def parametrized_test(session, numpy, scipy, matplotlib):
    session.install(".")  # install relife2
    session.install("pytest")
    # override installed versions by relife2 dependencies by this version
    session.install(f"numpy=={numpy}")
    session.install(f"scipy=={scipy}")
    session.install(f"matplotlib=={matplotlib}")
    if session.posargs:
        args = session.posargs
    else:
        args = ()
    session.run("pytest", "test/", *args)


@nox.session(tags=["type"])
def mypy(session):
    requirements = nox.project.load_toml("pyproject.toml")["project"]["dependencies"]
    session.install(*requirements)
    session.install("mypy")
    if session.posargs:
        args = session.posargs
    else:
        args = ()
    session.run("mypy", *args, "relife/.")
