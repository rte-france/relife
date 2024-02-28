from setuptools import find_packages, setup

setup(
    name="relife2",
    version="1.0",
    include_package_data=True,
    install_requires=[
        line.strip() for line in open("requirements.txt").readlines()
    ],
    packages=find_packages(),
)
