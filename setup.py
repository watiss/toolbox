from setuptools import setup, find_packages

setup(
    name="toolbox",
    version="0.1.0",
    description='Miscellaneous utilities',
    author='Valeh Valiollah Pour Amiri',
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "polars",
        "rich",
        "lightning",
    ],
)
