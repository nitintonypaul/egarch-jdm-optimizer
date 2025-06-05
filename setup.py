from setuptools import setup, Extension
import pybind11

ext_modules = [
    Extension(
        "egarch",
        ["cpp/egarch.cpp"],
        include_dirs=[pybind11.get_include()],
        language="c++"
    ),
    Extension(
        "merton",
        ["cpp/merton.cpp"],
        include_dirs=[pybind11.get_include()],
        language="c++"
    ),
]

setup(
    name="models",
    version="0.1",
    ext_modules=ext_modules,
)
