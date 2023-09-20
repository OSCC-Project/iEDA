from setuptools import setup
from Cython.Build import cythonize

from distutils.extension import Extension
from Cython.Distutils import build_ext

extensions = [
    Extension("PyModel", sources=["PyModel.pyx"],
              extra_compile_args=["-O3"],
              language="c++",
              language_level=3)
]
setup(
    name='PyModel',
    cmdclass={"build_ext": build_ext},
    ext_modules=cythonize(extensions)
)
