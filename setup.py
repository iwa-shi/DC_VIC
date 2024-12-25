import time
from setuptools import setup, find_packages

from distutils.core import setup, Extension

if __name__ == '__main__':
    setup(
        name='dcvic',
        version='0.0.1',
        packages=find_packages(include=['src']),
    )