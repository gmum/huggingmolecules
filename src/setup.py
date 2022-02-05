import io
import os
import re

from setuptools import setup

# Get the version from huggingmolecules/__init__.py
# Adapted from https://stackoverflow.com/a/39671214
this_directory = os.path.dirname(os.path.realpath(__file__))
init_path = os.path.join(this_directory, 'huggingmolecules', '__init__.py')
version_matches = re.search(
    r'__version__\s*=\s*[\'"]([^\'"]*)[\'"]',
    io.open(init_path, encoding='utf_8_sig').read(),
)
if version_matches is None:
    raise Exception('Could not determine huggingmolecules version from __init__.py')
__version__ = version_matches.group(1)

setup(
    name='huggingmolecules',
    version=__version__,
    packages=['huggingmolecules'],
    install_requires=[
        'torch>=1.7.0',
        'scikit-learn>=0.23.2',
        'filelock>=3.0.12',
        'gdown>=3.12.2'
    ]
)
