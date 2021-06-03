from setuptools import setup
import os

with open(
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 '../requirements.txt')) as f:
  requirements = f.read().splitlines()

setup(name='bfseg',
      version='0.0',
      install_requires=requirements,
      packages=['bfseg', 'bfseg.data', 'bfseg.data.nyu', 'bfseg.utils'])
