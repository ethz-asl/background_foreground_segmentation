from setuptools import setup

setup(name='bfseg',
      version='0.0',
      install_requries=[
          'yapf', 'pylint', 'tensorflow==2.3.1', 'tensorflow-datasets',
          'segmentation-models', 'Pillow', 'scikit-image', 'matplotlib'
      ],
      packages=['bfseg', 'bfseg.data', 'bfseg.data.nyu', 'bfseg.data.meshdist','bfseg.utils'])
