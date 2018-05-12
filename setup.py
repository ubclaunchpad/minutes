from setuptools import setup

# TODO: Make sure this works with conda.
setup(name='minutes',
      version='0.0.1',
      description='Speaker Diarization Library',
      author='UBC Launchpad',
      author_email='team@ubclauncpad.com',
      url='https://www.ubclaunchpad.com',
      packages=['minutes'],
      install_requires=[
        'keras==2.1.5',
        'numpy==1.13.3',
        'scikit-learn==0.19.1',
        'scipy==0.19.1',
        'librosa==0.6.0',
        'matplotlib==2.2.0'
      ]
      )
