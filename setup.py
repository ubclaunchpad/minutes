from setuptools import setup

setup(
    name='minutes',
    version='0.0.1',
    description='Speaker diarisation via transfer learning',
    author='UBC Launch Pad',
    author_email='team@ubclaunchpad.com',
    url='https://www.ubclaunchpad.com',
    packages=['minutes'],
    install_requires=[
        'scipy==0.19.1',
        'keras==2.1.3',
        'tensorflow==1.6.0',
        'scikit-learn==0.19.1',
        'pysoundfile==0.9.0',
        'h5py==2.7.1'
    ]
)
