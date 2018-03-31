try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


def requirements():
    with open('requirements.txt') as f:
        install_requires = [line.strip() for line in f]

    return install_requires

setup(
    name='Minutes',
    url='https://github.com/ubclaunchpad/minutes',
    packages=['minutes'],
    install_requires=requirements(),
    classifiers=[],
    include_package_data=True,
    zip_safe=False,
)
