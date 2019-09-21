from os import path
from setuptools import setup, find_packages
try:  # for pip >= 10
    from pip._internal.req import parse_requirements
    from pip._internal.download import PipSession
except ImportError:  # for pip <= 9.0.3
    from pip.req import parse_requirements
    from pip.download import PipSession

__version__ = '0.1'

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), 'rb') as f:
    long_description = f.read().decode('utf-8')

lines = list(parse_requirements("requirements.txt", session=PipSession()))
install_requires = [str(l.req) for l in lines if l.original_link is None]

setup(
    name='od',
    author="Emilien Garreau",
    author_email="ppwwyyxxc@gmail.com",
    url="https://github.com/EmGarr/od",
    keywords="Tensorflow 2.0, deep learning, object detection",
    license="Apache",
    version=__version__,  # noqa
    description='Algorithm for Object  Detection using tensorflow.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    install_requires=install_requires,
)
