from setuptools import setup
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='alpha_project',
    version='1.0.0',
    description='Project',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Fedyushin I.A.',
    author_email='hechachi1997@gmail.com',
    keywords='deduplication python machine_learning python3 data',
    packages=[
        'alpha_project',
        'dadata',
        'dadata.plugins'
    ],
    # dependency_links=['http://github.com/tigrus/dadata-python/tarball/master#egg=dadata'],
    include_package_data=True,
    install_requires=[
        'pandas >= 0.22.0',
        'numpy >= 1.8.0',
        'fuzzywuzzy >= 0.16.0',
        'scikit-learn >= 0.19.0',
        'nltk >= 3.2.5',
        'requests >= 2.18.4',
        'gensim >= 3.4.0',
        'datasketch >= 1.2.5',
    ],
    python_requires='>=3.5',
    zip_safe=False
)
