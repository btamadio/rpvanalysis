try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

config = {
    'name':'rpvanalysis',
    'description': 'RPV Analysis',
    'author': 'Brian Amadio',
    'url': '',
    'download_url': '',
    'author_email': 'btamadio@gmail.com',
    'version': '0.1',
    'install_requires': ['nose','pandas','numpy','numba','matplotlib'],
    'packages': ['rpvanalysis'],
    'scripts': [],
    'test_suite':'nose.collector',
    'tests_require':['nose'],
    'scripts':['bin/convert','bin/run-2jet-analysis']
}

setup(**config)
