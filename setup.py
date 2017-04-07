try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

config = {
    'description': 'RPV Analysis',
    'author': 'Brian Amadio',
    'url': '',
    'download_url': '',
    'author_email': 'btamadio@gmail.com',
    'version': '0.1',
    'install_requires': ['nose'],
    'packages': ['NAME'],
    'scripts': [],
    'name': 'projectname'
}

setup(**config)
