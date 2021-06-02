import os
from io import open
from os import path
from typing import AnyStr, List

from setuptools import setup, find_packages

package_dir = path.abspath(path.dirname(__file__))


def _process_requirements() -> List[str]:
    requirements = []

    with open(path.join(package_dir, 'requirements.txt'), encoding='utf-8') as packages:
        packages = packages.read().strip().split('\n')

        for pkg in packages:
            if pkg.startswith('git+ssh'):
                return_code = os.system('pip install {}'.format(pkg))
                assert return_code == 0, 'error, status_code is: {}, exit!'.format(return_code)
            elif pkg.startswith('# '):
                continue
            else:
                requirements.append(pkg)

    return requirements


def _process_readme() -> AnyStr:
    with open(path.join(package_dir, 'README.md'), encoding='utf-8') as readme:
        long_description = readme.read()

    return long_description


setup(
    name='torecsys',
    version='0.0.1.dev2',
    license='MIT',
    description=(
        'ToR[e]cSys is a PyTorch Framework to implement recommendation system algorithms, '
        'including but not limited to click-through-rate (CTR) prediction, learning-to-ranking '
        '(LTR), and Matrix/Tensor Embedding. The project objective is to develop a ecosystem '
        'to experiment, share, reproduce, and deploy in real world in a smooth and easy way.'
    ),
    long_description=_process_readme(),
    long_description_content_type='text/markdown',
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent'
    ],
    keywords='recsys recommendersystem recommendationsystem machinelearning deeplearning research ctr clickthroughrate',
    python_requires='>=3.8',
    packages=find_packages(exclude=['contrib', 'docs', 'examples', 'tests']),
    install_requires=_process_requirements(),
    entry_points={
        'console_scripts': ['torecsys=torecsys.cli:main']
    },
    url='https://github.com/p768lwy3/torecsys',
    projects_url={
        'Bug Tracker': 'https://github.com/p768lwy3/torecsys/issues',
        'Documentation': 'https://torecsys.readthedocs.io/en/latest/',
        'Source': 'https://github.com/p768lwy3/torecsys'
    },
    author='Jasper Li',
    author_email='jasper_liwaiyin@protonmail.com'
)
