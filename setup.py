# coding=utf-8
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Setup script for NAMs."""

import pathlib
from setuptools import find_packages
from setuptools import setup

here = pathlib.Path(__file__).parent.resolve()

long_description = (here / 'README.md').read_text(encoding='utf-8')

install_requires = [
    "torch==1.7.0",
    "fsspec==0.8.4",
    "pandas==1.1.4",
    "tqdm==4.54.0",
    "sklearn==0.0",
    "absl-py==0.11.0",
    "gcsfs==0.7.1",
]

nam_description = ('Neural Additive Models (PyTorch): Intepretable ML with Neural Nets')

setup(
    name='nam-pt',
    version=0.3,
    description=nam_description,
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/kherud/neural-additive-models-pt',
    author='Konstantin Herud',
    classifiers=[
        'Development Status :: 4 - Beta',

        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',

        'License :: OSI Approved :: Apache Software License',

        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3 :: Only',

        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',

    ],
    keywords='nam, interpretability, machine, learning, research',
    include_package_data=True,
    packages=find_packages(exclude=['docs']),
    install_requires=install_requires,
    license='Apache 2.0',
)
