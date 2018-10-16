#!/usr/bin/python
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='bilm',
    version='0.1.post5',
    url='http://github.com/allenai/bilm-tf',
    packages=setuptools.find_packages(),
    tests_require=[],
    zip_safe=False,
    entry_points='',
    description='Tensorflow implementation of contextualized word representations from bi-directional language models',
    long_description=long_description,
    long_description_content_type="text/markdown",
    license='Apache License 2.0',
    python_requires='>=3.5',
    install_requires=[
        'h5py',
    ],
    classifiers=[
        'License :: OSI Approved :: Apache Software License',  
        'Operating System :: OS Independent',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Text Processing',
    ],
    keywords='bilm elmo nlp embedding',
    author='Matthew E. Peters, Mark Neumann, Mohit Iyyer, Matt Gardner, Christopher Clark, Kenton Lee, Luke Zettlemoyer',
    author_email='allennlp-contact@allenai.org',
    maintainer='Matthew Peters',
)

