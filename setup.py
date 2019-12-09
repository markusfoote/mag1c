import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mag1c", 
    use_scm_version={
        'write_to': 'mag1c/version.py',
        'write_to_template': '__version__ = "{version}"',
        'fallback_version': '0.0.0-dev0'
    },
    setup_requires=['setuptools_scm'],
    author="Markus Foote",
    author_email="markusfoote@gmail.com",
    license='BSD',
    description="A Sparse Matched Filter Algorithm for Atmospheric Trace Gas Concentration Estimation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/markusfoote/mag1c",
    keywords='imaging spectrometer matched filter sparse pytorch',
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
        "Topic :: Scientific/Engineering :: Atmospheric Science",
        "Development Status :: 4 - Beta",  # "Development Status :: 5 - Production/Stable"
    ],
    python_requires='>=3.6',
    install_requires=[
        'torch>=1.3',
        'numpy>=1.16.4',
        'spectral>=0.19',
        'scikit-image>=0.15.0',
    ],
    package_data={'mag1c': ['ch4.lut', 'ch4.hdr']},
    entry_points={
        'console_scripts': [
            'mag1c = mag1c.mag1c:main',
            'sparsemf = mag1c.mag1c:main',
        ]
    }
)