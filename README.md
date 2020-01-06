# MAG1C:  Matched filter with Albedo correction and reweiGhted L1 sparsity Code

[![Article DOI:10.NNN/xxxx.xxxx](https://img.shields.io/badge/Article%20DOI-10.NNN%2Fxxxx.xxxx-blue)](https://doi.org) ![GitHub release (latest SemVer including pre-releases)](https://img.shields.io/github/v/release/markusfoote/mag1c?include_prereleases&sort=semver) ![PyPI](https://img.shields.io/pypi/v/mag1c) ![PyPI - Python Version](https://img.shields.io/pypi/pyversions/mag1c) ![PyPI - License](https://img.shields.io/pypi/l/mag1c)

Fast concentration estimation and detection of trace gas absorption from imaging spectrometer data.

<!--
## Citation
If you use this tool in a program or publication, please acknowledge its author(s) by adding the following reference:

<citation here>
-->
## Installation
``pip install mag1c``

## Requirements
mag1c depends on these software packages for math routines and data I/O. 

Python 3.6 (or newer) and the following python packages and versions:
- [`numpy`](https://www.numpy.org/)
- [`spectral`](https://www.spectralpython.net/)
- [`torch`](https://pytorch.org) 1.3+
- [`scikit-image`](https://scikit-image.org/)

## GPU Processing
If available, this code uses a compatible GPU for accelerated computation. See [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/) for details on how to install pytorch with gpu support for your system. You will then need to install the other dependencies.

The `--gpu` flag must be used to enable GPU acceleration.
### CPU-Only
If you know that you will **not** use a GPU, you can install the CPU-only version of pytorch. See [https://pytorch.org/get-started/locally/#no-cuda-1](https://pytorch.org/get-started/locally/#no-cuda-1) for how to install the CPU-only version. At time of writing, the install command for cpu-only torch and mag1c together through pip is: 
```
pip3 install mag1c torch==1.3.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
```

## Advanced Installation
The simplest way to obtain this program is through `pip`. To get the latest release:
```bash
pip install mag1c
```
or, to install a specific released version:
```
pip install magic==x.y.z
```
or, to get a specific point in history directly from github:
```
pip install git+https://github.com/markusfoote/mag1c@tag#egg=mag1c
```
where ``tag`` is any tag (e.g. ``v1.2.0``), branch name (e.g. ``master``) or commit hash. [PyPA has more detailed instructions.](https://pip.pypa.io/en/stable/reference/pip_install/#vcs-support)

This will install required dependencies (most notably, [pytorch](https://pytorch.org)) on linux systems. For Windows, or specific installation flavors of pytorch (like CPU-only), follow your choice of instructions on [PyTorch's website](https://pytorch.org/get-started/locally/), then install mag1c. Mag1c is compatible with PyTorch installed through `conda`, just make sure you are using the **environment's** `pip` to install mag1c, and activate the conda environment whenever you wish to run `mag1c`.

## Usage
### Entrypoints
This program can be invoked in multiple ways: 
1. `python /path/to/mag1c.py` works whenever you have a copy of the `mag1c.py` script. The versioning metadata may not work well with a standalone script.
2. `python -m mag1c` works when you install the python package (i.e. via `pip`).
3. `mag1c` is a direct entry point to the program when you install via `pip`.
4. `sparsemf` is exactly the same as `mag1c`, just with a debatably-more-readable name.

### Runtime Options
There are numerous options/flags that can be provided to modify processing behavior. Run `mag1c --help` for a full description of the available arguments.

## Examples
### Process a single file with defaults:
```bash
python mag1c.py /my/radiance --spec /my/target --out /some/output
```

### Process a single file with bash variables and some custom options, including GPU:
```bash
export CUDA_VISIBLE_DEVICES=0 # Restrict processing to the first GPU in the system
RDNFILE="/path/to/my/radiance data with spaces in filename"
TEMPLATE=/path/to/my/templatespectrum
OUTPUT=/path/to/outputfile_date_time_iteration25_grouping5
GLT=/path/to/my/gltfile
python mag1c.py "$RDNFILE" --spec $TEMPLATE --out $OUTPUT --outputgeo $GLT --iter 25 --group 5 --gpu -t 2 -b16
```

### Process all files in a folder:
```bash
TEMPLATE=/path/to/template.txt
for f in /path/to/folder/ang*_rdn_*_clip; do \
    python mag1c.py "${f}" "${TEMPLATE}" "/output/folder/$(basename "${f/rdn/mag1c}")" --iter 20
done;
```

### Process a file with detector saturation detection/masking:
For a geocorrected file:
```bash
sparsemf ${RDN_FILE} \
  --out $OUTPUTFOLDER$(basename ${b/_rdn_/_ch4_cmfr_}) \
  --geo ${RDN_FILE/img/glt} \
  --group 1 \
  --saturation \
  --saturationthreshold 6.0 \
  --maskgrowradius 150m \
  --mingrowarea 5 \
  --hfdi \
  --threads 8 \
  --gpu \
  --no-albedo-output \
  --visible-mask-growing-threshold 9.0
```
or for a non-geocorrected file:
```bash
sparsemf ${RDN_FILE} \
  --out $OUTPUTFOLDER$(basename ${b/_rdn_/_ch4_cmfr_}) \
  --outputgeo ${RDN_FILE/img/glt} \
  --group 1 \
  --saturation \
  --saturationthreshold 6.0 \
  --maskgrowradius 12px \
  --mingrowarea 5 \
  --hfdi \
  --threads 8 \
  --gpu \
  --no-albedo-output \
  --visible-mask-growing-threshold 9.0
```
Notice that the non-geocorrected file requires a **`maskgrowradius` in pixels**, as the file has no spatial metadata.