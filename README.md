# MAG1C:  Matched filter with Albedo correction and reweiGhted L1 sparsity Code

## Introduction
mag1c is designed for fast detection and concentration estimation of trace gas from imaging spectrometer data.

<!-- This algorithm was published in *journal* on *date*. See the article at the publisher's site [here]().
If you use this algorithm or results from it in your own publication, please cite:
**citation here once published** -->

## Requirements
mag1c depends on these software packages for math routines and data I/O. 

Python 3.6 (or newer) and the following python packages and versions:
- [`numpy`](https://www.numpy.org/)
- [`spectral`](https://www.spectralpython.net/)
- [`torch`](https://pytorch.org) 1.1+
- [`scikit-image`](https://scikit-image.org/)
- [`scipy`](https://www.scipy.org/install.html) (optional: Only required if you want to load target spectra from `.mat` files.)

Read on for a step-by-step setup guide of how to satisfy these requirements if they are not already met.


## GPU Processing
If available, this code uses a compatible GPU for accelerated computation. See [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/) for details on how to install pytorch with gpu support for your system. You will then need to install the other dependencies.

You must use the option `--gpu` when running to enable GPU acceleration.

If you know that you will **not** use a GPU, you can install the CPU-only version of pytorch by specifying the `cpuonly` package to conda in addition to `pytorch`. See [https://pytorch.org/get-started/locally/#no-cuda-1](https://pytorch.org/get-started/locally/#no-cuda-1) for how to install the CPU-only version using pip.

## Setup
The simplest way to obtain this program is through `pip`. 
```bash
pip install mag1c
```
This will install required dependencies (most notably, [pytorch](https://pytorch.org) on linux systems. For Windows, or specific installation flavors of pytorch (like CPU-only), follow your choice of instructions on PyTorch's website, then install mag1c. Mag1c is compatible with PyTorch installed through `conda`, just make sure you are using the environment's `pip` to install mag1c, and activate the conda environment whenver you wish to run `mag1c`.

## Advanced Setup with git clone of this repository
These instructions are targeted towards *nix systems. There are analagous steps on Windows platforms.
### 1. Clone this repository:
```bash
git clone https://github.com/markusfoote/mag1c.git
cd mag1c
```

### 2. Select one of these sub-steps to install dependencies:
#### 2a. pip
You can install all necessary dependencies by using the included requirements.txt file in this repository.
```bash
pip install -r requirements.txt
```
If you do not have administrator access on your system, you may need to request installation of the packages to a location that your user has installation privileges with the `--user` flag:
```bash
pip install --user -r requirements.txt
```
#### 2b. python virtual environment and pip.
Alternatively, you can satisfy the dependencies in a [virtual environment](https://docs.python.org/3.6/tutorial/venv.html) that is independent of your system-wide python installation.
```bash
python3 -m venv my-env
source my-env/bin/activate
pip install -r requirement.txt
```
This snippet first creates a new (and empty) virtual environment, then activates that environment. The required dependencies are then installed within this environment. 
To leave the virtual environment after finishing this how-to, type `deactivate`.
You will need to *activate* this virtual environment anytime you return in a new shell with `source /path/to/<your-environment-name>/bin/activate` before running the `mag1c` processing algorithm.

#### 2c. Anaconda
Python 3.6 or newer is required. If your system does not have a suitable python version, installing [Anaconda](https://www.anaconda.com/distribution/) is usually an acceptable and simple method to obtain a newer python version (as well as the other dependencies).
Conda will create an environment and install dependencies in one step:
```bash
conda create -n myenv 
conda install -n myenv pytorch cudatoolkit=10.0 spectral scipy -c defaults -c conda-forge -c pytorch
```
#### 2d. Install via your package manager
If you are on a linux system, these packages may be provided by your distribution's package manager (`apt` or `yum` or `zypper` ...). This method is typically very platform specific and some packages may not be available and therefore must still be installed by one of the above methods. Ask your system administrator for assistance if you want to go this route.

### 3. Test your installation
Invoking the `mag1c` script will attempt to load all the dependencies before parsing arguments. A simple test that everything is set up correctly is to report the version information:
```bash
python mag1c.py --version
```
or, if you installed `mag1c` through pip:
```bash
mag1c --version
```
If this command reports errors, something is wrong with your setup. Check that you've followed the above instructions, then consider opening an issue or contacting the author(s).

### 4. Process some data!
See below in [Usage](Usage) and Examples for how to process your own data.
Datasets are input and output in [ENVI format](https://www.harrisgeospatial.com/docs/ENVIImageFiles.html).


## Usage
This program can be invoked in multiple ways: 
1. `python /path/to/mag1c.py` works whenever you have a copy of the `mag1c.py` script.
2. `python -m mag1c` works when you install the python package (i.e. via `pip`).
3. `mag1c` is a direct entry point to the program when you install via `pip`.
4. `sparsemf` is exactly the same as `mag1c`, just with a debatably-more-readable name.

# Runtime Options
There are numerous options/flags that can be provided to the mag1c script. Run `magic.py --help` for a full description of the available arguments.

```
usage: mag1c [--spec TARGET_SPEC_FILE] [--out OUTPUT_FILE] [--gpu] [-i N]
             [-g N] [-b N] [-s] [-t N] [-o] [-d] [-p] [--asap] [--noalbedo]
             [--nonnegativeoff] [--onlypositiveradiance] [--outputgeo GLT]
             [--geo GLT] [--optimize] [-q] [--version] [-T THRESHOLD]
             [-W LOW HIGH] [-M [RADIUS]] [-A [AREA]] [--hfdi]
             [--saturation-processing-block-length N] [-S] [-h]
             RADIANCE_FILE

       M atched filter with
       A lbedo correction and
 rewei G hted
     L 1 sparsity
       C ode

University of Utah Albedo-Corrected Reweighted-L1 Matched Filter

positional arguments:
  RADIANCE_FILE         ENVI format radiance file to process -- Provide the
                        data file itself, not the header

optional arguments:
  --spec TARGET_SPEC_FILE
                        Target spectrum file to use. If no file is specified,
                        a target spectrum will be calculated from an internal
                        methane absorption spectrum that is convolved to the
                        band centers and FWHMs stored in the radiance image
                        header.
  --out OUTPUT_FILE     File to write output into. Provide the data file
                        itself, not the header.
  --gpu                 Use GPU for accelerated computation. (default: False)
  -i N, --iter N        Number of iterations to run. (default: 30)
  -g N, --group N       Grouping size for detectors. (default: 1)
  -b N, --batch N       Number of groups processed simultaneously. (default:
                        1)
  -s, --single          Use single/float32 instead of double/float64 in
                        computation. (default: False)
  -t N, --threads N     Number of additional threads for loading data.
                        (default: 0)
  -o, --overwrite       Force the output files to overwrite any existing
                        files. (default: False)
  -d, --writedouble     Write output data file as double precision instead of
                        single. Does not affect compute. (default: False)
  -p, --noprefill       Don't write NO_DATA value (-9999) to the output file
                        upon creation. (default: False)
  --asap                Force the results to be written to disk ASAP after
                        each batch, instead of with some additional delay from
                        the memory-mapped file caching. (default: False)
  --noalbedo            Calculate results withOUT albedo correction. (default:
                        False).
  --nonnegativeoff      Do not apply non-negativity constraint to MF result.
                        (default: False)
  --onlypositiveradiance
                        Only process pixels that have strictly non-negative
                        radiance for all wavelengths. This flag overrides any
                        batch size setting to 1.
  --outputgeo GLT       Use the provided GLT file to perform geocorrection to
                        the resulting ENVI file. This result will have the
                        same name as the provided output file but ending with
                        '_geo'.
  --geo GLT             If using an orthocorrected radiance file, provide the
                        corresponding GLT file. The output will also be
                        geocorrected -- Do not use with --outputgeo to apply
                        glt.
  --optimize            Use Pytorch torchscript optimizations. Experimental.
  -q, --quiet           Do not write status updates to console.
  --version             show program's version number and exit
  -T THRESHOLD, --saturationthreshold THRESHOLD
                        specify the threshold used for classifying pixels as
                        saturated (default: 2.0)
  -W LOW HIGH, --saturationwindow LOW HIGH
                        specify the wavelength window, as "-W LOW HIGH",
                        within which to detect saturation (default: 1945, 2485
                        nanometers)
  -M [RADIUS], --maskgrowradius [RADIUS]
                        radius to use for expanding the saturation mask to
                        cover (and exclude) flare-related anomalies. This
                        value must include units: meters (abbreviated as m) or
                        pixels (abbreviated as px). If flag is given without a
                        value, 100m will be used.
  -A [AREA], --mingrowarea [AREA]
                        minimum number of pixels that must constitute a
                        2-connected saturation region for it to be grown by
                        the mask-grow-radius value. If flag is provided
                        without a value, 6 pixels will be assumed as the
                        value.
  --hfdi                calculate the Hyperspectral Fire Detection Index (doi:
                        10.1016/j.rse.2009.03.010) and append band to output.
  --saturation-processing-block-length N
  -S, --saturation      enable saturation detection (and avoid processing such
                        pixels). This flag overrides any batch size setting to
                        1. Required if any saturation related flags are
                        provided. (default: False)
  -h, --help            show this help message and exit

When using this software, please cite: 
 Foote et al. 2019, "Title Here" doi:xxxx.xxxx
```


### Examples
Process a single file with defaults:
```bash
python mag1c.py /my/radiance --spec /my/target --out /some/output
```

Process a single file with bash variables and some custom options, including GPU:
```bash
export CUDA_VISIBLE_DEVICES=0 # Restrict processing to the first GPU in the system
RDNFILE="/path/to/my/radiance data with spaces in filename"
TEMPLATE=/path/to/my/templatespectrum
OUTPUT=/path/to/outputfile_date_time_iteration25_grouping5
GLT=/path/to/my/gltfile
python mag1c.py "$RDNFILE" --spec $TEMPLATE --out $OUTPUT --outputgeo $GLT --iter 25 --group 5 --gpu -t 2 -b16
```

Process all files in a folder:
```bash
TEMPLATE=/path/to/template.txt
for f in /path/to/folder/ang*_rdn_*_clip; do \
    python mag1c.py "${f}" "${TEMPLATE}" "/output/folder/$(basename "${f/rdn/mag1c}")" --iter 20
done;
```
