#!/usr/bin/env python
#
#       M ethane detection with
#       A lbedo correction and
# rewei G hted
#     L 1 sparsity
#       C ode
#
# BSD 3-Clause License
#
# Copyright (c) 2019,
#   Scientific Computing and Imaging Institute and
#   Utah Remote Sensing Applications Lab
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Author: Markus Foote (foote@sci.utah.edu)

import torch
import numpy as np
import os
import time
import copy
import argparse
import spectral
import torch.utils.data
from skimage import morphology, measure
from typing import Tuple, Optional, Union, List

RGB = [640, 550, 460]
NODATA = -9999
SAT_THRESH_DEFAULT = 6.0
try:  # get version from packaging information (i.e. pip installation metadata)
    import pkg_resources
    SCRIPT_VERSION = pkg_resources.get_distribution('mag1c').version
except Exception:  # otherwise, assume this is a development version
    SCRIPT_VERSION = '0.0.0-dev0'


@torch.no_grad()
def acrwl1mf(x: torch.Tensor,
             template: torch.Tensor,
             num_iter: int,
             albedo_override: bool,
             zero_override: bool,
             sparse_override: bool,
             covariance_update_scaling: float,
             alpha: float,
             mask: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Calculate the albedo-corrected reweighted-L1 matched filter on radiance data.

    :param x: Radiance Data to process. See notes below on format.
    :param template: Target spectrum for detection.
    :param num_iter: Number of iterations to run.
    :param albedo_override: Do not calculate or apply albedo correction factor.
    :param zero_override: Do not apply non-negativity constraint on matched filter results.
    :param sparse_override: Do not use sparse regularization in iterations when True.
    :param covariance_update_scaling: scalar value controls contribution of previous filter values in removing target
        signal in covariance and mean updates.
    :param alpha: scalar value to perform diagonal scaling of the covariance matrix
    :param mask: An optional mask to mark where data should contribute to covariance and mean.
    :returns mf, albedo

        x must be 3-dimensional:
        batch (columns or groups of columns) x
        pixels (samples) x
        spectrum (bands)

        Notice that the samples dimension must be shared
        by the batch, so only batches/columns with the same number
        of pixels to process may be combined into a batch.
    """
    # Some constants / arrays to preallocate
    dtype = x.dtype
    device = x.device
    N = x.shape[1]  # number of samples
    if mask is None:
        mask = torch.ones_like(x, dtype=torch.bool)
    mask = torch.squeeze(mask, 0)
    regularizer = torch.zeros(x.shape[0], x.shape[1], 1, dtype=dtype, device=device)
    modx = x[:, mask]#torch.zeros_like(x, dtype=dtype, device=device, layout=torch.strided)
    template = template.unsqueeze(0).unsqueeze(0)
    scaling = torch.tensor(1e5, dtype=dtype, device=device)
    epsilon = torch.tensor(1e-9, dtype=dtype, device=device)
    covariance_update_scaling = torch.tensor(-covariance_update_scaling, dtype=dtype, device=device)
    alpha = torch.tensor(alpha, dtype=dtype, device=device)
    # energy = torch.zeros(num_iter + 1, dtype=dtype, device=device)
    # Initialize with normal robust matched filter
    mu = torch.mean(modx, 1, keepdim=True)  # [batch x 1 x spectrum]
    if albedo_override:
        R = torch.ones((x.shape[0], N, 1), dtype=dtype, device=device)
    else:
        R = torch.div(torch.bmm(x, torch.transpose(mu, 1, 2)),  # [b x p x s] * [b x s x 1] = [b x p x 1]
                      torch.bmm(mu, torch.transpose(mu, 1, 2)))  # [b x 1 x s] * [b x s x 1] = [b x 1 x 1]
    target = torch.mul(template, mu)  # [1 x 1 x s] * [b x 1 x s] = [b x 1 x s]
    xmean = modx - mu
    C = torch.div(torch.bmm(torch.transpose(xmean, 1, 2), xmean), N)  # [b x s x p] * [b x p x s] = [b x s x s]
    C = C.lerp_(torch.diag_embed(torch.diagonal(C, dim1=-2, dim2=-1)), alpha)  # C = (1-alpha) * S + alpha * diag(S)
    # Cit, _ = torch.gesv(torch.transpose(target, 1, 2), C)  # [b x s x 1] \ [b x s x s] = [b x s x 1]
    Cit = torch.cholesky_solve(torch.transpose(target, 1, 2), torch.cholesky(C, upper=False), upper=False)
    normalizer = torch.bmm(target, Cit)  # [b x 1 x s] * [b x s x 1] = [b x 1 x 1]
    mf = torch.div(torch.bmm(x - mu, Cit), torch.mul(R, normalizer))  # [b x p x s] * [b x s x 1] = [b x p x 1]
    if not zero_override:
        mf = torch.nn.functional.relu_(mf)  # max(mf, 0)
    # TODO Calculate Energy
    # Reweighted L1 Algorithm
    for i in range(num_iter):
        # Calculate new regularizer weights
        if not sparse_override:  # regularizer pre-defined as zeros.
            regularizer = torch.reciprocal(torch.mul(R, mf + epsilon), out=regularizer)
        # Re-calculate statistics
        modx = torch.add(x[:, mask], alpha=covariance_update_scaling, other=R[:, mask] * mf[:, mask] * target, out=modx)
        mu = torch.mean(modx, 1, keepdim=True, out=mu)
        target = torch.mul(template, mu, out=target)
        xmean = torch.add(modx, alpha=-1, other=mu, out=xmean)
        C = torch.div(torch.bmm(torch.transpose(xmean, 1, 2), xmean), other=torch.tensor(N), out=C)
        C = C.lerp_(torch.diag_embed(torch.diagonal(C, dim1=-2, dim2=-1)), alpha)
        Cit = torch.cholesky_solve(torch.transpose(target, 1, 2), torch.cholesky(C, upper=False), upper=False)
        # Compute matched filter with regularization
        normalizer = torch.bmm(target, Cit, out=normalizer)
        if torch.sum(torch.lt(normalizer, 1)):
            normalizer = normalizer.clamp_(min=1)
        mf = torch.div(torch.bmm(x - mu, Cit) - regularizer, torch.mul(R, normalizer))
        mf = torch.nn.functional.relu_(mf)
        # TODO energy
    mf = torch.mul(mf, scaling, out=mf)
    return mf, R


def get_censor_mask(x: np.ndarray) -> np.ndarray:
    """Determines the regions of the image that are censored. This assumes that an
    entire 'bar'/sample-direction (in un-geocorrected data) is censored together.

    :param x: Radiance image data to analyze for censored regions.
    :return: Boolean 1D ndarray (corresponding to column/flight direction) with True where censoring is detected.
    """
    is_censored = np.diff(x, axis=0) == 0
    is_censored = np.all(is_censored, axis=(1, 2))
    is_censored = np.logical_or(np.logical_or(np.concatenate(([0, 0], is_censored[:-1])),
                                              np.concatenate((is_censored[1:], [0, 0]))),
                                np.logical_or(np.concatenate(([1], is_censored)),
                                              np.concatenate((is_censored, [1]))))
    # Also skip regions that are marked as No Data, value -9999
    is_censored = np.logical_or(is_censored, np.any(x == NODATA, axis=(1, 2)))  # TODO might want to split this out
    return is_censored


def get_saturation_mask(data: np.ndarray, wave: np.ndarray,
                        threshold: Optional[float] = None,
                        waverange: Optional[Tuple[float, float]] = None) -> np.ndarray:
    """Calculates a mask of pixels that appear saturated (in the SWIR, by default).
    Pixels containing ANY radiance value above the provided threshold (default 6.0) within
    the wavelength window provided (default 1945 - 2485 nm).

    :param data: Radiance image to screen for sensor saturation.
    :param wave: vector of wavelengths (in nanometers) that correspond to the bands (last dimension) in the data.
    Caution: No input validation is performed, so this vector MUST be the same length as the data's last dimension.
    :param threshold: radiance value that defines the edge of saturation.
    :param waverange: wavelength range, defined as a tuple (low, high), to screen within for saturation.
    :return: Binary Mask with 1/True where saturation occurs, 0/False for normal pixels
    """
    if threshold is None:
        threshold = SAT_THRESH_DEFAULT
    if waverange is None:
        waverange = (1945, 2485)
    is_saturated = (data[..., np.logical_and(wave >= waverange[0], wave <= waverange[1])] > threshold).any(axis=-1)
    return is_saturated


def calculate_hfdi(data: Union[np.ndarray, torch.Tensor],
                   wave: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """Calculates the Hyperspectral Fire Detection Index.

    :param data: Radiance Image.
    :param wave: vector of wavelengths (in nanometers) that correspond to the bands (last dimension) in the data.
    Caution: No input validation is performed, so this vector MUST be the same length as the data's last dimension.
    :return: HFDI index.
    """
    BANDS = (2430, 2060)
    band_idx = np.argmin(np.absolute(wave - np.asarray((BANDS,)).T), axis=1)
    hfdi = (data[..., band_idx[0]] - data[..., band_idx[1]]) / (data[..., band_idx[0]] + data[..., band_idx[1]])
    return hfdi


def generate_template_from_bands(centers: Union[np.ndarray, List], fwhm: Union[np.ndarray, List]) -> np.ndarray:
    """Calculate a unit absorption spectrum for methane by convolving with given band information.

    :param centers: wavelength values for the band centers, provided in nanometers.
    :param fwhm: full width half maximum for the gaussian kernel of each band.
    :return template: the unit absorption spectum
    """
    # import scipy.stats
    SCALING = 1e5
    centers = np.asarray(centers)
    fwhm = np.asarray(fwhm)
    if np.any(~np.isfinite(centers)) or np.any(~np.isfinite(fwhm)):
        raise RuntimeError('Band Wavelengths Centers/FWHM data contains non-finite data (NaN or Inf).')
    if centers.shape[0] != fwhm.shape[0]:
        raise RuntimeError('Length of band center wavelengths and band fwhm arrays must be equal.')
    lib = spectral.io.envi.open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ch4.hdr'),
                                os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ch4.lut'))
    rads = np.asarray(lib.asarray()).squeeze()
    wave = np.asarray(lib.bands.centers)
    concentrations = np.asarray([0, 500, 1000, 2000, 4000, 8000, 16000])
    # sigma = fwhm / ( 2 * sqrt( 2 * ln(2) ) )  ~=  fwhm / 2.355
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    # response = scipy.stats.norm.pdf(wave[:, None], loc=centers[None, :], scale=sigma[None, :])
    # Evaluate normal distribution explicitly
    var = sigma ** 2
    denom = (2 * np.pi * var) ** 0.5
    numer = np.exp(-(wave[:, None] - centers[None, :])**2 / (2*var))
    response = numer / denom
    # Normalize each gaussian response to sum to 1.
    response = np.divide(response, response.sum(axis=0), where=response.sum(axis=0) > 0, out=response)
    # implement resampling as matrix multiply
    resampled = rads.dot(response)
    lograd = np.log(resampled, out=np.zeros_like(resampled), where=resampled > 0)
    slope, _, _, _ = np.linalg.lstsq(np.stack((np.ones_like(concentrations), concentrations)).T, lograd, rcond=None)
    spectrum = slope[1, :] * SCALING
    target = np.stack((centers, spectrum)).T  # np.stack((np.arange(spectrum.shape[0]), centers, spectrum)).T
    return target

def read_template_from_txt(txt_file: str) -> np.ndarray:
    """Reads a template spectrum stored in a .txt file, stripping band numbers from the first column.

    :param txt_file: Filename of the .txt file to read from.
    :return:
    """
    # Assumes 3 column format, no header
    # Band Number - Wavelength (nm) - Template Value
    template = np.loadtxt(txt_file, ndmin=2)[:, 1:]
    return template


def read_template_from_mat(mat_file: str) -> np.ndarray:
    """Reads a template spectrum stored in the 'spec' variable from a .mat file.

    :param mat_file: Filename of the .txt file to read from.
    :return:
    """
    from scipy.io import loadmat
    template_mat = loadmat(mat_file)
    template = template_mat['spec']
    return template


def get_mask_bad_bands(wave: np.ndarray) -> np.ndarray:
    """Calculates a mask of the wavelengths to keep based on water vapor absorption features.
    Rejects wavelengths: - Below 400 nm
                         - Above 2485 nm
                         - Between 1350-1420 nm (water absorption region)
                         - Between 1800-1945 nm (water absorption region)

    :param wave: Vector of wavelengths to evaluate.
    :return:
    """
    keep_mask = ~(np.logical_or(np.logical_or(wave < 400, wave > 2485),
                                np.logical_or(np.logical_and(wave > 1350,
                                                             wave < 1420),
                                              np.logical_and(wave > 1800,
                                                             wave < 1945))))
    return keep_mask


def get_rbg_band_indexes(wave: np.ndarray) -> np.ndarray:
    """Choose indexes of the channels closest to canonical red, green, and blue wavelength centers.

    :param wave: Vector of wavelengths to choose from.
    :return: a 1D array of three indexes for red, green, and blue, in that order.
    """
    return np.argmin(np.abs(np.array(RGB)[:, np.newaxis] - wave), axis=1)


def apply_glt(glt, raster, background_value=NODATA, out=None):
    """Performs georeferencing on the raster image with the provided GLT file.

    :param glt: ndarray with 3rd dimension length 2 containing sample and line indexes for resampling.
    :param raster: raster image to resample
    :param background_value: Value to write into empty pixels
    :param out: optional ndarray-line object to write glt data into, if memmap, must be BIP order
    :return: ndarray with same spatial size as GLT and band size as raster
    """
    if out is None:
        out = np.zeros((glt.shape[0], glt.shape[1], raster.shape[2]))
    if not np.array_equal(out.shape, [glt.shape[0], glt.shape[1], raster.shape[2]]):
        raise RuntimeError('Image dimensions of the output array do not match provided GLT and Raster image.')
    # GLT may contain negative values - just means the pixel is repeated, still use it for lookup.
    glt_pos = np.absolute(glt)
    # GLT value of zero means no data, extract this because python has zero-indexing.
    glt_mask = np.all(glt_pos == 0, axis=2)
    # Fill the output where the GLT prescribes no data to be the no data value
    out[glt_mask, :] = background_value
    # Do the GLT lookup and assignment to valid locations in the output, minus 1 to map to zero-based indexing
    out[~glt_mask, :] = raster[glt_pos[~glt_mask, 1] - 1, glt_pos[~glt_mask, 0] - 1, :]
    return out


def get_radius_in_pixels(value_str, metadata):
    if value_str.endswith('px'):
        return np.ceil(float(value_str.split('px')[0]))
    if value_str.endswith('m'):
        if 'map info' not in metadata:
            raise RuntimeError('Image does not have resolution specified. Try giving values in pixels.')
        if 'meters' not in metadata['map info'][10].lower():
            raise RuntimeError('Unknown unit for image resolution.')
        meters_per_pixel_x = float(metadata['map info'][5])
        meters_per_pixel_y = float(metadata['map info'][6])
        if meters_per_pixel_x != meters_per_pixel_y:
            qprint('Warning: x and y resolutions are not equal, the average resolution will be used.')
            meters_per_pixel_x = (meters_per_pixel_y + meters_per_pixel_x) / 2.0
        pixel_radius = float(value_str.split('m')[0]) / meters_per_pixel_x
        return np.ceil(pixel_radius)
    raise RuntimeError('Unknown unit specified.')


class GroupedRadianceMemmappedFileDataset(torch.utils.data.Dataset):
    def __init__(self,
                 rdn_memmap_file: np.core.memmap,
                 band_keep: np.ndarray,
                 group_size: int,
                 sat_mask_full: np.ndarray) -> None:
        """Initializes with the memory-mapped file and information for how it should be read.

        :param rdn_memmap_file: Memory-mapped file to read
        :param band_keep: Mask of data bands to pass along for processing
        :param group_size: Number of columns from the image to return at once.
        """
        self.rdn_file = rdn_memmap_file
        self.band_keep_mask = band_keep
        self.group_size = group_size
        self.sat_mask_full = sat_mask_full
        # Determine how many partitions there will be from the group size request
        self.num_lines, self.num_samples, self.num_bands = self.rdn_file.shape
        num_full_batches, num_remaining_cols = np.divmod(self.num_samples, self.group_size)
        num_total_columns = self.num_samples
        if num_remaining_cols != 0:
            num_total_columns += (self.group_size - num_remaining_cols)
        self.column_idx = np.arange(num_total_columns)
        self.column_idx = self.column_idx.reshape((-1, self.group_size))
        if num_remaining_cols != 0:
            self.column_idx[-1, :] -= (self.group_size - num_remaining_cols)
        if not (self.column_idx.shape[0] == (num_full_batches if num_remaining_cols == 0 else num_full_batches + 1)):
            raise RuntimeError('The column grouping has resulted in an unexpected size. Please report to developers.')

    def __len__(self):
        """Provides the number of image partitions that exist."""
        return self.column_idx.shape[0]

    def __getitem__(self, item: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """Get the item-th grouping of columns from the radiance image.

        :param item: the enumerated group to retrieve.
        :return: data for processing the requested group of columns, including the column indexes, the regions detected
        as being censored, the time (in seconds) that was, and the data columns themselves
        """
        load_time = time.time()
        # Extract which columns are included in this image partition
        col_idx = self.column_idx[item, :]
        # Read those columns from the datafile on disk through the memmapped file
        data = np.asarray(self.rdn_file[:, col_idx, :])
        # Filter out the bands that are not used (due to water vapor absorption)
        data = data[:, :, self.band_keep_mask]
        # Compute the region (if any) that was censored and remove it from the data to be passed on
        censor_mask = get_censor_mask(data)
        data = data[~censor_mask, :, :]
        # lookup indexes in saturation mask
        sat_mask = None if self.sat_mask_full is None else self.sat_mask_full[:, col_idx][~censor_mask, ...]
        # Store how long this process took
        load_time = time.time() - load_time
        # # Convert censor_mask to integer type (PyTorch does not support bool)
        # censor_mask = censor_mask.astype(np.uint8)
        # Return data and where it is from in the image to main thread
        if sat_mask is not None:
            return data, censor_mask, col_idx, load_time, sat_mask
        return data, censor_mask, col_idx, load_time

class GeocorrectedGroupedRadianceMemmappedFileDataset(torch.utils.data.Dataset):
    def __init__(self,
                 rdn_memmap_file: np.core.memmap,
                 band_keep: np.ndarray,
                 group_size: int,
                 src_glt_memmap_file: np.core.memmap,
                 sat_mask_full: np.ndarray) -> None:
        self.rdn_file = rdn_memmap_file
        self.band_keep_mask = band_keep
        self.group_size = group_size
        self.glt_file = src_glt_memmap_file
        self.sat_mask_full = sat_mask_full
        # Determine number of partitions from the group size request and size of original rdn represented in glt
        abs_glt = np.absolute(self.glt_file)
        nonzero_glt_lines = abs_glt[abs_glt[:, :, 1] > 0, 1]
        nonzero_glt_samples = abs_glt[abs_glt[:, :, 0] > 0, 0]
        self.num_lines = nonzero_glt_lines.max() - nonzero_glt_lines.min() + 1
        self.num_samples = nonzero_glt_samples.max() - nonzero_glt_samples.min() + 1
        self.num_bands = self.rdn_file.shape[2]
        num_full_batches, num_remaining_cols = np.divmod(self.num_samples, self.group_size)
        num_total_columns = self.num_samples
        if num_remaining_cols != 0:
            num_total_columns += (self.group_size - num_remaining_cols)
        self.column_idx = np.arange(num_total_columns)
        self.column_idx = self.column_idx.reshape((-1, self.group_size))
        if num_remaining_cols != 0:
            self.column_idx[-1, :] -= (self.group_size - num_remaining_cols)
        if not (self.column_idx.shape[0] == (num_full_batches if num_remaining_cols == 0 else num_full_batches + 1)):
            raise RuntimeError('The column grouping has resulted in an unexpected size. Please report to developers.')

    def __len__(self):
        """Provides the number of image partitions that exist."""
        return self.column_idx.shape[0]

    def __getitem__(self, item):
        load_time = time.time()
        # Extract which columns are included in this image partition
        col_idx = self.column_idx[item, :]
        # Find locations within the GLT file that contain those columns -- Columns (Sample) are stored in first dim.
        # Column + 1 because GLT is 1-based indexing
        glt_idx = np.any(np.equal(np.absolute(self.glt_file[:, :, (0,)]), col_idx[None, None, :] + 1), axis=2)
        # Read those locations from the datafile on disk through the memmapped file
        data = np.asarray(self.rdn_file[glt_idx, :])
        # Filter out the bands that are not used (due to water vapor absorption)
        data = data[:, self.band_keep_mask]
        # Censor detection is more difficult because the image is flattened.
        # We can unflatten it with the corresponding sample values from the GLT, accounting for 1-based indexing:
        reconstructed_raw = NODATA * np.ones((self.num_lines, col_idx.shape[0], data.shape[1]))
        reconstructed_raw[np.absolute(self.glt_file[glt_idx, 1]) - 1,
                          np.absolute(self.glt_file[glt_idx, 0]) - col_idx[0] - 1,
                          :] = data
        # Censor detection accounts for empty (-9999/NoData) pixels
        censor_mask = get_censor_mask(reconstructed_raw)  # TODO shouldn't this be filtering the data with censor_mask?
        # Lookup saturation mask values
        sat_mask = None if self.sat_mask_full is None else self.sat_mask_full[glt_idx]
        # Store how long this process took
        load_time = time.time() - load_time
        # # Convert censor_mask to integer type (PyTorch does not support bool)
        # censor_mask = censor_mask.astype(np.uint8)
        # glt_idx = glt_idx.astype(np.uint8)
        # Return data and where it is from in the image to main thread
        if sat_mask is not None:
            return data, censor_mask, glt_idx, load_time, sat_mask
        return data, censor_mask, glt_idx, load_time

class QuietPrinter(object):
    def __init__(self, quiet):
        self.quiet = quiet

    def __call__(self, *args, **kwargs):
        if not self.quiet:
            print(*args, **kwargs)

    def set_quiet(self, be_quiet):
        self.quiet = be_quiet

qprint = QuietPrinter(False)

def main():
    parser = argparse.ArgumentParser(description='       M atched filter with\n'
                                                 '       A lbedo correction and\n'
                                                 ' rewei G hted\n'
                                                 '     L 1 sparsity\n'
                                                 '       C ode\n\n'
                                                 'University of Utah Albedo-Corrected Reweighted-L1 Matched Filter\n'
                                                 f'v{SCRIPT_VERSION}',
                                     epilog='When using this software, please cite: \n' +
                                            ' Foote et al. 2019, "Title Here" doi:xxxx.xxxx\n',
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     add_help=False, allow_abbrev=False)
    parser.add_argument('--spec', type=str, metavar='TARGET_SPEC_FILE',
                        help='Target spectrum file to use. If no file is specified, a target spectrum will be '
                             'calculated from an internal methane absorption spectrum that is convolved to the band '
                             'centers and FWHMs stored in the radiance image header.')
    parser.add_argument('--out', type=str, metavar='OUTPUT_FILE',
                        help='File to write output into. Provide the data file itself, not the header.')
    parser.add_argument('--gpu', action='store_true',
                        help='Use GPU for accelerated computation. (default: %(default)s)')
    parser.add_argument('-i', '--iter', type=int, default=30, metavar='N',
                        help='Number of iterations to run. (default: %(default)s)')
    parser.add_argument('-g', '--group', type=int, default=1, metavar='N',
                        help='Grouping size for detectors. (default: %(default)s)')
    parser.add_argument('-b', '--batch', type=int, default=1, metavar='N',
                        help='Number of groups processed simultaneously. (default: %(default)s)')
    parser.add_argument('-s', '--single', action='store_true',
                        help='Use single/float32 instead of double/float64 in computation. (default: %(default)s)')
    parser.add_argument('-t', '--threads', type=int, default=0, metavar='N',
                        help='Number of additional threads for loading data. (default: %(default)s)')
    parser.add_argument('-o', '--overwrite', action='store_true',
                        help='Force the output files to overwrite any existing files. (default: %(default)s)')
    parser.add_argument('-d', '--writedouble', action='store_true',
                        help='Write output data file as double precision instead of single. Does not affect compute.' +
                             ' (default: %(default)s)')
    parser.add_argument('-p', '--noprefill', action='store_true',
                        help='Don\'t write NO_DATA value (-9999) to the output file upon creation. ' +
                             '(default: %(default)s)')
    parser.add_argument('--asap', action='store_true',
                        help='Force the results to be written to disk ASAP after each batch, instead of with some ' +
                             'additional delay from the memory-mapped file caching. (default: %(default)s)')
    # parser.add_argument('--energy', type=str, nargs='?', const='./energy.csv', metavar='LOGFILE',
    #                     help='Calculate and write out a log of the energy optimization to the provided file. If this'+
    #                          ' flag is provided but no filename provided, the log will be written to energy.csv ' +
    #                          'in the current directory. This additional computation will cause the entire process to'+
    #                          ' take significantly longer to complete and is meant for debugging. NOT YET IMPLEMENTED')
    parser.add_argument('--noalbedo', action='store_true',
                        help='Calculate results withOUT albedo correction. (default: %(default)s).')
    parser.add_argument('--no-albedo-output', action='store_true', help='Do not include albedo band in result file.')
    parser.add_argument('--nonnegativeoff', action='store_true',
                        help='Do not apply non-negativity constraint to MF result. (default: %(default)s)')
    parser.add_argument('--onlypositiveradiance', action='store_true',
                        help='Only process pixels that have strictly non-negative radiance for all wavelengths. ' +
                             'This flag overrides any batch size setting to 1.')
    parser.add_argument('--outputgeo', type=str, metavar='GLT',
                        help='Use the provided GLT file to perform geocorrection to the resulting ENVI file. This ' +
                             'result will have the same name as the provided output file but ending with \'_geo\'.')
    parser.add_argument('--geo', type=str, metavar='GLT', dest='rdnfromgeo',
                        help='If using an orthocorrected radiance file, provide the corresponding GLT file. '
                             'The output will also be geocorrected. Do not use with --outputgeo to apply glt.')
    parser.add_argument('--optimize', action='store_true', help='Use Pytorch torchscript optimizations. Experimental.')
    parser.add_argument('-q', '--quiet', action='store_true',
                        help='Do not write status updates to console.')
    parser.add_argument('--version', action='version',
                        version='%(prog)s \n University of Utah Albedo-Corrected Reweighted-L1 Matched Filter.\n' +
                                f'v{SCRIPT_VERSION}\n See Foote et al. 2019 DOI: XXXx.xxxx for details.')
    parser.add_argument('-T', '--saturationthreshold', type=float, metavar='THRESHOLD',
                        help='specify the threshold used for classifying pixels as saturated '
                             f'(default: {SAT_THRESH_DEFAULT})')
    parser.add_argument('-W', '--saturationwindow', type=float, nargs=2, metavar=('LOW', 'HIGH'),
                        help='specify the contiguous wavelength window within which to detect saturation, independent '
                             'of bands used in the filter (default: 1945, 2485 nanometers)')
    parser.add_argument('-M', '--maskgrowradius', type=str, metavar='RADIUS', nargs='?', const='150m', default=None,
                        help='radius to use for expanding the saturation mask to cover (and exclude) flare-related '
                             'anomalies. This value must include units: meters (abbreviated as m) or pixels '
                             '(abbreviated as px). If flag is given without a value, %(default)s will be used. This is '
                             'a combined flag for enabling mask dilation and setting the distance to dilate.')
    parser.add_argument('-A', '--mingrowarea', type=int, metavar='PX_AREA', nargs='?', const=5, default=None,
                        help='minimum number of pixels that must constitute a 2-connected saturation region for it to '
                             'be grown by the mask-grow-radius value. If flag is provided without a value, '
                             '%(const)s pixels will be assumed as the value.')
    parser.add_argument('--hfdi', action='store_true',
                        help='calculate the Hyperspectral Fire Detection Index (doi: 10.1016/j.rse.2009.03.010) '
                             'and append band to output.')
    parser.add_argument('--saturation-processing-block-length', type=int, metavar='N', default=500,
                        help='control the number of data lines pre-processed at once when using masking options')
    parser.add_argument('--no-sparsity', action='store_true',
                        help='disable computation of sparsity weighting factors and sets them to zero')
    parser.add_argument('--covariance-update-scaling', type=float, default=1.0, metavar='FLOAT',
                        help='modify the amount of retrieved signal that is removed from the data for iterative '
                             'statistics updates. 1 removes identically the signal that is detected. 0 does not modify '
                             'the data used for statistical estimates and effectively does not update the covariance '
                             'in iterative steps (default: %(default)s)')
    parser.add_argument('--covariance-lerp-alpha', type=float, default=0.0, metavar='FLOAT',
                        help='modify the covariance at all iterations by emphasizing the diagonal through linearly '
                             'interpolating non-diagonal entries to zero by this fractional weight [0, 1] '
                             '(default: %(default)s)')
    parser.add_argument('--use-wavelength-range', type=float, default=(2122, 2488), nargs=2, metavar=('MIN', 'MAX'),
                        help='defines what contiguous range of wavelengths (in nanometers) should be included in the '
                             'filter calculation (default: %(default)s nm)')
    parser.add_argument('--visible-mask-growing-threshold', type=float, default=9.0, metavar='FLOAT',
                        help='restrict mask dilation to only occur when 500 nm radiance is less than this value')
    # The saturation argument is required only if the saturation threshold or window has been specified.
    # Save all (potentially) required arguments until after this initial parse so that we can pass without errors.
    args, rem_args = parser.parse_known_args()
    parser.add_argument('-S', '--saturation', action='store_true', required=(args.saturationthreshold is not None or
                                                                             args.saturationwindow is not None or
                                                                             args.maskgrowradius is not None or
                                                                             args.mingrowarea is not None),
                        help='enable saturation detection and masking - masked pixels are excluded from statistical '
                             'calculations that influence all pixels, but the statistics from other pixels being '
                             'processed will be used to filter the masked pixel. This flag overrides any batch size '
                             'setting to 1. Required if any saturation related flags are provided. '
                             '(default: %(default)s)')
    parser.add_argument('rdn', type=str, metavar='RADIANCE_FILE',
                        help='ENVI format radiance file to process -- Provide the data file itself, not the header')
    # Add the help option back, because we had to remove it from initial parsing so that all options are present.
    parser.add_argument('-h', '--help', action='help', help='show this help message and exit')
    # Finally, parse all the (remaining) arguments, this will catch the --help option, and fail on required arguments
    args = parser.parse_args(rem_args, namespace=args)
    start = time.time()

    qprint.set_quiet(args.quiet)
    qprint(f'Beginning processing of {args.rdn}')
    if args.onlypositiveradiance and args.batch is not 1:
        args.batch = 1
        qprint('Forced batch size to be 1 for positive radiance filtering.')
    if args.saturation and args.batch is not 1:
        args.batch = 1
        qprint('Forced batch size to be 1 for saturation detection.')
    if args.rdnfromgeo and args.batch is not 1:
        args.batch = 1
        qprint('Forced batch size to be 1 to use geocorrected source radiance.')
    if args.rdnfromgeo is not None and args.outputgeo is not None:
        args.outputgeo = None
        qprint('Overriding attempt to geocorrect output with GLT: output will already be '
               'geocorrected because the input is geocorrected.')
    # Determine compute device to use based on availability and user request
    device = torch.device('cuda') if args.gpu and torch.cuda.is_available() else torch.device('cpu')
    device_name = 'cpu' if device == torch.device('cpu') else torch.cuda.get_device_name(device)
    qprint(f'Using compute device: {device_name}.')
    # Determine the floating point precision to use for compute/output based on user request
    dtype = torch.float32 if args.single else torch.float64
    output_dtype = np.float64 if args.writedouble else np.float32
    qprint(f'Computing with precision {"float64" if dtype == torch.float64 else "float32"}.' +
           f' Output will be written as {"float32" if output_dtype == np.float32 else "float64"}.')

    # Open the specified radiance file as a memory-mapped object
    qprint('Opening radiance data file.')
    rdn_file = spectral.io.envi.open(args.rdn + '.hdr')
    rdn_file_memmap = rdn_file.open_memmap(interleave='bip', writable=False)

    if args.spec is None:  # Convolve internal methane spectrum for header bands/fwhm.
        if str(rdn_file.bands.band_unit).lower() in ['nanometers', 'nanometer', 'nm']:
            target = generate_template_from_bands(centers=rdn_file.bands.centers, fwhm=rdn_file.bands.bandwidths)
        else:
            raise RuntimeError(f'Unknown band wavelength unit: {rdn_file.bands.band_unit}')
        qprint('Target spectrum generated successfully.')
    else:  # Load the target spectrum file that was provided by the user
        qprint(f'Loading target spectrum from {args.spec}')
        if args.spec.endswith('.txt'):
            target = read_template_from_txt(args.spec)
        elif args.spec.endswith('.mat'):
            target = read_template_from_mat(args.spec)
        else:
            raise RuntimeError('Invalid target file, expected file types are .txt or .mat.')
        qprint('Target spectrum loaded successfully.')

    # Check target spectrum wavelengths match the target, otherwise fail (mainly for loaded spectrum files)
    qprint('Checking that the provided target is compatible with the data file wavelengths...')
    wavelengths = np.array(rdn_file.bands.centers)  # np.asarray(rdn_file.metadata['wavelength']).astype(float).T
    if wavelengths.shape[0] != target.shape[0]:
        qprint('Warning: Provided target spectrum does not have the same number of bands as data! Trying zero pad...')
        padded_spec = np.zeros((wavelengths.shape[0], 2))
        padded_spec[:, 0] = wavelengths
        # if np.max(wavelengths) < np.max(target[:, 0]) or np.min(wavelengths) > np.min(target[:, 0]):
        #    raise RuntimeError('Provided target spectrum has values for wavelengths outside of the '
        #                       'range of wavelengths in the data.')
        qprint(' Using wavelengths from provided target that are within 0.2 nm of the data wavelengths...')
        for wave in target:
            padded_spec[np.absolute(padded_spec[:, 0] - wave[0]) < 0.2, :] = wave
        qprint(' Zero padding provided spectrum complete.')
        target = padded_spec
    if not np.allclose(wavelengths, target[:, 0], atol=0.2):
        raise RuntimeError('Target spectrum has different wavelengths than radiance file.')
    qprint('Target spectrum checks complete.')

    # Determine the index and wavelengths of the RGB data that will be included in the output
    rgb_idx = get_rbg_band_indexes(wavelengths)
    rgb_wavelengths = wavelengths[rgb_idx]

    # Determine mask of what wavelengths will be used in processing -- these bands are independent of thresholding
    band_keep = get_mask_bad_bands(wavelengths)
    band_keep[wavelengths < args.use_wavelength_range[0]] = False
    band_keep[wavelengths > args.use_wavelength_range[1]] = False
    wave_keep = wavelengths[band_keep]

    # If thresholding is enabled, calculate the mask and (if enabled) preprocess with dilation
    sat_mask_full = None
    if args.saturation:
        qprint('Detecting saturated pixels...', end='')
        if args.maskgrowradius is not None:
            grow_radius_px = get_radius_in_pixels(args.maskgrowradius, rdn_file.metadata)
            selem = morphology.disk(radius=grow_radius_px, dtype=np.bool)
            idx_500 = np.argmin(np.absolute(wavelengths - 500))
        sat_mask_full = np.zeros((rdn_file.nrows, rdn_file.ncols), dtype=np.uint8)
        block_overlap = np.ceil((args.mingrowarea if args.mingrowarea is not None else 0) + (grow_radius_px if args.maskgrowradius is not None else 0)).astype(np.int64)
        block_step = args.saturation_processing_block_length
        block_length = block_step + block_overlap
        line_idx_start_values = np.arange(start=0, stop=rdn_file.nrows, step=block_step)
        for line_block_start in line_idx_start_values:
            qprint('.', end='', flush=True)
            line_block_end = np.minimum(rdn_file.nrows, line_block_start + block_length)
            block_data = rdn_file.read_subregion((line_block_start, line_block_end), (0, rdn_file.ncols))
            sat_mask_block = get_saturation_mask(data=block_data[:, :, :], wave=wavelengths,
                                                 threshold=args.saturationthreshold, waverange=args.saturationwindow)
            if args.maskgrowradius is not None:
                sat_mask_grow_regions = np.zeros_like(sat_mask_block, dtype=np.uint8)
                for region in measure.regionprops(measure.label(sat_mask_block.astype(np.uint8), connectivity=2)):
                    if args.mingrowarea is None or region.area >= args.mingrowarea:
                        # Mark these large regions in the mask to get dilated
                        for c in region.coords:
                            sat_mask_grow_regions[c[0], c[1]] = 1 if block_data[c[0], c[1], idx_500] < args.visible_mask_growing_threshold else 0
                sat_mask_large_grown = morphology.binary_dilation(image=sat_mask_grow_regions.astype(np.bool),
                                                                  selem=selem)
                sat_mask_out = sat_mask_large_grown.astype(np.uint8)
                sat_mask_out[sat_mask_block] = np.asarray(2, dtype=np.uint8)
                sat_mask_block = sat_mask_out
            sat_mask_full[line_block_start:line_block_end, ...][
                np.logical_and(sat_mask_block == 1, sat_mask_full[line_block_start:line_block_end, ...] != 2)] = 1
            sat_mask_full[line_block_start:line_block_end, ...][sat_mask_block == 2] = 2
        qprint(' Done.')

    # Create an iterable dataset that loads the data into memory as we need it
    if args.rdnfromgeo:
        # Open GLT file for source radiance lookups.
        src_glt_file = spectral.io.envi.open(args.rdnfromgeo + '.hdr')
        src_glt_file_memmap = src_glt_file.open_memmap(interleave='bip', writable=False)
        dataset = torch.utils.data.DataLoader(GeocorrectedGroupedRadianceMemmappedFileDataset(rdn_file_memmap,
                                                                                              band_keep,
                                                                                              args.group,
                                                                                              src_glt_file_memmap,
                                                                                              sat_mask_full),
                                              num_workers=args.threads,
                                              batch_size=args.batch,  # Should be 1 -- MUST be 1
                                              pin_memory=device is not torch.device('cpu'))
    else:
        dataset = torch.utils.data.DataLoader(GroupedRadianceMemmappedFileDataset(rdn_file_memmap,
                                                                                  band_keep,
                                                                                  args.group,
                                                                                  sat_mask_full),
                                              num_workers=args.threads,
                                              batch_size=args.batch,
                                              pin_memory=device is not torch.device('cpu'))

    # Create an image file for the output
    output_metadata = {'description': 'University of Utah Albedo-Corrected Reweighted-L1 Matched Filter Result. ' +
                                      'v' + SCRIPT_VERSION +
                                      '  Citation doi:xxxx.xxxx',
                       'wavelength': np.concatenate((rgb_wavelengths,
                                                     [wavelengths[np.argmin(target[:, 1])]])),
                       'wavelength units': rdn_file.metadata[
                           'wavelength units'] if 'wavelength units' in rdn_file.metadata else 'Nanometers',
                       'data ignore value': NODATA,
                       'band names': ['Red Radiance (uW nm-1 cm-2 sr-1)', 'Green Radiance', 'Blue Radiance',
                                      ('Masked ' if args.saturation else '') + 'Matched Filter Results (ppm m)'],
                       'interleave': 'bsq',
                       'lines': rdn_file_memmap.shape[0],
                       'samples': rdn_file_memmap.shape[1],
                       'bands': 4,
                       'data type': spectral.io.envi.dtype_to_envi[np.dtype(output_dtype).char],
                       'algorithm settings': '{' f'version: {SCRIPT_VERSION}, ' +
                                             f'grouping: {args.group}, iterations: {args.iter}, ' +
                                             f'albedocorrection: {not args.noalbedo}, ' +
                                             f'saturationrejection: {args.saturation}, ' +
                                             (f'saturationwindow: {args.saturationwindow if args.saturationwindow is not None else (1945, 2485)}, ' if args.saturation else '') +
                                             (f'saturationthreshold: {args.saturationthreshold if args.saturationthreshold is not None else SAT_THRESH_DEFAULT}, ' if args.saturation else '') +
                                             f'target spectrum: {args.spec if args.spec is not None else "generate"}, ' +
                                             f'compute: {device_name}, batch: {args.batch}, ' +
                                             (f'buffer distance: {args.maskgrowradius if args.maskgrowradius is not None else 0}, ' if args.saturation else '') +
                                             (f'min contiguous px for buffer: {args.mingrowarea if args.mingrowarea is not None else 0}, ' if args.saturation else '') +
                                             f'filter wavelength range: {args.use_wavelength_range[0]} to {args.use_wavelength_range[1]}, ' +
                                             (f'500 nm mask buffering threshold: {args.visible_mask_growing_threshold}, ' if args.saturation else '') +
                                             f'parsed cmdline args: {args}'
                                             '}'}
    if args.rdnfromgeo:
        output_metadata.update({'map info': rdn_file.metadata['map info']})
    if not args.no_albedo_output:  # Must be first so that index is 4
        output_metadata.update({'band names': output_metadata['band names'] + ['Albedo Factor (dimensionless)']})
        output_metadata.update({'bands': output_metadata['bands'] + 1})
        output_metadata.update({'wavelength': np.concatenate((output_metadata['wavelength'], [wavelengths[0]]))})
    if args.saturation:
        output_metadata.update({'band names': output_metadata['band names'] + ['Saturation Mask (dimensionless)', 'Unmasked Matched Filter Results (ppm m)']})
        output_metadata.update({'bands': output_metadata['bands'] + 2})
        output_metadata.update({'wavelength': np.concatenate((output_metadata['wavelength'],
                                                              wave_keep[None, 0],  # Mask
                                                              [wavelengths[np.argmin(target[:, 1])]]))})  # Unmasked MF
        sat_idx = output_metadata['bands'] - 2
        unmasked_idx = output_metadata['bands'] - 1
    if args.hfdi:
        output_metadata.update({'band names': output_metadata['band names'] + ['HFDI (dimensionless)']})
        output_metadata.update({'bands': output_metadata['bands'] + 1})
        output_metadata.update({'wavelength': np.concatenate((output_metadata['wavelength'], np.asarray((2430,))))})
        hfdi_idx = output_metadata['bands'] - 1
    if args.out is not None:
        output_filename = f'{args.out}.hdr'
    else:
        output_filename = os.path.basename(args.rdn)[:len('xxxYYYYMMDDtHHMMSS')] + '_ch4_cmfr'
    output_file = spectral.io.envi.create_image(output_filename, output_metadata, force=args.overwrite, ext='')
    output_memmap = output_file.open_memmap(interleave='bip', writable=True)
    qprint(f'Filter output will be written to {output_filename}')
    if not args.noprefill:
        output_memmap[:, :, 3:] = NODATA
    if args.saturation:
        output_memmap[:, :, sat_idx] = sat_mask_full

    # Copy RGB bands to the output file
    output_memmap[:, :, :3] = rdn_file_memmap[:, :, rgb_idx]

    # Prepare target spectrum by moving it to correct device, data type, chopping off wavelength labels, bad bands
    spec = torch.from_numpy(target[band_keep, 1]).to(device=device, dtype=dtype)

    qprint(f'Beginning main filtering, to be completed in {len(dataset)} steps:')
    if args.noalbedo:
        qprint('Albedo correction is NOT being applied.')
    # Loop over all batches of partitions in the image dataset
    for step, batch in enumerate(dataset, 1):
        end = '\n' if step % 15 == 0 else ''
        qprint(f'{step:{np.ceil(np.log10(len(dataset))).astype(np.int)}}, ', end=end, flush=True)
        # Unpack data from the batched object
        if args.saturation:
            rdn_data, censor_mask, col_idx, load_times, sat_mask = batch
        else:
            rdn_data, censor_mask, col_idx, load_times = batch
            sat_mask = None

        # Compute HFDI
        if args.hfdi:
            hfdi = calculate_hfdi(rdn_data, wave_keep)

        # Saturation Processing
        if args.saturation:
            # mask is now pre-calculated, it is unpacked as part of batch
            # # mask must be converted back to bool, as torch makes it uint8
            # sat_mask = sat_mask.cpu().numpy() >= 1
            #sat_mask = get_saturation_mask(rdn_data, wave_keep,
            #                               threshold=args.saturationthreshold, waverange=args.saturationwindow)
            sat_mask = sat_mask.flatten().unsqueeze(0)
            rdn_data = rdn_data.reshape(-1, rdn_data.shape[-1]).unsqueeze(0)  # batch size will be 1 when mask is used
            # ^ that was a no-op for geocorrected data, because the data loader already flattens it in the glt lookup
        else:
            # Flatten data into long columns, preserving first (0th) batch dimension and last (-1st) spectral dimension
            # No-op if the rdn data is already spatially flattened
            rdn_data = rdn_data.reshape((rdn_data.shape[0], np.prod(rdn_data.shape[1:-1]), rdn_data.shape[-1]))
            sat_mask = torch.zeros(rdn_data.shape[:-1], dtype=torch.bool)  # false indicates not saturated px
        # Move data to desired compute device, and cast to appropriate data type for processing
        rdn_data = rdn_data.to(device=device, dtype=dtype)
        sat_mask = sat_mask.to(device=device, dtype=torch.bool) if sat_mask is not None else None
        if args.onlypositiveradiance:  # Batch size is 1, so the first dimension can be eliminated within if statement
            # Identify pixels that have all positive radiance values, filter will only be applied on these pixels
            positive_mask = torch.ge(rdn_data, 0).all(dim=2)

            # Preallocate all outputs so that the results assignment can be masked by the positive pixels
            mf_out = NODATA * torch.ones((1, positive_mask.shape[1], 1), dtype=dtype, device=device)
            albedo_out = NODATA * torch.ones((1, positive_mask.shape[1], 1), dtype=dtype, device=device)

            # Do main filter processing on masked pixels
            mf_out[positive_mask, :], albedo_out[positive_mask, :] = acrwl1mf(rdn_data[positive_mask, :].unsqueeze_(0),
                                                                              spec, args.iter, args.noalbedo,
                                                                              args.nonnegativeoff, args.no_sparsity,
                                                                              args.covariance_update_scaling,
                                                                              args.covariance_lerp_alpha,
                                                                              torch.logical_not(sat_mask) if sat_mask is not None else None)
        else:
            # Do main filter processing
            mf_out, albedo_out = acrwl1mf(rdn_data,
                                          spec, args.iter, args.noalbedo,
                                          args.nonnegativeoff, args.no_sparsity,
                                          args.covariance_update_scaling, args.covariance_lerp_alpha,
                                          torch.logical_not(sat_mask) if sat_mask is not None else None)

        # Copy results back to cpu, no-op if already on cpu
        mf_out = mf_out.to(device=torch.device('cpu'))
        albedo_out = albedo_out.to(device=torch.device('cpu'))
        sat_mask = sat_mask.to(device=torch.device('cpu'))

        if args.rdnfromgeo:  # Modify how the data gets written back
            # col_idx is actually glt_idx, which is a boolean mask array stored as torch.uint8  # TODO use torch.bool
            glt_idx = col_idx.cpu().numpy()[0, ...] == 1
            if args.hfdi:
                output_memmap[glt_idx, hfdi_idx] = hfdi
            if args.saturation:
                # Write out data that is calculated on original 'full' data first
                # data is already written out to file output_memmap[glt_idx, sat_idx] = sat_mask
                # Now modify the glt_index for writing data that is calculated on a further subset.
                # Saturation Mask is only calculated on 'active' data -- where glt_idx is True
                # The mask needs to get and-ed with the mask at these True locations only
                ## Write the 'full' unmasked data to the unmasked result band, then modify the glt index and data
                output_memmap[glt_idx, unmasked_idx] = mf_out.squeeze(0).squeeze(1)
                mf_out = mf_out[torch.logical_not(sat_mask)].unsqueeze_(0)
                albedo_out = albedo_out[torch.logical_not(sat_mask)].unsqueeze_(0)
                np.place(glt_idx, glt_idx, np.logical_and(glt_idx[glt_idx], np.logical_not(sat_mask)))
            # Data just has to be written back to where it came from, marked by glt_idx, flat order is preserved
            output_memmap[glt_idx, 3] = mf_out[0, :, 0]
            if not args.no_albedo_output:
                output_memmap[glt_idx, 4] = albedo_out[0, :, 0]

        else:
            # Un-flatten the (maybe multiple) columns back to their original places and write to disk
            censor_mask_bool = censor_mask.cpu().numpy() == 0
            for i in range(args.batch):
                temp_out_reshape = NODATA * np.ones((censor_mask_bool.shape[1], col_idx.shape[1])).flatten()
                temp_albedo_reshape = NODATA * np.ones((censor_mask_bool.shape[1], col_idx.shape[1])).flatten()
                if args.saturation:  # Expand results back to size of censor_mask_bool, then following logic is same
                    ## Write the 'full' unmasked data to the unmasked result band, then re-mask data and proceed with previous logic
                    temp_full_reshape = temp_out_reshape.copy()
                    temp_full_reshape[np.repeat(censor_mask_bool[i], args.group)] = mf_out[i, :, 0].numpy()
                    temp_full_reshape = temp_full_reshape.reshape(censor_mask_bool.shape[1], col_idx.shape[1])
                    output_memmap[:, col_idx[i], unmasked_idx] = temp_full_reshape.squeeze()
                    mf_out = mf_out[torch.logical_not(sat_mask)].unsqueeze_(0)
                    albedo_out = albedo_out[torch.logical_not(sat_mask)].unsqueeze_(0)
                    # This is simplified because the batch size must be 1 when saturation is applied
                    # sat_mask is the size that is expected for censor unmasking, so make these expansions the same size
                    expand_mf = NODATA * torch.ones(sat_mask.shape, dtype=dtype).unsqueeze_(-1)
                    expand_albedo = NODATA * torch.ones(sat_mask.shape, dtype=dtype).unsqueeze_(-1)
                    expand_mf[~sat_mask, :] = mf_out
                    expand_albedo[~sat_mask, :] = albedo_out
                    expand_mf = expand_mf.view(1, -1, 1)
                    expand_albedo = expand_albedo.view(1, -1, 1)
                    mf_out = expand_mf
                    albedo_out = expand_albedo
                temp_out_reshape[np.repeat(censor_mask_bool[i], args.group)] = mf_out[i, :, 0].numpy()
                temp_albedo_reshape[np.repeat(censor_mask_bool[i], args.group)] = albedo_out[i, :, 0].numpy()
                temp_out_reshape = temp_out_reshape.reshape(censor_mask_bool.shape[1], col_idx.shape[1])
                temp_albedo_reshape = temp_albedo_reshape.reshape(censor_mask_bool.shape[1], col_idx.shape[1])
                output_memmap[:, col_idx[i], 3] = temp_out_reshape.squeeze()
                if not args.no_albedo_output:
                    output_memmap[:, col_idx[i], 4] = temp_albedo_reshape.squeeze()
                if args.saturation:  # Write back
                    pass
                    # Batch size must be 1, so this simplifies. censor_mask still applies though
                    #output_memmap[:, col_idx[i], sat_idx] = sat_mask
                    # saturation mask has already been written out to file
                    #  output_memmap[np.logical_and(censor_mask_bool[i][:, None],
                    #                             (col_idx[i, None].numpy().T == np.arange(output_memmap.shape[1])).any(axis=0)[None, :]), sat_idx] = sat_mask.numpy().astype(np.float64 if args.writedouble else np.float32).flat
                if args.hfdi:  # Calculated everywhere, so disregard saturation masking; censor mask still applies
                    # Is not flattened, so unflattening is not required
                    output_memmap[np.logical_and(censor_mask_bool[i][:, None],
                                                 (col_idx[i, None].numpy().T == np.arange(output_memmap.shape[1])).any(axis=0)[None, :]), hfdi_idx] = hfdi.numpy().astype(np.float64 if args.writedouble else np.float32).flat

        if args.asap:
            output_memmap.flush()

    output_memmap.flush()
    run_time = time.time() - start
    qprint(f'\nFilter processing completed in {run_time} seconds.')

    # Do geocorrection based on the provided GLT file, if provided.
    if args.outputgeo is not None:
        qprint(f'Beginning geocorrection with {args.outputgeo}')

        # Open the GLT file
        glt_file = spectral.io.envi.open(args.outputgeo + '.hdr')
        glt_memmap = glt_file.open_memmap(interleave='bip', writable=False)

        # Create an output file for the georeferenced data. Metadata based on filter output and GLT file essentials.
        geo_metadata = copy.deepcopy(output_metadata)  # has the same bands, wavelengths, description, etc.
        keys_from_glt = ['map info', 'samples', 'lines']
        geo_metadata = {**geo_metadata, **{k: glt_file.metadata[k] for k in glt_file.metadata.keys() & keys_from_glt}}
        geo_filename = f'{args.out}_geo.hdr'
        geo_file = spectral.io.envi.create_image(geo_filename, geo_metadata, force=args.overwrite, ext='')
        geo_memmap = geo_file.open_memmap(interleave='bip', writable=True)

        # Write the data from the filter output into the geo_file according to the glt
        apply_glt(glt_memmap, output_memmap, out=geo_memmap)

        # That's all!
        qprint(f'Wrote the geocorrected filter output to {geo_filename}')

    qprint(f'Done with all requested processing for {args.rdn}')


if __name__ == '__main__':
    main()
