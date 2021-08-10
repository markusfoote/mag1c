#!/usr/bin/env python3
"""sspixmf.py
Scene-Specific Pixel-wise Target-adjusted Matched Filter tool

This file implements a two-pass matched filter algorithm
using per-pixel unit enhancement spectrum target.

Eventually, this could be written to be truely two-pass with out-of-core data,
but for now just loads all data to process in-memory.
Also, a more advanced matched filter could be used (better covariance estimator, or sparse)

Markus Foote
2021
"""

import argparse
import spectral
import numpy as np
import h5py
import scipy
import torch
import time
from packaging.version import parse as parse_version
from target_generation import (get_5deg_zenith_angle_index,
                               get_5deg_sensor_height_index,
                               get_5deg_ground_altitude_index,
                               get_5deg_water_vapor_index,
                               get_5deg_methane_index)
from mag1c import GeocorrectedGroupedRadianceMemmappedFileDataset, get_mask_bad_bands


SCALING = 1e5

# Define inputs
parser = argparse.ArgumentParser(description='Pixel-wise Target Spectrum Matched Filter')
parser.add_argument('--img', type=str)
parser.add_argument('--glt', type=str)
parser.add_argument('--igm', type=str)
parser.add_argument('--obs', type=str)
parser.add_argument('--h2o', type=str)
parser.add_argument('--sensor', type=float)
parser.add_argument('--output', type=str)
parser.add_argument('--average', action='store_true')
parser.add_argument('--waterblur', type=int)
parser.set_defaults(  # defaults for development
                    img='/data/traceGas/CarbonMapper/ang20191018t141549_rdn_v2x1_img',
                    glt='/data/traceGas/CarbonMapper/ang20191018t141549_rdn_v2x1_glt',
                    igm='/data/traceGas/CarbonMapper/ang20191018t141549_rdn_v2x1_igm_ort',
                    obs='/data/traceGas/CarbonMapper/ang20191018t141549_rdn_v2x1_obs_ort',
                    h2o='/data/traceGas/CarbonMapper/ang20191018t141549_h2o_v2x1_img',
                    sensor=28300/3.28084/1000,  # = 28300 feet above sea level -> meters -> kilometers
                    output='/data/traceGas/CarbonMapper/testOutput46.hdr',
                    average=False,
                    waterblur=64,
                    )
opts = parser.parse_args()
# some other options that might be variable, not intended as fully supported now
column_group_size = 1
concentrations = [0.0, 1000, 2000, 4000, 8000, 16000, 32000, 64000]
window_min = 2122
window_max = 2488


# Image loader helper
def open(img: str) -> np.ndarray:
    input_file = spectral.io.envi.open(file=img+'.hdr', image=img)
    data = input_file.open_memmap(interleave='bip', writable=False).astype(np.float64).copy()
    return (data, (input_file.metadata['wavelength'], input_file.metadata['fwhm'])) if 'wavelength' in input_file.metadata else data


print('Loading... ', end='')
start_time = time.time()
# Load Radiance File
data, (centers, fwhm) = open(opts.img)
centers = np.array(centers, dtype=np.float64)
fwhm = np.array(fwhm, dtype=np.float64)
# Load Registration File
glt = open(opts.glt).astype(np.int64)
# Load Parameter Files
igm = open(opts.igm)
h2o, _ = open(opts.h2o)
obs = open(opts.obs)
# Extract parameters of interest
zenith = obs[:, :, 4, np.newaxis]  # Band 5 of OBS file contains to-sun zenith angle, degrees
water  = h2o[:, :, 0, np.newaxis]  # Band 1 of H2O file contains water vapor, cm  # noqa: E221
ground = igm[:, :, 2, np.newaxis]/1000  # Band 3 of IGM file contains surface elevation, meters -> kilometers

if opts.waterblur is not None:
    water[water == -9999] = np.mean(water, where=water != -9999)
    water[..., 0] = scipy.ndimage.median_filter(water[..., 0], size=opts.waterblur//2, mode='nearest')
    water[..., 0] = scipy.ndimage.gaussian_filter(water[..., 0], sigma=opts.waterblur/2.7, mode='nearest')
    # Also Override the solar zenith angle to be 4.6 degree or 10.7 variation, with 70 degree max
    sza_variation = 4.6
    unit_zenith = np.divide(zenith - np.amin(zenith, where=zenith != -9999, initial=90),
                            np.amax(zenith, where=zenith != -9999, initial=0) - np.amin(zenith, where=zenith != -9999, initial=90))
    unit_zenith[zenith == -9999] = -9999
    zenith = np.ones_like(zenith) * 70 - (unit_zenith * sza_variation)
    zenith[unit_zenith == -9999] = -9999

if opts.average:
    zenith = np.ones_like(zenith) * np.mean(zenith, keepdims=True, where=zenith != -9999)
    water = np.ones_like(water) * np.mean(water, keepdims=True, where=water != -9999)
    ground = np.ones_like(ground) * np.mean(ground, keepdims=True, where=ground != -9999)

# Instantiate a geocorrection-reversing for each dataset that will iterate over columns
rads  = GeocorrectedGroupedRadianceMemmappedFileDataset(rdn_memmap_file=data,  # noqa: E221
                                                        band_keep=np.ones(data.shape[2], dtype=bool),
                                                        group_size=column_group_size,
                                                        src_glt_memmap_file=glt,
                                                        sat_mask_full=None)
zcols = GeocorrectedGroupedRadianceMemmappedFileDataset(rdn_memmap_file=zenith,
                                                        band_keep=np.ones(zenith.shape[2], dtype=bool),
                                                        group_size=column_group_size,
                                                        src_glt_memmap_file=glt,
                                                        sat_mask_full=None)
wcols = GeocorrectedGroupedRadianceMemmappedFileDataset(rdn_memmap_file=water,
                                                        band_keep=np.ones(water.shape[2], dtype=bool),
                                                        group_size=column_group_size,
                                                        src_glt_memmap_file=glt,
                                                        sat_mask_full=None)
gcols = GeocorrectedGroupedRadianceMemmappedFileDataset(rdn_memmap_file=ground,
                                                        band_keep=np.ones(ground.shape[2], dtype=bool),
                                                        group_size=column_group_size,
                                                        src_glt_memmap_file=glt,
                                                        sat_mask_full=None)
print(f'{time.time()-start_time} sec.')
start_time = time.time()
print('Preparing... ', end='')

# Determine band selection / keep mask for this scene
band_keep = get_mask_bad_bands(centers)
band_keep[centers > window_max] = False
band_keep[centers < window_min] = False

# Use Band wavelengths and fwhm widths to resample LUT dataset for this scene's instrument calibration
datafile = h5py.File('modtran_ch4_full/dataset_ch4_full.hdf5', 'r', rdcc_nbytes=4194304)
# datafile['modtran_data'], datafile['modtran_param'], datafile['wave'], 'ch4'
lut = datafile['modtran_data'][...]
wave = datafile['wave'][...]
sigma = fwhm[band_keep] / (2.0 * np.sqrt(2.0 * np.log(2.0)))
# response = scipy.stats.norm.pdf(wave[:, None], loc=centers[None, :], scale=sigma[None, :])
# Evaluate normal distribution explicitly
var = sigma ** 2
denom = (2 * np.pi * var) ** 0.5
numer = np.exp(-(np.asarray(wave)[:, None] - centers[None, band_keep])**2 / (2*var))
response = numer / denom
# Normalize each gaussian response to sum to 1.
response = np.divide(response, response.sum(
    axis=0), where=response.sum(axis=0) > 0, out=response)
# implement resampling as matrix multiply
resampled = lut.dot(response)  # this is a big operation, could benefit from GPU or Sparse matrix


# function that takes parameters of column and produces target unit enhancement spectrum
def make_target_from_params(zenith, water, ground, sensor):
    coords = np.stack((get_5deg_zenith_angle_index(zenith),
                       get_5deg_sensor_height_index(sensor) * np.ones_like(zenith),
                       get_5deg_ground_altitude_index(ground),
                       get_5deg_water_vapor_index(water)
                       ),
                      axis=0)
    # Do interpolation lookup of radiative transfer simulation spectra at these coords
    # The dataset is already in memory, so don't bother with this chunk-reading specialty...
    # coords_fractional_part, coords_whole_part = np.modf(coords)
    # coords_near_slice = tuple((slice(int(c), int(c+2)) for c in coords_whole_part))
    # near_grid_data = resampled[coords_near_slice]
    # new_coord = np.concatenate((coords_fractional_part * np.ones((1, near_grid_data.shape[-1])),
    #                             np.arange(near_grid_data.shape[-1])[None, :]), axis=0)
    # lookup = scipy.ndimage.map_coordinates(near_grid_data, coordinates=new_coord, order=1, mode='nearest')
    lookup = np.ndarray((len(concentrations), resampled.shape[-1], zenith.shape[0]))
    for i, conc in enumerate(concentrations):
        conc_coord = get_5deg_methane_index(conc) * np.ones((1, coords.shape[-1]), dtype=np.float64)
        coord_with_conc = np.concatenate((coords, conc_coord), axis=0)
        lookup[i, ...] = np.asarray([scipy.ndimage.map_coordinates(rw,
                                                                   coordinates=coord_with_conc,
                                                                   order=1,
                                                                   mode='nearest')
                                     for rw in np.moveaxis(resampled, 5, 0)])
    # Do regression of points
    lograd = np.log(lookup, out=np.zeros_like(lookup), where=lookup > 0)
    spectrum = np.ndarray(lograd.shape[1:])
    if parse_version(torch.__version__) < parse_version('1.9.0'):  # Iterative fallback
        for n in range(lograd.shape[-1]):  # This loop could be replaced by a batched solver (ie, pytorch > 1.9.0)
            slope, _, _, _ = np.linalg.lstsq(torch.from_numpy(np.stack((np.ones_like(concentrations), concentrations)).T),
                                             lograd[..., n],
                                             rcond=None)
            spectrum[:, n] = slope[1, :] * SCALING
    else:  # batched pytorch solver
        slope, _, _, _ = torch.linalg.lstsq(torch.from_numpy(np.stack((np.ones_like(concentrations), concentrations)).T)[None, ...],
                                            torch.swapaxes(torch.from_numpy(lograd), 0, 1).T,
                                            rcond=None)
        spectrum = slope[:, 1, :].numpy().T * SCALING
    # slope, _, _, _ = np.linalg.lstsq(np.stack((np.ones_like(concentrations), concentrations)).T, lograd, rcond=None)
    # spectrum = slope[1, :] * SCALING
    # target = np.stack((np.arange(1, spectrum.shape[0]+1), centers, spectrum)).T
    # return target
    return spectrum


mf = np.ndarray((data.shape[0], data.shape[1]), dtype=np.float64)
targets = np.ndarray((band_keep.sum(), data.shape[0], data.shape[1]), dtype=np.float64)
print(f'{time.time() - start_time} sec.')
# Process each column with pixel-specific spectrum
for i, (r, z, w, g) in enumerate(zip(rads, zcols, wcols, gcols)):
    print(f'{i}... ', end='')
    start_time = time.time()
    # Extract data for this column
    rad_data, censor_mask, glt_idx, r_time = r
    zenith_data, _, _, z_time = z
    water_data, _, _, w_time = w
    ground_data, _, _, g_time = g
    # Generate pixel-specific spectra
    target = make_target_from_params(zenith=np.squeeze(zenith_data),
                                     water=np.squeeze(water_data),
                                     ground=np.squeeze(ground_data),
                                     sensor=np.asarray(opts.sensor, dtype=np.float64))
    # Select only the bands of interest that produce most useful methane results.
    rad_data = rad_data[:, band_keep]
    # target = target[band_keep, :]

    # Apply simple matched filter
    # Calculate mean and covariance for column
    mean = np.mean(rad_data, axis=0)
    cov = np.cov(rad_data.T)
    try:
        cov_inverse = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        print('inverse error, using pinv', end='')
        cov_inverse = np.linalg.pinv(cov)
    # Multiply the per-pixel unit enhancement spectrum by the column mean to produce the actual target spectrum
    mut = target * mean[:, np.newaxis]
    # Calculate matched filter
    if parse_version(torch.__version__) < parse_version('1.9.0'):  # Iterative fallback
        result = np.ndarray((mut.shape[1]), dtype=np.float)
        for j in range(mut.shape[1]):
            # Cit = np.linalg.solve(cov, mut)
            Cit = np.matmul(cov_inverse, mut)
            # Compute the matched filter results for all pixels in this 'column'
            result[j] = ((rad_data[j] - mean[np.newaxis, :]).dot(Cit)) / mut.T.dot(Cit)
    else:  # Pytorch >= 1.9.0 provides batched solver
        # Cit = torch.linalg.solve(torch.from_numpy(cov)[None, :, :], torch.from_numpy(mut.T)[:, :, None])
        Cit = torch.matmul(torch.from_numpy(cov_inverse)[None, :, :], torch.from_numpy(mut.T)[:, :, None])
        result = torch.div(torch.bmm((torch.from_numpy(rad_data) - torch.from_numpy(mean)[np.newaxis, :])[:, None, :],
                                     Cit),
                           torch.bmm(torch.from_numpy(mut).T[:, None, :],
                                     Cit))
    # Write mf result back to locations in mf image, as dictated by GLT
    mf[glt_idx] = result.numpy().squeeze().squeeze() * SCALING
    # Store the targets for later analysis
    targets[:, glt_idx] = target
    # Report time for this column
    print(f'{time.time() - start_time} sec.')
# write out
spectral.io.envi.save_image(opts.output, np.concatenate((mf[..., None], water, zenith, targets.swapaxes(1, 2).T), axis=2),
                            ext=None, interleave='bsq', force=True)
print(f'Done... {opts.output} .')
