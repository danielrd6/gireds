# STDLIB
import argparse

# THIRD PARTY
from astropy.io import fits
import numpy as np


def make_bpm(flat_seed, bpm_file, pixel_ranges):
    """
    Makes a BPM file from a flat.

    Parameters
    ----------
    flat_seed: string
        Name of the gprepared flatfield exposure that will be the seed
        for the BPM file.
    bpm_file: string
        Name of the output file.
    pixel_ranges: string
        Name of the ASCII file containing the pixel ranges to be
        masked, one per line in the format
            ext_number x0 x1 y0 y1

    Returns
    -------
    Nothing

    """

    bpm = fits.open(flat_seed, uint=True)
    mjd = bpm[1].header['MJD-OBS']

    pl = np.loadtxt(pixel_ranges, dtype='int16')

    if len(np.shape(pl)) == 1:
        pl = [pl]

    to_be_removed = [hdu for hdu in bpm if hdu.name != 'DQ']

    for hdu in to_be_removed[1:]:
        bpm.remove(hdu)

    for lims in pl:
        lims[1:] -= 1  # First pixel is 1 in pl, rather than 0.
        bpm['DQ', lims[0]].data[lims[3]:lims[4], lims[1]:lims[2]] = 1

    bpm[0].header['NEXTEND'] = len([i for i in bpm if i.name == 'DQ'])
    bpm[0].header['OBSTYPE'] = 'BPM'
    bpm[1].header['MJD-OBS'] = mjd
    bpm.writeto(bpm_file)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        'flat_seed',
        help='Gprepared flat field exposure  on which the BPM will be based.')
    parser.add_argument(
        'bpm_file',
        help='Output bad pixel mask.')
    parser.add_argument(
        'pixel_ranges',
        help='ASCII file containing the rectangular region definitions to be '
        'masked.')
    args = parser.parse_args()

    make_bpm(args.flat_seed, args.bpm_file, args.pixel_ranges)
