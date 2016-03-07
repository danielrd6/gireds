import pyfits as pf
import numpy as np
# import copy
import sys


def make_bpm(flat_seed, bpm_file, pixel_ranges):
    """
    Makes a BPM file from a flat.

    Parameters
    ----------
    flat: string
        Name of the gprepared flatfield exposure that will be the seed
        for the BPM file.
    bpm: string
        Name of the output file.
    pixel_ranges: string
        Name of the ASCII file containing the pixel ranges to be
        masked, one per line in the format
            ext_number x0 y0 x1 y1

    Returns
    -------
    Nothing

    """

    bpm = pf.open(flat_seed, uint=True)
    # info = bpm.info(output=False)

    pl = np.loadtxt(pixel_ranges, dtype='int16')

    ext_ver = 0
    for i, hdu in enumerate(bpm):
        if hdu.name == 'DQ':
            ext_ver += 1
            hdu.data[:, :] = 0
            hdu.name = 'DQ'
            hdu.ver = ext_ver

    for lims in pl:
        bpm['DQ', lims[0]].data[lims[3]:lims[4], lims[1]:lims[2]] = 1

    # bpm[0].header['NEXTEND'] = ext_ver

    bpm.writeto(bpm_file)


if __name__ == '__main__':

    make_bpm(sys.argv[1], sys.argv[2], sys.argv[3])
