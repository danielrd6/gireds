#!/usr/bin/env python

#
#
# ATTENTION!!!                                                    #
# DO NOT follow on a reduction process unless you are sure about  #
# the fiber masks in the MDF file. Disregarding this warning will #
# most certainly lead to major headaches at the final stages of   #
# the reduction.                                                  #
#
#

# Table of images

import os

from pyraf import iraf

import pipe
import pca
from reduction import cal_reduction, wl_lims
from .cube import CubeBuilder


def reduce_science(rawdir, rundir, flat, arc, twilight, twilight_flat, sciimg,
                   starimg, bias, overscan, vardq, observatory, lacos,
                   apply_lacos, lacos_xorder, lacos_yorder, lacos_sigclip,
                   lacos_objlim, bpm, instrument, slits, fl_gscrrej,
                   wltrim_frac, grow_gap, cube_bit_mask):
    """
    Reduction pipeline for standard star.

    Parameters
    ----------
    rawdir: string
        Directory containing raw images.
    rundir: string
        Directory where processed files are saved.
    flat: string
        Names of the files containing flat field images.
    arc: string
        Arc images.
    twilight: string
        Twilight flat images.
    twilight_flat: string
        Flat field for twilight image.
    starimg: string
        Name of the file containing the image to be reduced.
    bias: string
        Bias images.
    grow_gap: number
        Number of pixels by which to grow the bad pixel mask around
        the chip gaps.

    """

    iraf.set(stdimage='imtgmos')

    iraf.gemini()
    iraf.unlearn('gemini')

    iraf.gmos()
    iraf.unlearn('gmos')

    iraf.gemtools()
    iraf.unlearn('gemtools')

    # os.path.isfile('arquivo')

    iraf.unlearn('gemini')
    iraf.unlearn('gmos')

    iraf.task(lacos_spec=lacos)

    # set directories
    iraf.set(caldir=rawdir)  #
    iraf.set(rawdir=rawdir)  # raw files
    iraf.set(procdir=rundir)  # processed files

    iraf.gmos.logfile = 'logfile.log'
    iraf.gfextract.verbose = 'no'

    iraf.cd('procdir')

    flat = flat.replace('.fits', '')
    twilight = twilight.replace('.fits', '')
    twilight_flat = twilight_flat.replace('.fits', '')
    arc = arc.replace('.fits', '')
    starimg = starimg.replace('.fits', '')
    sciimg = sciimg.replace('.fits', '')
    mdffile = 'mdf' + flat + '.fits'

    iraf.gfreduce.bias = 'caldir$' + bias
    iraf.gfreduce.fl_fulldq = 'yes'
    iraf.gfreduce.fl_fixgaps = 'yes'
    iraf.gfreduce.grow = grow_gap
    iraf.gireduce.bpm = 'rawdir$' + bpm
    iraf.gfextract.verbose = 'no'

    cal_reduction(
        rawdir=rawdir, rundir=rundir, flat=flat, arc=arc, twilight=twilight,
        bias=bias, bpm=bpm, overscan=overscan, vardq=vardq,
        instrument=instrument, slits=slits, twilight_flat=twilight_flat,
        grow_gap=grow_gap)
    #
    #   Actually reduce science
    #
    image_name = 'rg' + sciimg + '.fits'
    if os.path.isfile(image_name):
        pipe.skipwarn(image_name)
    else:
        iraf.gfreduce(
            sciimg, slits='header', rawpath='rawdir$', fl_inter='no',
            fl_addmdf='yes', key_mdf='MDF', mdffile=mdffile, weights='no',
            fl_over=overscan, fl_trim='yes', fl_bias='yes', trace='no',
            recenter='no', fl_fulldq='yes',
            fl_flux='no', fl_gscrrej='no', fl_extract='no', fl_gsappwave='no',
            fl_wavtran='no', fl_novl='yes', fl_skysub='no', fl_vardq=vardq,
            mdfdir='procdir$')
    prefix = 'rg'

    # Gemfix
    image_name = 'p' + prefix + sciimg + '.fits'
    if os.path.isfile(image_name):
        pipe.skipwarn(image_name)
    else:
        iraf.gemfix(
            prefix + sciimg, out='p' + prefix + sciimg, method='fit1d',
            bitmask=65535, axis=1)
    prefix = 'p' + prefix

    # LA Cosmic - cosmic ray removal
    if apply_lacos:
        image_name = 'l' + prefix + sciimg + '.fits'
        if os.path.isfile(image_name):
            pipe.skipwarn(image_name)
        else:
            iraf.gemcrspec(
                prefix + sciimg, out='l' + prefix + sciimg, sigfrac=0.5,
                niter=4, fl_vardq=vardq, xorder=lacos_xorder,
                yorder=lacos_yorder, sigclip=lacos_sigclip,
                objlim=lacos_objlim)
        prefix = 'l' + prefix

    if fl_gscrrej:
        image_name = 'ex' + prefix + sciimg + '.fits'
    else:
        image_name = 'e' + prefix + sciimg + '.fits'

    if os.path.isfile(image_name):
        pipe.skipwarn(image_name)
    else:
        iraf.gfreduce(prefix + sciimg, slits='header', rawpath='./', fl_inter='no', fl_addmdf='no', key_mdf='MDF',
                      mdffile=mdffile, fl_over='no', fl_trim='no', fl_bias='no', trace='no', recenter='no',
                      fl_flux='no', fl_gscrrej=fl_gscrrej, fl_extract='yes', fl_gsappwave='yes', fl_wavtran='no',
                      fl_novl='no', fl_skysub='no', grow=grow_gap, reference='eprg' + flat, weights='no',
                      wavtraname='eprg' + arc, response='eprg' + flat + '_response.fits', fl_vardq=vardq,
                      fl_fulldq='yes', fl_fixgaps='yes')

    if fl_gscrrej:
        prefix = 'ex' + prefix
    else:
        prefix = 'e' + prefix

    # if wl2 > 7550.0:
    #     wl2 = 7550.0

    #
    #   Apply wavelength transformation
    #

    wl1, wl2 = wl_lims(prefix + sciimg + '.fits', wltrim_frac)

    image_name = 't' + prefix + sciimg + '.fits'
    if os.path.isfile(image_name):
        pipe.skipwarn(image_name)
    else:
        iraf.gftransform(
            prefix + sciimg, wavtraname='eprg' + arc, fl_vardq=vardq,
            w1=wl1, w2=wl2,
        )

    prefix = 't' + prefix
    #
    #   Sky subtraction
    #
    image_name = 's' + prefix + sciimg + '.fits'
    if os.path.isfile(image_name):
        pipe.skipwarn(image_name)
    else:
        iraf.gfskysub(
            prefix + sciimg, expr='default', combine='median',
            reject='avsigclip', scale='none', zero='none', weight='none',
            sepslits='yes', fl_inter='no', lsigma=1, hsigma=1,
        )

    prefix = 's' + prefix
    #
    #   Apply flux calibration to galaxy
    #
    image_name = 'c' + prefix + sciimg + '.fits'
    if os.path.isfile(image_name):
        pipe.skipwarn(image_name)
    else:
        iraf.gscalibrate(
            prefix + sciimg, sfuncti=starimg,
            extinct='onedstds$ctioextinct.dat',
            observatory=observatory, fluxsca=1, fl_vardq=vardq)
    prefix = 'c' + prefix
    #
    # Remove spurious data with PCA
    #
    image_name = 'x' + prefix + sciimg + '.fits'
    print(os.getcwd())
    print(image_name)
    if os.path.isfile(image_name):
        pipe.skipwarn(image_name)
    else:
        t = pca.Tomography(prefix + sciimg + '.fits')
        t.decompose()
        t.remove_cosmic_rays(sigma_threshold=10.0)
        t.write(image_name)
    prefix = 'x' + prefix
    #
    #   Create data cubes
    #
    image_name = 'd' + prefix + sciimg + '.fits'
    if os.path.isfile(image_name):
        pipe.skipwarn(image_name)
    else:
        data_cube = CubeBuilder(prefix + sciimg + '.fits')
        data_cube.build_cube()
        data_cube.fit_refraction_function()
        data_cube.fix_atmospheric_refraction()
        data_cube.write(image_name)
