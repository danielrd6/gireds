#!/usr/bin/env python

#
#
# ATTENTION!!!                                                        #
# DO NOT follow on a reduction process unless you are sure about  #
# the fiber masks in the MDF file. Disregarding this warning will #
# most certainly lead to major headaches at the final stages of   #
# the reduction.                                                  #
#
#

# Table of images

from pyraf import iraf
# import numpy as np
# import pyfits as pf
from reduction import cal_reduction, wl_lims


def reduce_science(rawdir, rundir, flat, arc, twilight, sciimg,
                   starimg, bias, overscan, vardq, observatory, lacos,
                   apply_lacos, lacos_xorder, lacos_yorder, bpm, instrument,
                   slits, fl_gscrrej, wltrim_frac):
    """
    Reduction pipeline for standard star.

    Parameters
    ----------
    rawdir: string
        Directory containing raw images.
    rundi: string
        Directory where processed files are saved.
    caldir: string
        Directory containing standard star calibration files.
    starobj: string
        Object keyword for the star image.
    stdstar: string
        Star name in calibration file.
    flat: list
        Names of the files containing flat field images.
    arc: list
        Arc images.
    twilight: list
        Twilight flat images.
    starimg: string
        Name of the file containing the image to be reduced.
    bias: list
        Bias images.

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

    iraf.cd('procdir')

    flat = flat.strip('.fits')
    twilight = twilight.strip('.fits')
    arc = arc.strip('.fits')
    starimg = starimg.strip('.fits')
    sciimg = sciimg.strip('.fits')
    iraf.gfreduce.bias = 'caldir$' + bias
    iraf.gireduce.bpm = 'rawdir$' + bpm
    mdffile = 'mdf' + flat + '.fits'

    cal_reduction(
        rawdir=rawdir, rundir=rundir, flat=flat, arc=arc, twilight=twilight,
        bias=bias, bpm=bpm, overscan=overscan, vardq=vardq,
        instrument=instrument, slits=slits)
    #
    #   Actually reduce science
    #
    iraf.gfreduce(
        sciimg, slits='header', rawpath='rawdir$', fl_inter='no',
        fl_addmdf='yes', key_mdf='MDF', mdffile=mdffile, weights='no',
        fl_over=overscan, fl_trim='yes', fl_bias='yes', trace='no',
        recenter='no',
        fl_flux='no', fl_gscrrej='no', fl_extract='no', fl_gsappwave='no',
        fl_wavtran='no', fl_novl='yes', fl_skysub='no', fl_vardq=vardq,
        mdfdir='procdir$')
    prefix = 'rg'

    # Gemfix
    iraf.gemfix(prefix + sciimg, out='p' + prefix + sciimg, method='fixpix',
                bitmask=1)
    prefix = 'p' + prefix

    if apply_lacos:
        iraf.gemcrspec(
            prefix + sciimg, out='l' + prefix + sciimg, sigfrac=0.32,
            niter=4, fl_vardq=vardq, xorder=lacos_xorder, yorder=lacos_yorder)
        prefix = 'l' + prefix

    iraf.gfreduce(
        prefix + sciimg, slits='header', rawpath='./', fl_inter='no',
        fl_addmdf='no', key_mdf='MDF', mdffile=mdffile, fl_over='no',
        fl_trim='no', fl_bias='no', trace='no', recenter='no', fl_flux='no',
        fl_gscrrej=fl_gscrrej, fl_extract='yes', fl_gsappwave='yes',
        fl_wavtran='no', fl_novl='no', fl_skysub='no',
        reference='eprg' + flat, weights='no', wavtraname='erg' + arc,
        response='eprg' + flat + '_response.fits', fl_vardq=vardq)
    if fl_gscrrej:
        prefix = 'ex' + prefix
    else:
        prefix = 'e' + prefix

    wl1, wl2 = wl_lims(prefix + sciimg + '.fits', wltrim_frac)
    if wl2 > 7550.0:
        wl2 = 7550.0

    iraf.gfreduce(
        prefix + sciimg, slits='header', rawpath='./', fl_inter='no',
        fl_addmdf='no', key_mdf='MDF', mdffile=mdffile, fl_over='no',
        fl_trim='no', fl_bias='no', trace='no', recenter='no', fl_flux='no',
        fl_gscrrej='no', fl_extract='no', fl_gsappwave='no',
        fl_wavtran='yes', fl_novl='no', fl_skysub='yes',
        reference='eprg' + flat, weights='no', wavtraname='erg' + arc,
        response='eprg' + flat + '_response.fits', fl_vardq=vardq,
        w1=wl1, w2=wl2)

    prefix = 'st' + prefix
    #
    #   Apply flux calibration to galaxy
    #
    iraf.gscalibrate(
        prefix + sciimg, sfuncti='sens' + starimg,
        extinct='onedstds$ctioextinct.dat',
        observatory=observatory, fluxsca=1, fl_vardq=vardq)
    prefix = 'c' + prefix
    #
    #   Create data cubes
    #
    iraf.gfcube(
        prefix + sciimg, outimage='d' + prefix + sciimg, ssample=.1,
        fl_atmdisp='yes', fl_var=vardq, fl_dq=vardq)