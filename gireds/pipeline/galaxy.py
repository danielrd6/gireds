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

from pyraf import iraf
from reduction import cal_reduction, wl_lims
import pipe
import os


def reduce_science(rawdir, rundir, flat, arc, twilight, twilight_flat, sciimg,
                   starimg, bias, overscan, vardq, observatory, lacos,
                   apply_lacos, lacos_xorder, lacos_yorder, bpm, instrument,
                   slits, fl_gscrrej, wltrim_frac, grow_gap):
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
    imagename = 'rg' + sciimg + '.fits'
    if os.path.isfile(imagename):
        pipe.skipwarn(imagename)
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
    imagename = 'p' + prefix + sciimg + '.fits'
    if os.path.isfile(imagename):
        pipe.skipwarn(imagename)
    else:
        iraf.gemfix(
            prefix + sciimg, out='p' + prefix + sciimg, method='fit1d',
            bitmask=65535, axis=1)
    prefix = 'p' + prefix

    # LA Cosmic - cosmic ray removal
    if apply_lacos:
        imagename = 'l' + prefix + sciimg + '.fits'
        if os.path.isfile(imagename):
            pipe.skipwarn(imagename)
        else:
            iraf.gemcrspec(
                prefix + sciimg, out='l' + prefix + sciimg, sigfrac=0.32,
                niter=4, fl_vardq=vardq, xorder=lacos_xorder,
                yorder=lacos_yorder)
        prefix = 'l' + prefix

    if fl_gscrrej:
        imagename = 'ex' + prefix + sciimg + '.fits'
    else:
        imagename = 'e' + prefix + sciimg + '.fits'

    if os.path.isfile(imagename):
        pipe.skipwarn(imagename)
    else:
        iraf.gfreduce(
            prefix + sciimg, slits='header', rawpath='./', fl_inter='no',
            fl_addmdf='no', key_mdf='MDF', mdffile=mdffile, fl_over='no',
            fl_trim='no', fl_bias='no', trace='no', recenter='no',
            fl_flux='no', fl_gscrrej=fl_gscrrej, fl_extract='yes',
            fl_gsappwave='yes', fl_wavtran='no', fl_novl='no', fl_skysub='no',
            grow=grow_gap, reference='eprg' + flat, weights='no',
            wavtraname='erg' + arc, response='eprg' + twilight +
            '_response.fits', fl_vardq=vardq, fl_fulldq='yes',
            fl_fixgaps='yes')

    if fl_gscrrej:
        prefix = 'ex' + prefix
    else:
        prefix = 'e' + prefix

    wl1, wl2 = wl_lims(prefix + sciimg + '.fits', wltrim_frac)
    # if wl2 > 7550.0:
    #     wl2 = 7550.0

    imagename = 'st' + prefix + sciimg + '.fits'
    if os.path.isfile(imagename):
        pipe.skipwarn(imagename)
    else:
        iraf.gfreduce(
            prefix + sciimg, slits='header', rawpath='./', fl_inter='no',
            fl_addmdf='no', key_mdf='MDF', mdffile=mdffile, fl_over='no',
            fl_trim='no', fl_bias='no', trace='no', recenter='no',
            fl_flux='no', fl_gscrrej='no', fl_extract='no', fl_gsappwave='no',
            fl_wavtran='yes', fl_novl='no', fl_skysub='yes', reference='eprg' +
            flat, weights='no', wavtraname='erg' + arc, response='eprg' +
            twilight + '_response.fits', fl_vardq=vardq, w1=wl1, w2=wl2,
            fl_fulldq='yes')

    prefix = 'st' + prefix
    #
    #   Apply flux calibration to galaxy
    #
    imagename = 'c' + prefix + sciimg + '.fits'
    if os.path.isfile(imagename):
        pipe.skipwarn(imagename)
    else:
        iraf.gscalibrate(
            prefix + sciimg, sfuncti=starimg,
            extinct='onedstds$ctioextinct.dat',
            observatory=observatory, fluxsca=1, fl_vardq=vardq)
    prefix = 'c' + prefix
    #
    #   Create data cubes
    #
    #   GFCUBE has a problem when interpolating over the chip gaps, so
    #   the recommended value for bitmask is 8, in order to only interpolate
    #   cosmic-rays and similiar short period variations. Nevertheless,
    #   when building the combined cube, the actual data quality values
    #   are needed, to ignore the bad pixels in one exposure and keep
    #   the good ones from the other exposure. Therefore, in a true
    #   "gambiarra", GFCUBE is run two time, and the correct
    #   dataquality plane is inserted into the correct science data
    #   cube.
    #
    #
    imagename = 'd' + prefix + sciimg + '.fits'
    if os.path.isfile(imagename):
        pipe.skipwarn(imagename)
    else:
        iraf.gfcube(
            prefix + sciimg, outimage='d' + prefix + sciimg, ssample=.1,
            fl_atmdisp='yes', fl_var=vardq, fl_dq=vardq, bitmask=8,
            fl_flux='yes')
        iraf.gfcube(
            prefix + sciimg, outimage='dataquality.fits', ssample=.1,
            fl_atmdisp='yes', fl_var=vardq, fl_dq=vardq, bitmask=65535,
            fl_flux='yes')
        iraf.imcopy(
            'dataquality.fits[DQ]', 'd' + prefix + sciimg + '[DQ, OVERWRITE]')
        iraf.delete('dataquality.fits')
