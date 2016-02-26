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
import os


def cal_reduction(rawdir, rundir, flat, arc, twilight, bias, bpm, overscan,
                  vardq, mdfdir):
    """
    Reduction pipeline for basic calibration images.

    Parameters
    ----------
    rawdir: string
        Directory containing raw images.
    rundi: string
        Directory where processed files are saved.
    flat: string
        Name of the file containing flat field image.
    arc: string
        Arc image.
    twilight: string
        Twilight image.
    bias: string
        Bias image.
    """

    iraf.set(stdimage='imtgmos')

    iraf.gemini()
    iraf.unlearn('gemini')

    iraf.gmos()
    iraf.unlearn('gmos')

    iraf.gemtools()
    iraf.unlearn('gemtools')

    # set directories
    iraf.set(rawdir=rawdir)  # raw files
    iraf.set(procdir=rundir)  # processed files

    iraf.gmos.logfile = 'logfile.log'

    iraf.cd('procdir')

    flat = flat.strip('.fits')
    twilight = twilight.strip('.fits')
    arc = arc.strip('.fits')
    iraf.gfreduce.bias = 'caldir$' + bias
    iraf.gireduce.bpm = 'rawdir$' + bpm

    #
    #   Flat reduction
    #
    if not os.path.isfile('eprg' + flat + '.fits'):
        iraf.gfreduce(
            flat, slits='header', rawpath='rawdir$', fl_inter='no',
            fl_addmdf='yes', key_mdf='MDF', mdffile='default', weights='no',
            fl_over=overscan, fl_trim='yes', fl_bias='yes', trace='no',
            fl_flux='no', fl_gscrrej='no', fl_extract='no', fl_gsappwave='no',
            fl_wavtran='no', fl_novl='no', fl_skysub='no',
            recenter='no', fl_vardq=vardq, mdfdir=mdfdir)

        # Gemfix
        iraf.gemfix('rg' + flat, out='prg' + flat, method='fixpix',
                    bitmask=1)

        iraf.gfreduce(
            'prg' + flat, slits='header', rawpath='./', fl_inter='no',
            fl_addmdf='no', key_mdf='MDF', mdffile='default', weights='no',
            fl_over='no', fl_trim='no', fl_bias='no', trace='yes',
            t_order=4, fl_flux='no', fl_gscrrej='no', fl_extract='yes',
            fl_gsappwave='no', fl_wavtran='no', fl_novl='no', fl_skysub='no',
            reference='', recenter='yes', fl_vardq=vardq)

    #
    #   The twilight always has to match exactly the extraction of the
    #   flat field image, therefore it must be re-reduced for every
    #   new exposure requiring a flat.
    #
    if os.path.isfile('eprg' + twilight + '.fits'):
        iraf.delete('*' + twilight + '.fits')
        iraf.delete('./database/ap*' + twilight + '*')

    iraf.gfreduce(
        twilight, slits='header', rawpath='rawdir$', fl_inter='no',
        fl_addmdf='yes', key_mdf='MDF', mdffile='default', weights='no',
        fl_over=overscan, fl_trim='yes', fl_bias='yes', trace='no',
        fl_flux='no', fl_gscrrej='no', fl_extract='no', fl_gsappwave='no',
        fl_wavtran='no', fl_novl='no', fl_skysub='no',
        recenter='no', fl_vardq=vardq, mdfdir=mdfdir)

    # Gemfix
    iraf.gemfix('rg' + twilight, out='prg' + twilight, method='fixpix',
                bitmask=1)

    iraf.gfreduce(
        'prg' + twilight, slits='header', rawpath='./', fl_inter='no',
        fl_addmdf='no', key_mdf='MDF', mdffile='default', weights='no',
        fl_over='no', fl_trim='no', fl_bias='no', trace='yes',
        t_order=4, fl_flux='no', fl_gscrrej='no', fl_extract='yes',
        fl_gsappwave='no', fl_wavtran='no', fl_novl='no', fl_skysub='no',
        reference='eprg' + flat, recenter='no', fl_vardq=vardq)

    #
    #   Response function
    #
    if not os.path.isfile('eprg' + flat + '_response.fits'):
        iraf.gfresponse(
            'eprg' + flat, out='eprg' + flat + '_response',
            skyimage='eprg' + twilight, order=95, fl_inter='no',
            func='spline3', sample='*', verbose='yes')
    #
    #   Arc reduction
    #
    if not os.path.isfile('erg' + arc + '.fits'):
        iraf.gfreduce(
            arc, slits='header', rawpath='rawdir$', fl_inter='no',
            fl_addmdf='yes', key_mdf='MDF', mdffile='default', weights='no',
            fl_over=overscan, fl_trim='yes', fl_bias='no', trace='no',
            recenter='no', fl_flux='no', fl_gscrrej='no', fl_extract='yes',
            fl_gsappwave='no', fl_wavtran='no', fl_novl='no', fl_skysub='no',
            reference='eprg' + flat, fl_vardq='no', mdfdir=mdfdir)
    #
    #   Finding wavelength solution
    #   Note: the automatic identification is very good
    #
    if not os.path.isfile('./database/iderg' + arc + '_001'):
        iraf.gswavelength(
            'erg' + arc, function='chebyshev', nsum=15, order=4, fl_inter='no',
            nlost=5, ntarget=20, aiddebug='s', threshold=5,
            section='middle line')
    #
    #   Apply wavelength solution to the lamp 2D spectra
    #
    if not os.path.isfile('teprg' + arc):
        iraf.gftransform(
            'erg' + arc, wavtran='erg' + arc, outpref='t', fl_vardq='no')

    return mdfdir
