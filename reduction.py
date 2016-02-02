#########################################################################
#                                                                       #
#   ATTENTION!!!                                                        #
#       DO NOT follow on a reduction process unless you are sure about  #
#       the fiber masks in the MDF file. Disregarding this warning will #
#       most certainly lead to major headaches at the final stages of   #
#       the reduction.                                                  #
#                                                                       #
#########################################################################

# Table of images

from pyraf import iraf
import numpy as np
import pyfits as pf
import os

def cal_reduction(rawdir, rundir, flat, arc, twilight, bias, overscan,
        vardq):
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
    iraf.gemtools()
    iraf.gmos()

    #iraf.unlearn('gemini')
    #iraf.unlearn('gmos')
    
    iraf.task(lacos_spec=lacos)
    
    #set directories
    iraf.set(rawdir=rawdir)  # raw files
    iraf.set(procdir=rundir)  # processed files
    
    iraf.gmos.logfile='logfile.log'
    
    iraf.cd('procdir')

    flat = flat.strip('.fits')
    twilight = twilight.strip('.fits')   
    arc = arc.strip('.fits')
    iraf.gfreduce.bias = 'caldir$'+bias
    
    #
    #   Flat reduction
    #
    if not os.path.isfile('erg'+flat+'.fits'):
        iraf.gfreduce(
            flat, slits='header', rawpath='rawdir$', fl_inter='no',
            fl_addmdf='yes', key_mdf='MDF', mdffile='default', weights='no',
            fl_over=overscan, fl_trim='yes', fl_bias='yes', trace='yes',
            t_order=4, fl_flux='no', fl_gscrrej='no', fl_extract='yes',
            fl_gsappwave='no', fl_wavtran='no', fl_novl='no', fl_skysub='no',
            reference='', recenter='yes', fl_vardq=vardq)
    if not os.path.isfile('erg'+twilight+'.fits'):
        iraf.gfreduce(
            twilight, slits='header', rawpath='rawdir$', fl_inter='no',
            fl_addmdf='yes', key_mdf='MDF', mdffile='default', weights='no',
            fl_over=overscan, fl_trim='yes', fl_bias='yes', trace='no',
            t_order=4, fl_flux='no', fl_gscrrej='no', fl_extract='yes',
            fl_gsappwave='no', fl_wavtran='no', fl_novl='no', fl_skysub='no',
            reference='erg'+flat, recenter='no', fl_vardq=vardq)
    #
    #   Response function
    #
    if not os.path.isfile('erg'+flat+'_response.fits'):
        iraf.gfresponse(
            'erg'+flat, out='erg'+flat+'_response', skyimage='erg'+twilight,
            order=95, fl_inter='no', func='spline3', sample='*', verbose='yes')
    #
    #   Arc reduction
    #
    if not os.path.isfile('erg'+arc+'.fits'):
        iraf.gfreduce(
            arc, slits='header', rawpath='rawdir$', fl_inter='no',
            fl_addmdf='yes', key_mdf='MDF', mdffile='default', weights='no',
            fl_over=overscan, fl_trim='yes', fl_bias='no', trace='no',
            recenter='no', fl_flux='no', fl_gscrrej='no', fl_extract='yes',
            fl_gsappwave='no', fl_wavtran='no', fl_novl='no', fl_skysub='no',
            reference='erg'+flat, fl_vardq='no')
    # 
    #   Finding wavelength solution
    #   Note: the automatic identification is very good
    #
    if not os.path.isfile('./database/iderg'+arc+'_001'):
        iraf.gswavelength(
            'erg'+arc, function='chebyshev', nsum=15, order=4, fl_inter='no',
            nlost=5, ntarget=20, aiddebug='s', threshold=5,
            section='middle line')
    #
    #   Apply wavelength solution to the lamp 2D spectra
    #
    if not os.path.isfile('terg'+arc): 
        iraf.gftransform(
            'erg'+arc, wavtran='erg'+arc, outpref='t', fl_vardq='no')
