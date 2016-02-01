#!/usr/bin/env python

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
import glob
import time
import os

def reduce_science(rawdir, rundir, flat, arc, twilight, sciimg,
        starimg, bias, overscan, vardq, observatory, lacos, apply_lacos):
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
    iraf.gemtools()
    iraf.gmos()

    #os.path.isfile('arquivo')
    
    #iraf.unlearn('gemini')
    #iraf.unlearn('gmos')
    
    iraf.task(lacos_spec=lacos)
    
    tstart = time.time()
    
    #set directories
    iraf.set(caldir=rawdir)  # 
    iraf.set(rawdir=rawdir)  # raw files
    iraf.set(procdir=rundir)  # processed files
    
    iraf.gmos.logfile='logfile.log'
    
    iraf.cd('procdir')

    flat = flat.strip('.fits')
    twilight = twilight.strip('.fits')   
    arc = arc.strip('.fits')
    starimg = starimg.strip('.fits')
    sciimg = sciimg.strip('.fits')
    iraf.gfreduce.bias = 'caldir$'+bias
    
    #
    #   Flat reduction
    #
    
    iraf.gfreduce(
        flat+','+twilight, slits='header', rawpath='rawdir$',
        fl_inter='no',
        fl_addmdf='yes', key_mdf='MDF', mdffile='default', weights='no',
        fl_over=overscan, fl_trim='yes', fl_bias='yes',
        trace='yes', t_order=4,
        fl_flux='no', fl_gscrrej='no', fl_extract='yes',
        fl_gsappwave='no',
        fl_wavtran='no', fl_novl='no', fl_skysub='no', reference='',
        recenter='yes', fl_vardq=vardq)
    #
    #   Response function
    #
    iraf.gfresponse(
        'erg'+flat, out='erg'+flat+'_response',
        skyimage='erg'+twilight, order=95, fl_inter='no', func='spline3',
        sample='*', verbose='yes')
    #
    #   Arc reduction
    #
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
    
    iraf.gswavelength(
        'erg'+arc, function='chebyshev', nsum=15, order=4, fl_inter='no',
        nlost=5, ntarget=20, aiddebug='s', threshold=5, section='middle line')
    #
    #   Apply wavelength solution to the lamp 2D spectra
    #
    iraf.gftransform(
        'erg'+arc, wavtran='erg'+arc, outpref='t', fl_vardq='no')
    #
    #   Actually reduce science
    #
    iraf.gfreduce(
        sciimg, slits='header', rawpath='rawdir$', fl_inter='no',
        fl_addmdf='yes', key_mdf='MDF', mdffile='default', weights='no',
        fl_over=overscan, fl_trim='yes', fl_bias='yes', trace='no',
        recenter='no', fl_flux='no', fl_gscrrej='no', fl_extract='no', 
        fl_gsappwave='no', fl_wavtran='no', fl_novl='yes',
        fl_skysub='no', fl_vardq=vardq)
    
    if apply_lacos:
        iraf.gemcrspec(
            'rg'+sciimg, out='lrg'+sciimg, sigfrac=0.32, 
            niter=4, fl_vardq=vardq)
        prefix = 'lrg'
    else:
        prefix = 'rg'

    iraf.gfreduce(
        prefix+sciimg, slits='header', rawpath='./', fl_inter='no',
        fl_addmdf='no', key_mdf='MDF', mdffile='default',
        fl_over='no', fl_trim='no', fl_bias='no', trace='no',
        recenter='no', fl_flux='no', fl_gscrrej='no', fl_extract='yes',
        fl_gsappwave='yes', fl_wavtran='yes', fl_novl='no',
        fl_skysub='yes',
        reference='erg'+flat, weights='no',
        wavtraname='erg'+arc,
        response='erg'+flat+'_response.fits', fl_vardq=vardq)
    prefix = 'ste'+prefix
    #
    #   Apply flux calibration to galaxy
    #
    iraf.gscalibrate(
         prefix+sciimg, sfuncti='sens'+starimg, 
         extinct='onedstds$ctioextinct.dat', 
         observatory=observatory, fluxsca=1, fl_vardq=vardq)
    prefix = 'c'+prefix
    #
    #   Create data cubes
    #
    iraf.gfcube(
         prefix+sciimg, outimage='d'+prefix+sciimg, ssample=.1, 
         fl_atmdisp='yes', fl_var=vardq, fl_dq=vardq)
    
    tend = time.time()
    
    print('Elapsed time in reduction: {:.2f}'.format(tend - tstart))
