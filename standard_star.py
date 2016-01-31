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
import matplotlib.pyplot as plt
import numpy as np
import pyfits as pf
import glob
import time
import os

def reduce_stdstar(rawdir, rundir, caldir, starobj, stdstar, flat,
    arc, twilight, starimg, bias, overscan, vardq, lacos, observatory,
    apply_lacos):
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

    tstart = time.time()

    iraf.set(stdimage='imtgmos')

    iraf.task(lacos_spec=lacos)
    iraf.gemini()
    iraf.gmos()
    iraf.gemtools()
    
    iraf.gmos.logfile='logfile.log'
    iraf.gemtools.gloginit.logfile='logfile.log'

    #set directories
    iraf.set(caldir=rawdir)  # 
    iraf.set(rawdir=rawdir)  # raw files
    iraf.set(procdir=rundir)  # processed files   

    #os.path.isfile('arquivo')
    
    #iraf.unlearn('gemini')
    #iraf.unlearn('gmos')

    iraf.cd('procdir')

    flat = flat.strip('.fits')
    twilight = twilight.strip('.fits')
    arc = arc.strip('.fits')
    starimg = starimg.strip('.fits')

    
    iraf.gfreduce.bias = 'rawdir$'+bias
    #
    #   Flat reduction
    #
    iraf.gfreduce(
        flat+','+twilight, slits='header', rawpath='rawdir$', fl_inter='no',
        fl_addmdf='yes', key_mdf='MDF', mdffile='default', weights='no',
        fl_over=overscan, fl_trim='yes', fl_bias='yes', trace='yes', t_order=4,
        fl_flux='no', fl_gscrrej='no', fl_extract='yes', fl_gsappwave='no',
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
        recenter='no',
        fl_flux='no', fl_gscrrej='no', fl_extract='yes', fl_gsappwave='no',
        fl_wavtran='no', fl_novl='no', fl_skysub='no', reference='erg'+flat,
        fl_vardq='no')
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
    #   Actually reduce star
    #
    iraf.gfreduce(
        starimg, slits='header', rawpath='rawdir$', fl_inter='no',
        fl_addmdf='yes', key_mdf='MDF', mdffile='default', weights='no',
        fl_over=overscan, fl_trim='yes', fl_bias='yes', trace='no',
        recenter='no',
        fl_flux='no', fl_gscrrej='no', fl_extract='no', fl_gsappwave='no',
        fl_wavtran='no', fl_novl='yes', fl_skysub='no', fl_vardq=vardq)
    
    if apply_lacos:
        iraf.gemcrspec(
            'rg'+starimg, out='lrg'+starimg, sigfrac=0.32,
            niter=4, fl_vardq=vardq)
        prefix = 'lrg'
    else:
        prefix = 'rg'

         
    iraf.gfreduce(
        prefix+starimg, slits='header', rawpath='./', fl_inter='no',
        fl_addmdf='no', key_mdf='MDF', mdffile='default',
        fl_over='no', fl_trim='no', fl_bias='no', trace='no',
        recenter='no',
        fl_flux='no', fl_gscrrej='no', fl_extract='yes',
        fl_gsappwave='yes',
        fl_wavtran='yes', fl_novl='no', fl_skysub='yes',
        reference='erg'+flat, weights='no',
        wavtraname='erg'+arc,
        response='erg'+flat+'_response.fits',
        fl_vardq=vardq)
    prefix = 'ste'+prefix
    #
    #   Apsumming the stellar spectra
    #
    iraf.gfapsum(
        prefix+starimg, fl_inter='no', lthreshold=400.,
        reject='avsigclip')
    #
    #   Building sensibility function
    # 
    iraf.gsstandard(
        'a'+prefix+starimg, starname=stdstar, observatory=observatory,
        sfile='std'+starimg, sfunction='sens'+starimg, caldir=caldir)
    #
    #   Apply flux calibration to star
    #
    iraf.gscalibrate(
         prefix+starimg, sfuncti='sens'+starimg, 
         extinct='onedstds$ctioextinct.dat', 
         observatory=observatory, fluxsca=1, fl_vardq=vardq)
    #
    #   Create data cubes
    #
    iraf.gfcube(
         'c'+prefix+starimg, outimage='dcstelrg'+starimg, ssample=.1,
         fl_atmdisp='yes', fl_var=vardq, fl_dq=vardq)

    #
    # Test calibration
    #
    iraf.cd(caldir)
    caldata = np.loadtxt(stdstar+'.dat', unpack=True)
    iraf.cd('procdir')
    calflux = mag2flux(caldata[0], caldata[1])

    iraf.gscalibrate(
        'a'+prefix+starimg, sfuncti='sens'+starimg,
        extinct='onedstds$ctioextinct.dat',
        observatory=observatory, fluxsca=1)

    sumflux = pf.getdata('ca'+prefix+starimg+'.fits', ext=2)
    sumhead = pf.getheader('ca'+prefix+starimg+'.fits', ext=2)
    sumwl = sumhead['crval1'] + np.arange(sumhead['naxis1'])*sumhead['cdelt1']

    plt.close('all')
    plt.plot(sumwl, sumflux, 'b', lw=.5)
    plt.plot(caldata[0], calflux, 'r', lw=1.5)
    plt.xlim(sumwl[0]*.99, sumwl[-1]*1.01)
    plt.ylim(min(calflux)*.8, max(calflux)*1.2)
    plt.savefig('calib'+starimg+'.eps')
    
    tend = time.time()
    
    print('Elapsed time in reduction: {:.2f}'.format(tend - tstart))

def mag2flux(wl, mag):
    """
    Convert magnitube[m_AB] to fna (flux per unit wavelenth
    [ergs/cm/cm/s/A]). First, it converts m_AB to fnu (flux per unit
    frequency [ergs/cm/cm/s/Hz]), using equation from 'standard' task
    help from IRAF. Then, it converts fnu to fna with:
    fna = fnu*c/wl/wl, where c is the speed of ligth in
    angstroms/second and wl is the wavelength in angstroms.

    Parameters
    ----------
    wl: array
        Wavelength in angstrons
    mag: array
        Magnitude from calibration star
    
    Returns
    -------
    fna : array
        Flux per unit wavelength [ergs/cm/cm/s/A]
    """
    fnu = 3.68E-20*10**(-0.4*mag)
    return fnu*2.99792458E18/wl/wl
