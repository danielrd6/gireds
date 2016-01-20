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

def reduce_stdstar(rawdir, rundir, caldir, starobj, stdstar, flat,
    arc, twilight, starimg, bias, overscan, vardq):
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
    
    #iraf.unlearn('gemini')
    #iraf.unlearn('gmos')
    
    iraf.task(lacos_spec='/storage/work/gemini_pairs/lacos_spec.cl')
    
    tstart = time.time()
    
    #set directories
    iraf.set(caldir=rawdir)  # 
    iraf.set(rawdir=rawdir)  # raw files
    iraf.set(procdir=rundir)  # processed files
    
    iraf.gmos.logfile='logfile.log'
    
    iraf.cd('procdir')
    
    # building lists
    
    def range_string(l):
        return (len(l)*'{:4s},').format(*[i[-9:-5] for i in l])
    
    iraf.gemlist(range=range_string(flat), root=flat[0][:-9],
        Stdout='flat.list')
    iraf.gemlist(range=range_string(arc), root=arc[0][:-9],
        Stdout='arc.list')
    #iraf.gemlist(range=range_string(star), root=star[0][:-4],
    #    Stdout='star.list')
    iraf.gemlist(range=range_string(twilight),
        root=twilight[0][:-9], Stdout='twilight.list')
    
    iraf.gfreduce.bias = 'caldir$'+bias[0]
    
    #######################################################################
    #######################################################################
    ###   Star reduction                                                  #
    #######################################################################
    #######################################################################
    
    #
    #   Flat reduction
    #
    
    iraf.gfreduce(
        '@flat.list', slits='header', rawpath='rawdir$', fl_inter='no',
        fl_addmdf='yes', key_mdf='MDF', mdffile='default', weights='no',
        fl_over=overscan, fl_trim='yes', fl_bias='yes', trace='yes', t_order=4,
        fl_flux='no', fl_gscrrej='no', fl_extract='yes', fl_gsappwave='no',
        fl_wavtran='no', fl_novl='no', fl_skysub='no', reference='',
        recenter='yes', fl_vardq=vardq)
    
    iraf.gfreduce('@twilight.list', slits='header', rawpath='rawdir$',
        fl_inter='no', fl_addmdf='yes', key_mdf='MDF',
        mdffile='default', weights='no',
        fl_over=overscan, fl_trim='yes', fl_bias='yes', trace='yes',
        recenter='no',
        fl_flux='no', fl_gscrrej='no', fl_extract='yes', fl_gsappwave='no',
        fl_wavtran='no', fl_novl='no', fl_skysub='no',
        reference='erg'+flat[0], fl_vardq=vardq)
    #
    #   Response function
    #
    
    
    for i, j in enumerate(flat):

        j = j[:-5]
    
        iraf.imdelete(j+'_response')
        iraf.gfresponse('erg'+j+'.fits', out='erg'+j+'_response',
            skyimage='erg'+twilight[i], order=95, fl_inter='no',
            func='spline3',
            sample='*', verbose='yes')
    
    #   Arc reduction
    #
    
    iraf.gfreduce(
        '@arc.list', slits='header', rawpath='rawdir$', fl_inter='no',
        fl_addmdf='yes', key_mdf='MDF', mdffile='default', weights='no',
        fl_over=overscan, fl_trim='yes', fl_bias='yes', trace='no',
        recenter='no',
        fl_flux='no', fl_gscrrej='no', fl_extract='yes', fl_gsappwave='no',
        fl_wavtran='no', fl_novl='no', fl_skysub='no', reference='erg'+flat[0],
        fl_vardq=vardq)
    
    
    #   Finding wavelength solution
    #   Note: the automatic identification is very good
    #
    
    for i in arc:
        
        iraf.gswavelength('erg'+i, function='chebyshev', nsum=15, order=4,
            fl_inter='no', nlost=5, ntarget=20, aiddebug='s', threshold=5,
            section='middle line')
    
    #
    #   Apply wavelength solution to the lamp 2D spectra
    #
    
        iraf.gftransform('erg'+i, wavtran='erg'+i, outpref='t', fl_vardq=vardq)
    
    ##
    ##   Actually reduce star
    ##
    
    
    iraf.gfreduce(
        starimg, slits='header', rawpath='rawdir$', fl_inter='no',
        fl_addmdf='yes', key_mdf='MDF', mdffile='default', weights='no',
        fl_over=overscan, fl_trim='yes', fl_bias='yes', trace='no',
        recenter='no',
        fl_flux='no', fl_gscrrej='no', fl_extract='no', fl_gsappwave='no',
        fl_wavtran='no', fl_novl='yes', fl_skysub='no', fl_vardq=vardq)
    
    iraf.gemcrspec('rg{:s}'.format(starimg), out='lrg'+starimg, sigfrac=0.32, 
         niter=4, fl_vardq=vardq)
         
    iraf.gfreduce(
        'lrg'+starimg, slits='header', rawpath='./', fl_inter='no',
        fl_addmdf='no', key_mdf='MDF', mdffile='default',
        fl_over='no', fl_trim='no', fl_bias='no', trace='no',
        recenter='no',
        fl_flux='no', fl_gscrrej='no', fl_extract='yes',
        fl_gsappwave='yes',
        fl_wavtran='yes', fl_novl='no', fl_skysub='yes',
        reference='erg'+flat[0][:-5], weights='no',
        wavtraname='erg'+arc[0][:-5],
        response='erg'+flat[0][:-5]+'_response.fits',
        fl_vardq=vardq)
    #
    #   Apsumming the stellar spectra
    #
    iraf.gfapsum(
        'stexlrg'+starimg, fl_inter='no', lthreshold=400.,
        reject='avsigclip')
    #
    #   Building sensibility function
    #
    
    
    iraf.gsstandard(
        ('astexlrg{:s}').format(starimg), starname=stdstar,
        observatory='Gemini-South', sfile='std', sfunction='sens',
        caldir=caldir)
    #
    #   Apply flux calibration to galaxy
    #
    #
    ##iraf.imdelete('cstexlrg@objr4.list')
    #
    ##iraf.gscalibrate('stexlrg@objr4.list',sfunction='sens.fits',fl_ext='yes',extinct='onedstds$ctioextinct.dat',observatory='Gemini-South',fluxsca=1)
    #
    ##
    ##   Create data cubes
    ##
    #
    #
    ##for i in objs:
    ##  iraf.imdelete('d0.1cstexlrg'+i+'.fits')
    ##  iraf.gfcube('cstexlrg'+i+'.fits',outpref='d0.1',ssample=0.1,fl_atmd='yes',fl_flux='yes')
    #
    ##
    ## Combine cubes
    ##
    #
    #
    ##iraf.imdelete('am2306-721r4_wcsoffsets.fits')
    ##iraf.imcombine('d0.1cstexlrgS20141113S00??.fits[1]',output='am2306-721r4_wcsoffsets.fits',combine='average',reject='sigclip',masktype='badvalue',lsigma=2,hsigma=2,offset='wcs',outlimits='2 67 2 48 100 1795')
    #
    
    tend = time.time()
    
    print('Elapsed time in reduction: {:.2f}'.format(tend - tstart))
    
