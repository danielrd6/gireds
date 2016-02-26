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
import matplotlib.pyplot as plt
import numpy as np
import pyfits as pf
from reduction import cal_reduction


def circular_aperture(image, radius=1):
    """
    Returns a list of aperture numbers to be used by apsum, that
    define a circular pupil of a given radius, centred on the object
    field of GMOS.

    Parameters
    ----------
    image : string
        Name of the file containing the standard star spectra after
        sky subtraction.
    radius : number
        Radius of the circular pupil given in arcseconds.
    """

    hdu = pf.open(image)

    instrument = hdu[0].header['instrume']

    for i in hdu:
        if type(i) is pf.hdu.table.BinTableHDU:
            mdf = i.data
            break

    x, y = mdf['XINST'], mdf['YINST']

    x0 = {'GMOS-N': 3.4, 'GMOS-S': 61.7}
    y0 = 2.45

    r = np.sqrt((x - x0[instrument]) ** 2 + (y - y0) ** 2)

    cond = (mdf['BEAM'] != -1) & (r < 1)
    apids = mdf['APID'][cond]

    apertures = (len(apids) * ' {:d},').format(*apids)

    return apertures


def reduce_stdstar(
        rawdir, rundir, caldir, starobj, stdstar, flat, arc, twilight,
        starimg, bias, overscan, vardq, lacos, observatory, apply_lacos,
        lacos_xorder, lacos_yorder, bpm, instrument, mdffile, slits):
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

    iraf.task(lacos_spec=lacos)

    iraf.gemini()
    iraf.unlearn('gemini')

    iraf.gmos()
    iraf.unlearn('gmos')

    iraf.gemtools()
    iraf.unlearn('gemtools')

    iraf.gmos.logfile = 'logfile.log'
    iraf.gemtools.gloginit.logfile = 'logfile.log'

    # set directories
    iraf.set(caldir=rawdir)  #
    iraf.set(rawdir=rawdir)  # raw files
    iraf.set(procdir=rundir)  # processed files

    # os.path.isfile('arquivo')

    iraf.cd('procdir')

    flat = flat.strip('.fits')
    twilight = twilight.strip('.fits')
    arc = arc.strip('.fits')
    starimg = starimg.strip('.fits')

    iraf.gfreduce.bias = 'rawdir$'+bias
    iraf.gireduce.bpm = 'rawdir$'+bpm

    cal_reduction(
        rawdir=rawdir, rundir=rundir, flat=flat, arc=arc, twilight=twilight,
        bias=bias, bpm=bpm, overscan=overscan, vardq=vardq, mdffile=mdffile,
        instrument=instrument, slits=slits)
    #
    #   Actually reduce star
    #
    iraf.gfreduce(
        starimg, slits='header', rawpath='rawdir$', fl_inter='no',
        fl_addmdf='yes', key_mdf='MDF', mdffile='default', weights='no',
        fl_over=overscan, fl_trim='yes', fl_bias='yes', trace='no',
        recenter='no',
        fl_flux='no', fl_gscrrej='no', fl_extract='no', fl_gsappwave='no',
        fl_wavtran='no', fl_novl='yes', fl_skysub='no', fl_vardq=vardq,
        mdfdir='procdir$')
    prefix = 'rg'

    # Gemfix
    iraf.gemfix(prefix + starimg, out='p' + prefix + starimg, method='fixpix',
                bitmask=1)
    prefix = 'p' + prefix

    if apply_lacos:
        iraf.gemcrspec(
            prefix + starimg, out='l' + prefix + starimg, sigfrac=0.32,
            niter=4, fl_vardq=vardq, xorder=lacos_xorder, yorder=lacos_yorder)
        prefix = 'l' + prefix

    iraf.gfreduce(
        prefix + starimg, slits='header', rawpath='./', fl_inter='no',
        fl_addmdf='no', key_mdf='MDF', mdffile='default', fl_over='no',
        fl_trim='no', fl_bias='no', trace='no', recenter='no', fl_flux='no',
        fl_gscrrej='no', fl_extract='yes', fl_gsappwave='yes',
        fl_wavtran='yes', fl_novl='no', fl_skysub='yes',
        reference='eprg' + flat, weights='no', wavtraname='erg' + arc,
        response='eprg' + flat + '_response.fits', fl_vardq=vardq)
    prefix = 'ste' + prefix
    #
    #   Apsumming the stellar spectra
    #

    xinst = pf.getdata(prefix + starimg + '.fits', ext=1)['XINST']
    if instrument == 'GMOS-N':
        x0 = np.average(xinst[xinst < 10])
    elif instrument == 'GMOS-S':
        x0 = np.average(xinst[xinst > 10])

    ap_expression = '((XINST-{:.2f})**2 + (YINST-2.45)**2)**0.5 < 1'.format(x0)

    iraf.gfapsum(
        prefix + starimg, fl_inter='no', lthreshold=400.,
        reject='avsigclip', expr=ap_expression)
    #
    #   Building sensibility function
    #
    iraf.gsstandard(
        'a' + prefix + starimg, starname=stdstar, observatory=observatory,
        sfile='std' + starimg, sfunction='sens' + starimg, caldir=caldir)
    #
    #   Apply flux calibration to star
    #
    iraf.gscalibrate(
        prefix + starimg, sfuncti='sens' + starimg,
        extinct='onedstds$ctioextinct.dat', observatory=observatory,
        fluxsca=1, fl_vardq=vardq)
    #
    #   Create data cubes
    #
    iraf.gfcube(
        'c' + prefix + starimg, outimage='dc' + prefix + starimg, ssample=.1,
        fl_atmdisp='yes', fl_var=vardq, fl_dq=vardq)

    #
    # Test calibration
    #
    iraf.cd(caldir)
    caldata = np.loadtxt(stdstar + '.dat', unpack=True)
    iraf.cd('procdir')
    calflux = mag2flux(caldata[0], caldata[1])

    iraf.gscalibrate(
        'a' + prefix + starimg, sfuncti='sens' + starimg,
        extinct='onedstds$ctioextinct.dat',
        observatory=observatory, fluxsca=1)

    sumflux = pf.getdata('ca' + prefix + starimg + '.fits', ext=2)
    sumhead = pf.getheader('ca' + prefix + starimg + '.fits', ext=2)
    sumwl = sumhead['crval1'] + np.arange(
        sumhead['naxis1']) * sumhead['cdelt1']

    plt.close('all')
    plt.plot(sumwl, sumflux, 'b', lw=.5)
    plt.plot(caldata[0], calflux, 'r', lw=1.5)
    plt.xlim(sumwl[0] * .99, sumwl[-1] * 1.01)
    plt.ylim(min(calflux) * .8, max(calflux) * 1.2)
    plt.savefig('calib' + starimg + '.eps')


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
    fnu = 3.68E-20 * 10 ** (-0.4 * mag)
    return fnu * 2.99792458E18 / wl / wl
