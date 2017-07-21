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
import numpy as np
import pyfits as pf
# import pkg_resources


def merge_cubes(rawdir, rundir, name, observatory, imgcube, xoff,
                yoff, crval3, cdelt3, cdelt1):
    """
    Merge cubes.

    Parameters
    ----------
    rawdir: string
        Directory containing raw images.
    rundir: string
        Directory where processed files are saved.
    name: string
        Name of the object.
    observatory: string
        Gemini-South/Gemini-North.
    imgcube: list of strings
        Cube file for each object cube.
    xoff: list of floats
        x-offset for each object cube.
    yoff: list of floats
        y-offset for each object cube.
    crval3: list of floats
        crval3 for each object cube.
    cdelt3: list of floats
        cdelt3 for each object cube.
    cdelt1: list of floats
        cdelt1 for each object cube.
    """

    rundir = rundir + '/'

    iraf.set(stdimage='imtgmos')

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

    iraf.cd('procdir')

    #
    #   Creation of file/offset files
    #
    nCubes = len(imgcube)

    in_filesSCI = 'files_' + name + '_SCI'
    in_filesVAR = 'files_' + name + '_VAR'
    in_offset = 'offsets_' + name

    with open(in_filesSCI, 'w') as f:
        for img in imgcube:
            f.write(rundir + img + '[1]' + '\n')
    with open(in_filesVAR, 'w') as f:
        for img in imgcube:
            f.write(rundir + img + '[2]' + '\n')

    # Invert (x,y)offsets if in gemini-north
    sign = -1 if (observatory.lower() == 'gemini-north') else 1
    with open(in_offset, 'w') as f:
        for k in range(nCubes):
            f.write(
                "{:.5f} {:.5f} {:.5f}\n".format(
                    sign * (xoff[k] - xoff[0]) / cdelt1[k],
                    sign * (yoff[k] - yoff[0]) / cdelt1[k],
                    (crval3[k] - crval3[0]) / cdelt3[k]
                    )
                )

    #
    #   Definition of in/output files. And header modification.
    #
    in_sci = [img + '[1]' for img in imgcube]
    in_var = [img + '[2]' for img in imgcube]
    in_dq = [img + '[3]' for img in imgcube]
    out_sci = name + '_SCI.fits'
    out_var = name + '_VAR.fits'
    out_sigIN = name + '_SIGIN.fits'
    out_exp = name + '_EXP'

    # Convert DQ extension to 'pl' and add the its filename to 'bpm' key
    # --- Change to other key. (Other rotines use this key) - Improve
    # --- Change also the key for bpm used by 'fixpix' ------ Improve
    out_dqPL = [img[:-5] + '_DQ.pl' for img in imgcube]

    for k in range(nCubes):
        print(in_sci[k], in_var[k], in_dq[k], out_dqPL[k])
        iraf.imcopy(in_dq[k],  out_dqPL[k])
        iraf.hedit(in_sci[k], 'BPM', out_dqPL[k], add='yes', verify='no')
        iraf.hedit(in_var[k], 'BPM', out_dqPL[k], add='yes', verify='no')

    #
    #   Merge sci/var cubes
    #
    iraf.imcombine("@" + in_filesSCI, out_sci, offsets=in_offset,
                   combine='average', reject='avsigclip', masktype='goodvalue',
                   maskvalue=0, expmasks=out_exp, sigmas=out_sigIN)

    iraf.imcombine("@" + in_filesVAR, out_var, offsets=in_offset,
                   combine='sum', reject='none', masktype='goodvalue',
                   maskvalue=0)

    #
    #   Criate correct error cube
    #
    iraf.imcopy(out_exp + '.pl', out_exp.replace('.pl', '.fits'))

    # Read cubes
    sci_cube = pf.getdata(out_sci)
    var_cube = pf.getdata(out_var)
    sigIN_cube = pf.getdata(out_sigIN)
    exp_cube = pf.getdata(out_exp + '.fits')

    # --- Identify problem with negative values ---- Improve
    # RuntimeWarning: invalid value encountered in divide
    exp_MASK = np.ma.array(exp_cube, mask=(exp_cube == 0))
    err_cube = np.sqrt(abs(var_cube / exp_MASK ** 2).data)

    #
    #   Criate hypercube
    #
    # ---- Maybe don't need header for each extension -- Improve
    pry = pf.PrimaryHDU(header=pf.getheader(out_sci))
    hdu1 = pf.ImageHDU(sci_cube, header=pf.getheader(out_sci), name='SCI')
    hdu2 = pf.ImageHDU(err_cube, header=pf.getheader(out_var), name='ERR')
    hdu4 = pf.ImageHDU(sigIN_cube, header=pf.getheader(out_sigIN),
                       name='SIG_IN')
    hdu3 = pf.ImageHDU(exp_cube, header=pf.getheader(out_exp + '.fits'),
                       name='NCUBE')

    hdu = pf.HDUList([pry, hdu1, hdu2, hdu3, hdu4])
    hdu.writeto(name + '_HYPERCUBE.fits')
