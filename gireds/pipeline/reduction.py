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
import random
import warnings

import numpy as np
import pkg_resources
from astropy.io import fits
from pyraf import iraf

from gireds.utils import auto_apertures


def rename_log(logFileName):
    iraf.gloginit(
        logFileName, 'bogus', 'bogus', 'bogus', fl_append='yes')

    if iraf.gloginit.status != 0:
        h = ('_{:32x}'.format(random.getrandbits(256)))[:8] + '.'
        p = logFileName.split('.')
        os.rename(logFileName, p[0] + h + p[1])

    return


def skipwarn(imageName):
    warnText = 'Skipping alread present image {:s}.'.format(imageName)
    warnings.warn(warnText)
    iraf.printlog('GIREDS: ' + warnText, iraf.gmos.logfile, 'yes')

    return


def wl_lims(image, trim_fraction=0.02):
    """
    Evaluates the wavelength trimming limits as a fraction
    of the total wavelength range.

    Parameters
    ----------
    image: string
        Name of the wavelength calibrated image that will be
        transformed.
    trim_fraction: float
        Fraction of the wavelength span to be trimmed at both ends.

    Returns
    -------
    wl0: float
        Lower limit in wavelength.
    wl1: float
        Upper limit in wavelength.
    """

    hdu = fits.open(image)

    nimages = 0
    for i in hdu:
        if i.name == 'SCI':
            nimages += 1

    if nimages == 1:
        h = hdu[2].header
        crval, crpix, dwl = [h[i] for i in ['CRVAL1', 'CRPIX1', 'CD1_1']]
        npix = np.shape(hdu[2].data)[1]
        wl = crval + (np.arange(1, npix + 1, dtype='float32') - crpix) * dwl
    elif nimages == 2:
        h = [hdu[2].header, hdu[3].header]
        crval, crpix, dwl = [(h[0][i], h[1][i]) for i in ['CRVAL1', 'CRPIX1', 'CD1_1']]
        npix = (np.shape(hdu[2].data)[1], np.shape(hdu[3].data)[1])
        wl = np.append(*[crval[i] + (np.arange(1, npix[i] + 1, dtype='float32') - crpix[i]) * dwl[i] for i in range(2)])
    else:
        raise RuntimeError('nimages should be either 0 or 1.')

    wl.sort()

    span = wl[-1] - wl[0]

    wl0 = wl[0] + span * trim_fraction
    wl1 = wl[0] + span * (1.0 - trim_fraction)

    return wl0, wl1


def cal_reduction(rawdir, rundir, flat, arc, twilight, twilight_flat, bias, bpm, overscan, vardq, instrument, slits,
                  grow_gap=1, wltrim_frac=0.03):
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
    iraf.gfextract.verbose = 'no'

    iraf.cd('procdir')

    flat = flat.strip('.fits')
    twilight = twilight.strip('.fits')
    twilight_flat = twilight_flat.strip('.fits')
    arc = arc.strip('.fits')

    iraf.gfreduce.bias = 'caldir$' + bias
    iraf.gfreduce.fl_fulldq = 'yes'
    iraf.gfreduce.fl_fixgaps = 'yes'
    iraf.gfreduce.grow = grow_gap
    iraf.gireduce.bpm = 'rawdir$' + bpm
    iraf.gfextract.verbose = 'yes'

    #
    #   Flat reduction
    #
    imageName = 'eprg' + flat + '.fits'
    if os.path.isfile(imageName):
        skipwarn(imageName)
    else:
        imageName = flat + '.fits'
        for j in ['g', 'r', 'p']:
            imageName = j + imageName
            if os.path.isfile(imageName):
                iraf.printlog('GIREDS: WARNING: Removing file {:s}'.format(imageName), iraf.gmos.logfile, 'yes')
                iraf.delete(imageName)

        mdffile = 'mdf' + flat + '.fits'
        iraf.gfreduce(flat, slits='header', rawpath='rawdir$', fl_inter='no', fl_addmdf='yes', key_mdf='MDF',
                      mdffile='default', weights='no', fl_over=overscan, fl_trim='yes', fl_bias='yes', trace='no',
                      fl_flux='no', fl_gscrrej='no', fl_extract='no', fl_gsappwave='no', fl_wavtran='no', fl_novl='no',
                      fl_skysub='no', recenter='no', fl_vardq='yes', fl_fulldq='yes', mdfdir='gmos$data/')

        # Gemfix
        iraf.gemfix('rg' + flat, out='prg' + flat, method='fit1d', bitmask=1, axis=1, logfile=iraf.gmos.logfile)

        #
        #   Extract spectra
        #
        rename_log(iraf.gmos.logfile)

        ap = auto_apertures.AutoApertures('prg' + flat + '.fits')
        ap.find_dead_beams()
        ap.fix_mdf()

        extract_args = {'inimage': 'prg' + flat, 'exslits': '*', 'trace': 'yes', 'recenter': 'yes', 'order': 9,
                        't_nsum': 50, 'function': 'chebyshev', 'fl_novl': 'no', 'fl_fulldq': vardq,
                        'fl_gnsskysub': 'no', 'fl_fixnc': 'no', 'fl_fixgaps': 'yes', 'fl_vardq': 'yes',
                        'grow': grow_gap, 'fl_inter': 'no', 'verbose': 'no'}

        iraf.gfextract(**extract_args)

        time_out = 0
        while (ap.check_iraf('database/apeprg' + flat) != 0) and (time_out < 5):
            ap.fix_mdf()

            iraf.printlog('Aperture iteration #{:d}.'.format(time_out), iraf.gmos.logfile, 'yes')

            iraf.delete('eprg' + flat + '.fits')
            iraf.delete('database/apeprg' + flat + '*')

            iraf.gfextract(**extract_args)

            time_out += 1

        ap.write_mdf_file(mdffile)

    #
    # Twilight
    #
    # mdffile = 'mdf' + flat + '.fits'

    # imageName = 'eprg' + twilight + '.fits'
    # if os.path.isfile(imageName):
    #     skipwarn(imageName)
    # else:
    #     rename_log('extract.log')
    #     iraf.gfreduce(
    #         twilight, slits='header', rawpath='rawdir$', fl_inter='no',
    #         fl_addmdf='yes', key_mdf='MDF', mdffile=mdffile, weights='no',
    #         fl_over=overscan, fl_trim='yes', fl_bias='yes', trace='yes',
    #         fl_flux='no', fl_gscrrej='no', fl_extract='no', fl_gsappwave='no',
    #         fl_wavtran='no', fl_novl='no', fl_skysub='no',
    #         recenter='yes', fl_vardq=vardq, mdfdir='procdir$')
    #     # try:
    #     #     h = fits.open('rg' + i + '.fits')
    #     #     a = h['DQ', 1]
    #     #     del(a)
    #     # except KeyError:
    #     #     return
    #     # with fits.open('rg' + i + '.fits') as h:
    #     #     if 'DQ' not in h:
    #     #         return
    #     # del(h)
    #     # Gemfix
    #     iraf.gemfix('rg' + twilight, out='prg' + twilight, method='fixpix',
    #                 bitmask=1)

    #     rename_log(iraf.gmos.logfile)
    #     iraf.gfreduce(
    #         'prg' + twilight, slits='header', rawpath='./', fl_inter='no',
    #         fl_addmdf='no', key_mdf='MDF', mdffile=mdffile, weights='no',
    #         fl_over='no', fl_trim='no', fl_bias='no', trace='yes',
    #         t_order=4, fl_flux='no', fl_gscrrej='no', fl_extract='yes',
    #         fl_gsappwave='no', fl_wavtran='no', fl_novl='no', fl_skysub='no',
    #         reference='eprg' + flat, recenter='yes', fl_vardq=vardq,
    #         fl_fixgaps='yes', grow=grow_gap)
    #
    #   Response function
    #
    # NOTE: The twilight is optional in the gfresponse function, and
    # serves only to remove possible patterns in the IFU element
    # illumination. However, since the IFU has a very small field of
    # view the flat lamp should be flat enough.
    imageName = 'eprg' + flat + '_response.fits'
    if os.path.isfile(imageName):
        skipwarn(imageName)
    else:
        iraf.gfresponse('eprg' + flat, out='eprg' + flat + '_response', order=45, fl_inter='no', func='spline3',
                        sample='*', verbose='no', fl_fit='yes')
    #
    #   Arc reduction
    #
    imageName = 'eprg' + arc + '.fits'
    if os.path.isfile(imageName):
        skipwarn(imageName)
    else:
        iraf.gfreduce(
            arc, slits='header', rawpath='rawdir$', fl_inter='no',
            fl_addmdf='yes', key_mdf='MDF', mdffile=mdffile, weights='no',
            fl_over=overscan, fl_trim='yes', fl_bias='yes', trace='no',
            recenter='no', fl_flux='no', fl_gscrrej='no', fl_extract='no',
            fl_gsappwave='no', fl_wavtran='no', fl_novl='no', fl_skysub='no',
            reference='eprg' + flat, fl_vardq='yes', mdfdir='procdir$')
        iraf.gemfix('rg' + arc, out='prg' + arc, method='fixpix',
                    bitmask=1)
        iraf.gfextract(
            'prg' + arc, exslits='*', trace='no', recenter='no', order=4,
            t_nsum=10, fl_novl='no', fl_fulldq=vardq, fl_gnsskysub='no',
            fl_fixnc='no', fl_fixgaps='yes', fl_vardq=vardq, grow=grow_gap,
            fl_inter='no', reference='eprg' + flat,
            response='eprg' + flat + '_response')
    #
    #   Finding wavelength solution
    #   Note: the automatic identification is very good
    #
    imageName = './database/id' + 'eprg' + arc + '_001'
    if not os.path.isfile(imageName):
        iraf.gswavelength(
            'eprg' + arc, function='chebyshev', nsum=15, order=4,
            fl_inter='no', nlost=20, ntarget=20, aiddebug='s', threshold=5,
            section='middle line')
    else:
        skipwarn(imageName)
    #
    #   Apply wavelength solution to the lamp 2D spectra
    #
    wl1, wl2 = wl_lims('eprg' + arc + '.fits', wltrim_frac)

    imageName = 'teprg' + arc + '.fits'
    if os.path.isfile(imageName):
        skipwarn(imageName)
    else:
        iraf.gftransform(
            'eprg' + arc, wavtran='eprg' + arc, outpref='t', fl_vardq='no',
            w1=wl1, w2=wl2)


def apertures(flat, vardq, mdffile, overscan, instrument, slits):
    """
    Check the aperture solutions from GFEXTRACT and try to fix the
    problems if it's the case.

    Generally there's a few apertures with problems that, in
    consequence, shift others apertures to left/right, creating other
    errors. So, the strategy is to find this few apertures (one at
    time), mask/unamask it, and rerun GFEXTRACT to verify if the
    problems were solved.

    With the information from the "aperg" file (output from), the
    function separates the apertures in two categories: inside the
    the blocks of fibers and in the gaps between them. Analising this
    gap apertures, it identify if this are left/right shifted.

    To look for apertures with problems caused by a bad mdf, a
    calculation of the expected separation between apertures are made.
    This calculation is done using the medians of the separation
    between apertures (one for inside and other for gap apertures).
    Based in the residual of expected and the real separation, it
    can be identifyed dead apertures (or very weak) that are not
    masked in the mdf, good apertures that are masked, and fibers that
    are been identifyed by two apertures.

    Other searched error are the case when the first aperture are left
    shifted cause it is identifying the "bump" in the left of the first
    block. It is assumed that this error are consequence of other that
    propagate all the apertures to the left.

    Unnusual separation values: ...

    In the case of two slits, it were noted that the last apertures are
    contaminated, and should be masked. This case is abranged.

    With the informations of the shifts and the errors encountered, the
    function decides which solution it will apply. For now, it only
    mask/unmask apertures in the mdf. The decision of which solution to
    apply are based in pattern found in previous limited tests. It
    should be noted that many problems may not be abranged by this
    function. For while, it rerun GFEXTRACT with the default mdf (in the
    future it may have the option of do it interactively).

    Cases not tested: -slit=blue
    Cases not fixed: -when there are two errors that propagate to the
                     borders of the slits.
                     -detailed solution for unnusual values.
                     -add more

    Step followed:
    1 Identify shifts/errors.
    2 Apply solution if founded.
    3 In the case of two slit, repeat 1.
    4 If result is GOOD, exit.
    5 Rerun GFEXTRACT.
    6 If there's no solution or the iteration limit was achieved, exit.
    7 Repeat 1, to see if the errors were fixed

    Parameters
    ----------
    flat: sting
        Name of the file containing flat field image.
    vardq: string
        Use variance and DQ planes? [yes/no]
    mdffile: string
        Filename of the MDF.
    overscan: string
        Apply overscan correction? [yes/no]
    instrument: string
        Instrument [GMOS-S/GMOS-N].

    Bugs
    ----
    Some errors may not be identifyed as result of limited tests
    performed.
    Example: No tests were made for the case slits=blue.
    """
    # Read default mdf used before this function
    mdfDefaultData = fits.getdata('prg' + flat + '.fits', extname='MDF')

    # Number of slits
    if slits == 'both':
        numSlit = 2
    if (slits == 'red') or (slits == 'blue'):
        numSlit = 1

    # Start iteration
    isIterating = True
    resultError = False
    noSolution = False
    nIter = 0
    while isIterating:
        nIter += 1
        iraf.printlog(80 * "-", iraf.gmos.logfile, 'yes')
        iraf.printlog(29 * " " + "APERTURES", iraf.gmos.logfile, 'yes')
        iraf.printlog(
            28 * " " + "Iteration " + str(nIter), iraf.gmos.logfile, 'yes')

        reidentify_out = 0
        isGood_out = 0
        for slitNo in range(1, 1 + numSlit):
            # Read mdf data and create dictionary used in the iteration.
            mdf = {'No': 0, 'beam': 0, 'modify': False, 'reidentify': False,
                   'interactive': 'no', 'slits': slits, 'instr': instrument}
            mdfFlatData = fits.getdata('prg' + flat + '.fits', extname='MDF')
            mdfSlit = mdfFlatData[750 * (slitNo - 1):750 * slitNo]

            # Read center/aperture info from aperg* file
            apergFile = 'database/apeprg' + flat + '_' + str(slitNo)
            with open(apergFile, 'r') as f:
                lines = f.readlines()
            aperg_info = np.array([(i.split()[3], i.split()[5]) for i in lines
                                   if 'begin' in i], dtype=float)

            # Define structured array
            infoname = ['No', 'center', 'dNo', 'dCenter', 'No_next',
                        'expected', 'residual', 'where', 'error', 'errType',
                        'leftShift', 'rightShift']
            infofmt = ['int', 'float', 'int', 'float', 'int', 'float', 'float',
                       '|S25', 'bool', '|S25', 'bool', 'bool']
            info = np.zeros(len(aperg_info),
                            dtype={'names': infoname, 'formats': infofmt})
            # Add information from aperg file
            info['No'] = aperg_info[:, 0]
            info['center'] = aperg_info[:, 1]
            info['dNo'][:-1] = np.diff(info['No'])
            info['dCenter'][:-1] = np.diff(info['center'])
            info['No_next'][:-1] = aperg_info[1:, 0]
            # Separates apertures from inside of blocks and from the gaps.
            maskIN = info['dCenter'][
                     :-1] < 2.8 * np.median(info['dCenter'][:-1])
            info['where'][:-1] = ['inside' if m else 'gap' for m in maskIN]

            # Median values
            # Talvez fazer ele soh pegar valores sem muito desvio, para nao
            # levar em conta os valores no "bump"
            medianIN = np.median([m['dCenter'] for m in info
                                  if m['where'] == 'inside'])
            medianGAP = np.median([m['dCenter'] for m in info
                                   if m['where'] == 'gap'])

            # Gap apertures that were shifted
            infoGAP = info[info['where'] == 'gap']
            leftShift = infoGAP[np.logical_and(infoGAP['No'] % 50 > 0,
                                               infoGAP['No'] % 50 < 25)]
            rightShift = infoGAP[np.logical_or(infoGAP['No_next'] % 50 == 0,
                                               infoGAP['No_next'] % 50 > 25)]

            # Modify dCenter values that have unnusual dCenter values.
            # Otherwise, the error identification may not work.
            # It should be note that this correction may not work every time.
            # *****      vericar se o erro nao ocorre no aperg 2 tb     ******
            # *****      talvez fosse melhor retirar essas linhas       ******
            excep_ap = pkg_resources.resource_filename(
                'gireds', 'data/exceptional_apertures.dat')
            with open(excep_ap, 'r') as f:
                lines = f.readlines()
            fix_No = [int(i.split()[1]) for i in lines if flat in i]
            if (mdf['slits'] == 'red' or mdf['slits'] == 'both') and \
                    slitNo == 1:
                if mdf['instr'].lower() == 'gmos-s':
                    fix_No = np.append(
                        fix_No, infoGAP[abs(infoGAP['No'] - 450) < 10]['No'])
                elif mdf['instr'].lower() == 'gmos-n':
                    fix_No = np.append(
                        fix_No, infoGAP[abs(infoGAP['No'] - 550) < 10]['No'])
            for fix_i in fix_No:
                info['dCenter'][info['No'] == fix_i] += medianIN * .5

            # Calculate the expecteds dCenter values, and identify the errors
            # based in the residuals.
            for i in info[:-1]:
                if i['where'] == 'inside':
                    i['expected'] = i['dNo'] * medianIN
                else:
                    i['expected'] = medianGAP + (i['dNo'] - 1) * medianIN
            info['residual'] = abs(info['dCenter'] - info['expected'])
            tol = medianIN * .5
            info['error'][1:] = info['residual'][1:] > tol
            infoError = info[info['error']]
            # Specify the type of the errors.
            for i in infoError:
                if i['dCenter'] == 0.:
                    i['errType'] = 'twoID'  # 'doubleID'
                elif i['dCenter'] < i['expected']:
                    i['errType'] = 'good'  # 'good_but_masked'
                elif i['dCenter'] > i['expected']:
                    i['errType'] = 'dead'  # 'dead_but_unmasked'

            # Tests
            isLeftShift = len(leftShift)
            isRightShift = len(rightShift)
            isNoneShift = not (isLeftShift or isRightShift)
            isBothShift = isLeftShift and isRightShift
            isGood = (not len(infoError)) and isNoneShift
            # Unnusual values in gaps
            if all([i in infoGAP['No'] for i in infoError['No']]) and \
                    isNoneShift:
                isGood = True
            # Stop iteration if iteration limit was achieved.
            if (not (isGood) and nIter == 7):
                resultError = True

            # Is first aperture identifying the "bump"
            if isLeftShift and (
                    abs(leftShift[0]['No'] - (slitNo - 1) * 750) < 25):
                isFirstShifted = True
            else:
                isFirstShifted = False

            # Select the case
            errType = infoError['errType']
            if isGood:
                mdf['reidentify'] = False
                mdf['modify'] = False
                isGood_out += 1
                if isGood_out == numSlit:
                    isIterating = False
                    iraf.printlog(
                        "\nAPERTURES result: GOOD", iraf.gmos.logfile, 'yes')
            elif resultError:
                iraf.printlog("\nAPERTURES result: ERROR", iraf.gmos.logfile,
                              'yes')
                iraf.printlog("After 7 iteration, APERTURES didn't fix the" +
                              "problem.", iraf.gmos.logfile, 'yes')
                iraf.printlog("Repeat identification with default mdf.",
                              iraf.gmos.logfile, 'yes')
            elif isFirstShifted:
                mdf['modify'] = True
                mdf['reidentify'] = True
                iraf.printlog("\nAPERTURES result: ERROR", iraf.gmos.logfile,
                              'yes')
                iraf.printlog("First fiber identifyied the bump.",
                              iraf.gmos.logfile, 'yes')
                if (mdf['slits'] in ['both', 'red', 'blue']) and \
                        (infoGAP['No'][-1] > infoError['No'][-1]):
                    # Get the last unmasked aperture No. from mdf.
                    mdf['No'] = mdfSlit[mdfSlit['beam'] == 1]['No'][-1]
                    mdf['beam'] = -1
                    iraf.printlog("Assuming that the last aperture should" +
                                  "masked." + "(Slits='both')",
                                  iraf.gmos.logfile, 'yes')
                    iraf.printlog("Mask aperture:" + str(mdf['No']),
                                  iraf.gmos.logfile, 'yes')
                elif errType[0] == 'dead':
                    mdf['No'] = infoError[0]['No']
                    mdf['beam'] = -1
                    iraf.printlog("Bad fiber that is unmasked.",
                                  iraf.gmos.logfile, 'yes')
                    iraf.printlog("Mask aperture:" + str(mdf['No']),
                                  iraf.gmos.logfile, 'yes')
                else:
                    noSolution = True
            else:
                mdf['reidentify'] = True
                mdf['modify'] = True
                iraf.printlog("\nAPERTURES result: ERROR", iraf.gmos.logfile,
                              'yes')
                if isNoneShift:
                    iraf.printlog("No Shift.", iraf.gmos.logfile, 'yes')
                    # Just errors inside the blocks
                    infoError = infoError[infoError['where'] == 'inside']
                    if errType[-1] == 'dead':
                        mdf['No'] = infoError[0]['No_next']
                        mdf['beam'] = -1
                        iraf.printlog("Bad fiber that is unmasked.",
                                      iraf.gmos.logfile, 'yes')
                        iraf.printlog("Mask aperture:" + str(mdf['No']),
                                      iraf.gmos.logfile, 'yes')
                    else:
                        noSolution = True
                elif isBothShift:
                    iraf.printlog("Both shift.", iraf.gmos.logfile, 'yes')
                    if (errType[0] == errType[-1] == 'dead'):
                        mdf['No'] = infoError[0]['No_next']
                        mdf['beam'] = -1
                        iraf.printlog("Two bad fibers that are unmasked.",
                                      iraf.gmos.logfile, 'yes')
                        iraf.printlog("Mask aperture:" + str(mdf['No']),
                                      iraf.gmos.logfile, 'yes')
                    else:
                        noSolution = True
                else:
                    if isLeftShift:
                        iraf.printlog(
                            "Just left shift.", iraf.gmos.logfile, 'yes')
                        if (mdf['slits'] in ['both', 'red', 'blue']) and \
                                (infoGAP['No'][-1] > infoError['No'][-1]):
                            # Get the last unmasked aperture No. from mdf.
                            mdf['No'] = mdfSlit[mdfSlit['beam'] == 1]['No'][-1]
                            mdf['beam'] = -1
                            iraf.printlog("Assuming that the last aperture" +
                                          "should masked. (Slits=both)",
                                          iraf.gmos.logfile, 'yes')
                            iraf.printlog("Mask aperture:" + str(mdf['No']),
                                          iraf.gmos.logfile, 'yes')
                        elif errType[0] == 'good':
                            mdf['No'] = infoError[0]['No'] + 1
                            mdf['beam'] = 1
                            iraf.printlog("Good fiber that is masked.",
                                          iraf.gmos.logfile, 'yes')
                            iraf.printlog("Unmask aperture:" + str(mdf['No']),
                                          iraf.gmos.logfile, 'yes')
                        elif errType[-1] == 'dead':
                            iraf.printlog("Bad fiber that is unmasked.",
                                          iraf.gmos.logfile, 'yes')
                            mdf['beam'] = -1
                            if infoError[-1]['No_next'] in infoGAP['No']:
                                mdf['No'] = infoError[-1]['No_next']
                                iraf.printlog(
                                    "Mask aperture:" + str(mdf['No']),
                                    iraf.gmos.logfile, 'yes')
                            else:
                                mdf['No'] = infoError[-1]['No']
                                iraf.printlog(
                                    "Mask aperture:" + str(mdf['No']),
                                    iraf.gmos.logfile, 'yes')
                        else:
                            noSolution = True
                    elif isRightShift:
                        iraf.printlog("Just right shift.", iraf.gmos.logfile,
                                      'yes')
                        if errType[0] == 'dead':
                            mdf['No'] = infoError[0]['No_next']
                            mdf['beam'] = -1
                            iraf.printlog("Bad fiber that is unmasked.",
                                          iraf.gmos.logfile, 'yes')
                            iraf.printlog("Mask aperture:" + str(mdf['No']),
                                          iraf.gmos.logfile, 'yes')
                        else:
                            noSolution = True
                    else:
                        noSolution = True
            reidentify_out += mdf['reidentify']

            # No solution message
            if noSolution:
                iraf.printlog(
                    "\nAPERTURES result: ERROR", iraf.gmos.logfile, 'yes')
                iraf.printlog(
                    "No solution was found.", iraf.gmos.logfile, 'yes')
                iraf.printlog("Repeat identification with default mdf.",
                              iraf.gmos.logfile, 'yes')

            # Repeat identification with default mdf if resulted in error or
            # no solution was found.
            if noSolution or resultError:
                isIterating = False
                mdf['modify'] = False
                reidentify_out = True

                # Remove old aperg files
                apergPrefList = ['_', '_dq_', '_var_']
                [os.remove(apergFile.replace('_', i)) for i in apergPrefList]

                # Open flat data
                flatFits = fits.open('prg' + flat + '.fits')
                iraf.imdelete('prg' + flat + '.fits')

                # Use default mdf
                flatFits['MDF'].data = mdfDefaultData
                flatFits.writeto('prg' + flat + '.fits')
                flatFits.close()
                break

            # Modify mdf table
            if mdf['modify']:
                # Remove old aperg files
                apergPrefList = ['_', '_dq_', '_var_']
                [os.remove(apergFile.replace('_', i)) for i in apergPrefList]

                # Open flat data
                flatFits = fits.open('prg' + flat + '.fits')
                iraf.imdelete('prg' + flat + '.fits')

                # Modify mdf
                modify_mask = flatFits['MDF'].data['No'] == mdf['No']
                flatFits['MDF'].data['beam'][modify_mask] = mdf['beam']

                # Add new mdf to flat
                flatFits.writeto('prg' + flat + '.fits')
                flatFits.close()

        # Reidentify
        if reidentify_out:
            iraf.imdelete('eprg' + flat + '.fits')
            iraf.gfreduce(
                'prg' + flat, slits='header', rawpath='./', fl_inter='no',
                fl_addmdf='no', key_mdf='MDF', mdffile='default',
                mdfdir='procdir$', weights='no', fl_over='no',
                fl_trim='no', fl_bias='no', trace='yes',
                t_order=4, fl_flux='no', fl_gscrrej='no', fl_extract='yes',
                fl_gsappwave='no', fl_wavtran='no', fl_novl='no',
                fl_skysub='no', reference='', recenter='yes', fl_vardq=vardq)

    # Copy mdf used by flat.
    # File 'gsifu_slits_mdf.fits' is used as base (arbitrarily chosen)
    if 'UR_DIR_PKG' in iraf.envget('gemini'):
        mdfFits = fits.open(
            iraf.envget('UR_DIR_PKG') +
            iraf.envget('gemini').replace('UR_DIR_PKG$/', '') +
            'gmos/data/gsifu_slits_mdf.fits')
    else:
        mdfFits = fits.open(
            iraf.envget('gemini') + 'gmos/data/gsifu_slits_mdf.fits')

    mdfFits[0].header['filename'] = mdffile
    mdfFlatData = fits.getdata('prg' + flat + '.fits', extname='MDF')
    mdfFlatHeader = fits.getheader('prg' + flat + '.fits', extname='MDF')
    mdfFits[1].data = mdfFlatData
    mdfFits[1].header = mdfFlatHeader
    mdfFits.writeto(mdffile)
    mdfFits.close()
