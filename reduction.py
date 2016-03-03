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
import numpy as np
import pyfits as pf
import os


def cal_reduction(rawdir, rundir, flat, arc, twilight, bias, bpm, overscan,
                  vardq, instrument, mdffile, slits):
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
            recenter='no', fl_vardq=vardq, mdfdir='gmos$data/')

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

        # Apertures
        apertures(flat, vardq, mdffile, overscan, instrument, slits)

    # copy mdf used by flat
    if os.path.isfile(mdffile):
        iraf.imdelete(mdffile)
    mdfFits = pf.open(
        iraf.show('gemini', Stdout=1)[0] + 'gmos/data/' + mdffile)

    nsciext = pf.getval('prg' + flat + '.fits', ext=0, keyword='nsciext')
    mdfFlatData = pf.getdata('prg' + flat + '.fits', ext=nsciext + 1)
    mdfFits[1].data = mdfFlatData
    mdfFits.writeto(mdffile)
    mdfFits.close()

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
        recenter='no', fl_vardq=vardq, mdfdir='procdir$')

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
            reference='eprg' + flat, fl_vardq='no', mdfdir='procdir$')
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
    isIterating = True
    resultError = False
    noSolution = False
    nIter = 0
    while isIterating == True:
        nIter += 1
        iraf.printlog(80 * "-", 'logfile.log', 'yes')
        iraf.printlog(29 * " " + "APERTURES", 'logfile.log', 'yes')
        iraf.printlog(
            28 * " " + "Iteration " + str(nIter), 'logfile.log', 'yes')

        # Number of slits
        if slits == 'both':
            numSlit = 2
        if (slits == 'red') or (slits == 'blue'):
            numSlit = 1

        reidentify_out = 0
        isGood_out = 0
        for slitNo in range(1, 1 + numSlit):

            # Read mdf data and create dictionary.
            mdf = {'file': mdffile, 'dir': '', 'No': 0, 'beam': 0,
                   'modify': False, 'reidentify': False, 'interactive': 'no',
                   'slits': slits, 'instr': instrument}
            nsciext = pf.getval(
                'prg' + flat + '.fits', ext=0, keyword='nsciext')
            mdfFlatData = pf.getdata('prg' + flat + '.fits', ext=nsciext + 1)
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
            info['where'][:-1] = ['inside' if m == True else 'gap'
                                  for m in maskIN]

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
            if (mdf['slits'] == 'red' or mdf['slits'] == 'both') and\
                    slitNo == 1:
                if mdf['instr'].lower() == 'gmos-s':
                    fix_No = infoGAP[abs(infoGAP['No'] - 450) < 10]['No'][0]
                    info['dCenter'][info['No'] == fix_No] += medianIN * .5
                elif mdf['instr'].lower() == 'gmos-n':
                    fix_No = infoGAP[abs(infoGAP['No'] - 550) < 10]['No'][0]
                    info['dCenter'][info['No'] == fix_No] += medianIN * .5

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
                elif i['dCenter'] - i['expected'] < 0:
                    i['errType'] = 'good'  # 'good_but_masked'
                elif i['dCenter'] - i['expected'] > 0:
                    i['errType'] = 'dead'  # 'dead_but_unmasked'

            # Tests
            isLeftShift = len(leftShift)
            isRightShift = len(rightShift)
            isNoneShift = not(isLeftShift or isRightShift)
            isBothShift = isLeftShift and isRightShift
            isGood = (not len(infoError)) and isNoneShift
            # Unnusual values in gaps
            if all([i in infoGAP['No'] for i in infoError['No']]) and \
                    isNoneShift:
                isGood=True
            # Stop iteration if iteration limit was achieved.
            if (not(isGood) and nIter == 7):
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
                    iraf.printlog("\nAPERTURES result: GOOD", 'logfile.log',
                                  'yes')
            elif resultError:
                iraf.printlog("\nAPERTURES result: ERROR", 'logfile.log',
                              'yes')
                iraf.printlog("After 7 iteration, APERTURES didn't fix the" +
                              "problem.", 'logfile.log', 'yes')
                iraf.printlog("Repeat identification with default mdf.",
                              'logfile.log', 'yes')
            elif isFirstShifted:
                mdf['modify'] = True
                mdf['reidentify'] = True
                iraf.printlog("\nAPERTURES result: ERROR", 'logfile.log',
                              'yes')
                iraf.printlog("First fiber identifyied the bump.",
                              'logfile.log', 'yes')
                if (mdf['slits'] == 'both') and \
                        (infoGAP['No'][-1] > infoError['No'][-1]):
                    # Get the last unmasked aperture No. from mdf.
                    mdf['No'] = mdfSlit[mdfSlit['beam'] == 1]['No'][-1]
                    mdf['beam'] = -1
                    iraf.printlog("Assuming that the last aperture should" +
                                  "masked." + "(Slits='both')",
                                  'logfile.log', 'yes')
                    iraf.printlog("Mask aperture:" + str(mdf['No']),
                                  'logfile.log', 'yes')
                elif errType[0] == 'dead':
                    mdf['No'] = infoError[0]['No']
                    mdf['beam'] = -1
                    iraf.printlog("Bad fiber that is unmasked.",
                                  'logfile.log', 'yes')
                    iraf.printlog("Mask aperture:" + str(mdf['No']),
                                  'logfile.log', 'yes')
                else:
                    noSolution = True
            else:
                mdf['reidentify'] = True
                mdf['modify'] = True
                iraf.printlog("\nAPERTURES result: ERROR", 'logfile.log',
                              'yes')
                if isNoneShift:
                    iraf.printlog("No Shift.", 'logfile.log', 'yes')
                    if errType[-1] == 'dead':
                        mdf['No'] = infoError[0]['No_next']
                        mdf['beam'] = -1
                        iraf.printlog("Bad fiber that is unmasked.",
                                      'logfile.log', 'yes')
                        iraf.printlog("Mask aperture:" + str(mdf['No']),
                                      'logfile.log', 'yes')
                    else:
                        noSolution = True
                elif isBothShift:
                    iraf.printlog("Both shift.", 'logfile.log', 'yes')
                    if (errType[0] == errType[-1] == 'dead'):
                        mdf['No'] = infoError[0]['No_next']
                        mdf['beam'] = -1
                        iraf.printlog("Two bad fibers that are unmasked.",
                                      'logfile.log', 'yes')
                        iraf.printlog("Mask aperture:" + str(mdf['No']),
                                      'logfile.log', 'yes')
                    else:
                        noSolution = True
                else:
                    if isLeftShift:
                        iraf.printlog("Just left shift.", 'logfile.log', 'yes')
                        if (mdf['slits'] == 'both') and \
                                (infoGAP['No'][-1] > infoError['No'][-1]):
                            # Get the last unmasked aperture No. from mdf.
                            mdf['No'] = mdfSlit[mdfSlit['beam'] == 1]['No'][-1]
                            mdf['beam'] = -1
                            iraf.printlog("Assuming that the last aperture" +
                                          "should masked. (Slits=both)",
                                          'logfile.log', 'yes')
                            iraf.printlog("Mask aperture:" + str(mdf['No']),
                                          'logfile.log', 'yes')
                        elif errType[0] == 'good':
                            mdf['No'] = infoError[0]['No'] + 1
                            mdf['beam'] = 1
                            iraf.printlog("Good fiber that is masked.",
                                          'logfile.log', 'yes')
                            iraf.printlog("Unmask aperture:" + str(mdf['No']),
                                          'logfile.log', 'yes')
                        elif errType[-1] == 'dead':
                            iraf.printlog("Bad fiber that is unmasked.",
                                          'logfile.log', 'yes')
                            mdf['beam'] = -1
                            if infoError[-1]['No_next'] in infoGAP['No']:
                                mdf['No'] = infoError[-1]['No_next']
                                iraf.printlog("Mask aperture:" +
                                              str(mdf['No']), 'logfile.log',
                                              'yes')
                            else:
                                mdf['No'] = infoError[-1]['No']
                                iraf.printlog("Mask aperture:" +
                                              str(mdf['No']), 'logfile.log',
                                              'yes')
                        else:
                            noSolution = True
                    elif isRightShift:
                        iraf.printlog("Just right shift.", 'logfile.log',
                                      'yes')
                        if errType[0] == 'dead':
                            mdf['No'] = infoError[0]['No_next']
                            mdf['beam'] = -1
                            iraf.printlog("Bad fiber that is unmasked.",
                                          'logfile.log', 'yes')
                            iraf.printlog("Mask aperture:" + str(mdf['No']),
                                          'logfile.log', 'yes')
                        else:
                            noSolution = True
                    else:
                        noSolution = True
            reidentify_out += mdf['reidentify']

            # No solution message
            if noSolution:
                iraf.printlog("\nAPERTURES result: ERROR", 'logfile.log',
                              'yes')
                iraf.printlog("No solution was found.", 'logfile.log', 'yes')
                iraf.printlog("Repeat identification with default mdf.",
                              'logfile.log', 'yes')
                # iraf.printlog("Running GFEXTRACT in interctive mode.",
                #              'logfile.log', 'yes')
                # mdf['interactive'] = True

            # Repeat identification with default mdf if resulted in error or
            # no solution was found.
            if noSolution or resultError:
                isIterating = False
                mdf['modify'] = False
                reidentify_out = True

                # Remove old aperg files
                apergPrefList = ['_', '_dq_', '_var_']
                [os.remove(apergFile.replace('_', i)) for i in apergPrefList]

                # Get original mdf data
                iraf.cd('gmos$data/')
                mdfData = pf.getdata(mdffile, ext=1)
                iraf.cd('procdir')

                # Modify mdf table in flat
                flatFits = pf.open('prg' + flat + '.fits')
                iraf.imdelete('prg' + flat + '.fits')
                flatFits[nsciext + 1].data = mdfData
                flatFits.writeto('prg' + flat + '.fits')
                flatFits.close()
                break

            # Modify mdf table
            if mdf['modify']:
                # Remove old aperg files
                apergPrefList = ['_', '_dq_', '_var_']
                [os.remove(apergFile.replace('_', i)) for i in apergPrefList]

                # Open flat data
                flatFits = pf.open('prg' + flat + '.fits')
                iraf.imdelete('prg' + flat + '.fits')

                # Modify mdf
                mdfext = nsciext + 1
                modify_mask = flatFits[mdfext].data['No'] == mdf['No']
                flatFits[mdfext].data['beam'][modify_mask] = mdf['beam']

                # Add new mdf to flat
                flatFits.writeto('prg' + flat + '.fits')
                flatFits.close()

        # Reidentify
        if reidentify_out:
            iraf.imdelete('eprg' + flat + '.fits')
            iraf.gfreduce(
                'prg' + flat, slits='header', rawpath='./',
                fl_inter=mdf['interactive'],
                fl_addmdf='no', key_mdf='MDF', mdffile='default',
                mdfdir='procdir$', weights='no',
                fl_over=overscan, fl_trim='no', fl_bias='no', trace='yes',
                t_order=4, fl_flux='no', fl_gscrrej='no', fl_extract='yes',
                fl_gsappwave='no', fl_wavtran='no', fl_novl='no',
                fl_skysub='no', reference='', recenter='yes', fl_vardq=vardq)
