#!/usr/bin/env python

import pyfits as pf
import numpy as np
import os
import ConfigParser
import time
import glob
from standard_star import reduce_stdstar
from galaxy import reduce_science


def closest_in_time(images, target):
    """
    Takes a list of images, and returns the one taken closest in time
    to the target image.

    """

    tgt_mjd = pf.getheader(target, ext=1)['mjd-obs']
    mjds = np.array([pf.getheader(i, ext=1)['mjd-obs'] for i in images])

    return images[abs(mjds - tgt_mjd).argsort()[1]]


class pipeline():

    """
    GIREDS - Gmos Ifu REDuction Suite

    This class is intended to concentrate and streamline all the of the
    reduction steps for the reduction of Integral Fiedl Unit (IFU) data
    from GMOS.
    """

    def __init__(self, config_file):

        config = ConfigParser.SafeConfigParser()
        config.read(config_file)
        self.cfg = config

        self.dry_run = config.getboolean('main', 'dry_run')
        self.fl_over = config.get('reduction', 'fl_over')
        self.fl_vardq = config.get('reduction', 'fl_vardq')

        self.reduction_step = config.getint('main', 'reduction_step')
        self.single_step = config.getboolean('main', 'single_step')

        self.starinfo_file = config.get('associations', 'starinfo_file')
        self.lacos_file = config.get('third_party', 'lacos_file')
        self.apply_lacos = config.getboolean('reduction', 'apply_lacos')
        # Define directory structure
        self.raw_dir = config.get('main', 'raw_dir')
        self.products_dir = config.get('main', 'products_dir')

        if (self.single_step and (self.reduction_step != 0)):
            self.run_dir = config.get('main', 'run_dir')
        else:
            self.run_dir = self.products_dir\
                + time.strftime('%Y-%m-%dT%H:%M:%S')

        if not self.dry_run:
            try:
                os.mkdir(self.products_dir)
            except OSError as err:
                if err.errno == 17:
                    pass
                else:
                    raise err
            try:
                os.mkdir(self.run_dir)
            except OSError as err:
                if err.errno == 17:
                    pass
                else:
                    raise err

    # @profile
    def associate_files(self):
        """
        Investigates raw_dir for images to be reduced, and associates
        calibration exposures to science exposures.

        """
        # Open starinfo file and define structured array
        starinfo_file = self.starinfo_file
        nstar = sum(1 for line in open(starinfo_file))
        infoname = ['obj', 'std', 'caldir', 'altname']
        infofmt = ['|S25', '|S25', '|S25', '|S25']
        starinfo = np.zeros(nstar,  dtype={'names': infoname, 'formats':
                                           infofmt})
        with open(starinfo_file, 'r') as arq:
            for i in range(nstar):
                linelist = arq.readline().split()
                for j in range(len(infoname)):
                    starinfo[i][j] = linelist[j]

        os.chdir(self.raw_dir)

        l = glob.glob('*.fits')
        l.sort()

        headers = [pf.getheader(i, ext=0) for i in l]
        headers_ext1 = [pf.getheader(i, ext=1) for i in l]
        mjds = [i['mjd-obs'] for i in headers_ext1]
        idx = np.arange(len(l))

        images = np.array([
            l[i] for i in idx if (
                (headers[i]['obstype'] == 'OBJECT') &
                (headers[i]['object'] != 'Twilight') &
                (headers[i]['obsclass'] != 'acq'))])

        associated = []

        for i, j in enumerate(images):

            # Take great care when changing this.
            hdr = pf.getheader(j, ext=0)
            hdr_ext1 = pf.getheader(j, ext=1)
            mjd = hdr_ext1['mjd-obs']

            element = {
                'image': j, 'observatory': hdr['observat'],
                'instrument': hdr['instrume'],
                'detector': hdr['detector'], 'grating_wl': hdr['grwlen'],
                'mjd': mjd, 'grating': hdr['grating'],
                'filter1': hdr['filter1'], 'obsclass': hdr['obsclass'],
                'object': hdr['object']
            }

            element['standard_star'] = [
                l[k] for k in idx if (
                    (headers[k]['obstype'] == 'OBJECT') &
                    (headers[k]['obsclass'] in ['partnerCal', 'progCal']) &
                    (headers[k]['object'] != 'Twilight') &
                    (headers[k]['observat'] == hdr['observat']) &
                    (headers[k]['detector'] == hdr['detector']) &
                    (headers[k]['grating'] == hdr['grating']) &
                    (headers[k]['filter1'] == hdr['filter1']) &
                    (abs(headers[k]['grwlen'] - hdr['grwlen']) <=
                        self.cfg.getfloat('associations', 'stdstar_wltol')) &
                    (abs(mjds[k] - mjd) <= self.cfg.getfloat('associations',
                                                             'stdstar_ttol')))]

            element['flat'] = [
                l[k] for k in idx if (
                    (headers[k]['obstype'] == 'FLAT') &
                    (headers[k]['observat'] == hdr['observat']) &
                    (headers[k]['grating'] == hdr['grating']) &
                    (headers[k]['grwlen'] == hdr['grwlen']) &
                    (headers[k]['detector'] == hdr['detector']) &
                    (abs(mjds[k] - mjd) <= self.cfg.getfloat('associations',
                                                             'flat_ttol')))]

            element['twilight'] = [
                l[k] for k in idx if (
                    (headers[k]['object'] == 'Twilight') &
                    (headers[k]['obstype'] == 'OBJECT') &
                    (headers[k]['observat'] == hdr['observat']) &
                    (headers[k]['detector'] == hdr['detector']) &
                    (headers[k]['grating'] == hdr['grating']) &
                    (abs(headers[k]['grwlen'] - hdr['grwlen']) <=
                        self.cfg.getfloat('associations', 'twilight_wltol')) &
                    (abs(mjds[k] - mjd) <= self.cfg.getfloat(
                        'associations', 'twilight_ttol')))]

            element['arc'] = [
                l[k] for k in idx if (
                    (headers[k]['object'] == 'CuAr') &
                    (headers[k]['obstype'] == 'ARC') &
                    (headers[k]['observat'] == hdr['observat']) &
                    (headers[k]['detector'] == hdr['detector']) &
                    (headers[k]['grating'] == hdr['grating']) &
                    (headers[k]['grwlen'] == hdr['grwlen']) &
                    (abs(mjds[k] - mjd) <= self.cfg.getfloat(
                        'associations', 'arc_ttol')))]

            element['bias'] = [
                l[k] for k in idx if (
                    (headers[k]['obstype'] == 'BIAS') &
                    (headers[k]['observat'] == hdr['observat']) &
                    (headers[k]['detector'] == hdr['detector']) &
                    (abs(mjds[k] - mjd) <= self.cfg.getfloat('associations',
                                                             'bias_ttol')) &
                    (
                        (('overscan' in headers_ext1[k]) &
                         (self.fl_over == 'yes')) or
                        (('overscan' not in headers_ext1[k]) &
                         (self.fl_over == 'no'))
                    ))]

            element['bpm'] = [
                l[k] for k in idx if (
                    (headers[k]['obstype'] == 'BPM') &
                    (headers[k]['observat'] == hdr['observat']) &
                    (headers[k]['detector'] == hdr['detector']) &
                    (headers_ext1[k]['ccdsum'] == hdr_ext1['ccdsum']) &
                    (headers_ext1[k]['detsec'] == hdr_ext1['detsec']))]

            categories = ['flat', 'bias', 'arc', 'twilight', 'standard_star',
                          'bpm']
            for c in categories:
                if len(element[c]) > 1:
                    element[c] = closest_in_time(element[c], j)
                elif element[c] == []:
                    element[c] = ''
                else:
                    element[c] = (element[c])[0]

            associated.append(element)

        # Define mdf filename
        # Based in gprepare.cl
        # Did not account for observation in Nod-and-Shuffle
        for i in associated:
            header_flat = [
                k for j, k in enumerate(headers) if l[j] == i['flat']
            ]
            if len(header_flat):
                header_flat = header_flat[0]
                mdfPrefix = "g" + header_flat['instrume'][-1].lower() + "ifu_"
                mdfSufix = ".fits"
                if header_flat['dettype'] == "S10892":
                    mdfSufix = "_HAM.fits"
                MaskName = header_flat['maskname']
                if MaskName == "IFU-2":
                    mdffile = mdfPrefix + "slits_mdf" + mdfSufix
                elif MaskName == "IFU-B":
                    mdffile = mdfPrefix + "slitb_mdf" + mdfSufix
                elif MaskName == "IFU-R":
                    mdffile = mdfPrefix + "slitr_mdf" + mdfSufix
                else:
                    mdffile = 'default'
                i['mdffile'] = mdffile

        sci_ims = [i for i in associated if i['obsclass'] == 'science']
        std_ims = [
            i for i in associated if i['obsclass'] in ['partnerCal', 'progCal']
            ]

        # Get star info from starinfo.dat
        possible_names = np.concatenate((starinfo['obj'], starinfo['std'],
                                         starinfo['altname']))
        n_names = len(possible_names)

        for i, j in enumerate(possible_names):
            possible_names[i] = (j.lower()).replace(' ', '')

        for i in std_ims:
            # Removes the 'standard_star' key if the dictionary
            # element in question refers to a standard star.
            del i['standard_star']
            starname = (i['object'].lower()).replace(' ', '')

            try:
                stdstar_idx = (
                    np.arange(n_names)[possible_names == starname] %
                    (n_names / 3))[0]
            except:
                raise Exception(
                    'Standard star named {:s} not found in file {:s}'.
                    format(starname, starinfo_file))

            i['stdstar'] = starinfo[stdstar_idx]['std']
            i['caldir'] = starinfo[stdstar_idx]['caldir']

        self.sci = sci_ims
        self.std = std_ims

        # Writes the file association dictionary to an ASCII file
        # in the run directory.
        if not self.dry_run:
            os.chdir(self.run_dir)
            file('file_associations_sci.dat', 'w').write(repr(sci_ims))
            file('file_associations_std.dat', 'w').write(repr(std_ims))

    def stdstar(self, dic):

        reduce_stdstar(
            rawdir=self.raw_dir, rundir=self.run_dir, caldir=dic['caldir'],
            starobj=dic['object'], stdstar=dic['stdstar'], flat=dic['flat'],
            arc=dic['arc'], twilight=dic['twilight'], starimg=dic['image'],
            bias=dic['bias'], overscan=self.fl_over, vardq=self.fl_vardq,
            lacos=self.lacos_file, observatory=dic['observatory'],
            apply_lacos=self.apply_lacos, instrument=dic['instrument'],
            lacos_xorder=self.cfg.getint('reduction', 'lacos_xorder'),
            lacos_yorder=self.cfg.getint('reduction', 'lacos_yorder'),
            bpm=dic['bpm'], mdffile=dic['mdffile'])

    def science(self, dic):

        reduce_science(
            rawdir=self.raw_dir, rundir=self.run_dir, flat=dic['flat'],
            arc=dic['arc'], twilight=dic['twilight'], sciimg=dic['image'],
            starimg=dic['standard_star'], bias=dic['bias'],
            overscan=self.fl_over, vardq=self.fl_vardq, lacos=self.lacos_file,
            observatory=dic['observatory'], apply_lacos=self.apply_lacos,
            lacos_xorder=self.cfg.getint('reduction', 'lacos_xorder'),
            lacos_yorder=self.cfg.getint('reduction', 'lacos_yorder'),
            bpm=dic['bpm'], mdffile=dic['mdffile'])


if __name__ == "__main__":
    import sys

    pip = pipeline(sys.argv[1])
    print('##################################################\n'
          '# GIREDS (Gmos Ifu REDuction Suite)              #\n'
          '##################################################\n'
          'Starting reduction at: {:s}\n'.format(time.asctime()))

    if (pip.reduction_step == 0) or\
            ((pip.single_step is False) and (pip.reduction_step >= 0)):

        print('Starting reduction step 0\n'
              'on directory {:s}\n'.format(pip.raw_dir))

        pip.associate_files()

    if (pip.reduction_step == 1) or\
            ((pip.single_step is False) and (pip.reduction_step >= 1)):

        os.chdir(pip.run_dir)
        print('Starting reduction step 1\n'
              'on directory {:s}\n'.format(os.getcwd()))

        r = file('file_associations_sci.dat', 'r').read()
        pip.sci = eval(r)
        r = file('file_associations_std.dat', 'r').read()
        pip.std = eval(r)

        cal_categories = np.array(['bias', 'flat', 'twilight', 'arc'])

        for star in pip.std:

            cal = np.array([
                True if star[i] != '' else False for i in cal_categories])

            if not cal.all():
                print(('ERROR! Image {:s} does not have all the necessary\n'
                      'calibration files: ' + len(cal[~cal]) * '{:s} ')
                      .format(star['image'], *cal_categories[~cal]))
                print('Skipping image {:s}.'.format(star['image']))
                continue
            else:
                pip.stdstar(star)

    if (pip.reduction_step == 2) or\
            ((pip.single_step is False) and (pip.reduction_step >= 2)):

        os.chdir(pip.run_dir)
        print('Starting reduction step 2\n'
              'on directory {:s}\n'.format(os.getcwd()))

        r = file('file_associations_sci.dat', 'r').read()
        pip.sci = eval(r)
        r = file('file_associations_std.dat', 'r').read()
        pip.std = eval(r)

        cal_categories = np.array([
            'bias', 'flat', 'twilight', 'arc', 'standard_star'])

        for sci in pip.sci:

            cal = np.array([
                True if sci[i] != '' else False for i in cal_categories])

            if not cal.all():
                print(('ERROR! Image {:s} does not have all the necessary\n'
                      'calibration files: ' + len(cal[~cal]) * '{:s} ')
                      .format(sci['image'], *cal_categories[~cal]))
                print('Skipping image {:s}.'.format(sci['image']))
                continue
            else:
                pip.science(sci)
