#!/usr/bin/env python

import pyfits as pf
import numpy as np
import os
import ConfigParser
import time
import glob
from standard_star import reduce_stdstar

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

        # Define directory structure
        self.raw_dir = config.get('main', 'raw_dir')
        self.products_dir = config.get('main', 'products_dir')
        self.run_dir = self.products_dir + time.strftime('%Y-%M-%dT%H:%M:%S')
        self.dry_run = config.getboolean('main', 'dry_run')

        if not self.dry_run:
            try:
                os.mkdir(self.products_dir)
            except OSError as err:
                if err.errno == 17:
                    pass
                else:
                    raise err

        if not self.dry_run:
            os.mkdir(self.run_dir)


    def associate_files(self):
        """
        Investigates raw_dir for images to be reduced, and associates
        calibration exposures to science exposures.

        """
       
        os.chdir(self.raw_dir)

        l = glob.glob('*.fits')
        l.sort()

        headers = [pf.getheader(i, ext=0) for i in l]
        mjds = [pf.getheader(i, ext=1)['mjd-obs'] for i in l]
        idx = np.arange(len(l))

        images = np.array([
            l[i] for i in idx if (
                (headers[i]['obstype'] == 'OBJECT')&
                (headers[i]['object'] != 'Twilight')&
                (headers[i]['obsclass'] != 'acq'))])

        associated = []

        for i, j in enumerate(images):

            hdr = pf.getheader(j, ext=0)
            mjd = pf.getheader(j, ext=1)['mjd-obs']
            element = {
                'image':j, 'observatory': hdr['observat'],
                'detector': hdr['detector'], 'grating_wl': hdr['grwlen'],
                'mjd': mjd, 'grating': hdr['grating'],
                'filter1': hdr['filter1'], 'obsclass': hdr['obsclass'],
                'object': hdr['object']}

            element['standard_star'] = [
                l[k] for k in idx if (
                    (headers[k]['obstype'] == 'OBJECT')&
                    (headers[k]['obsclass'] == 'partnerCal')&
                    (headers[k]['object'] != 'Twilight')&
                    (headers[k]['observat'] == hdr['observat'])&
                    (headers[k]['detector'] == hdr['detector'])&
                    (headers[k]['grating'] == hdr['grating'])&
                    (headers[k]['filter1'] == hdr['filter1'])&
                    (headers[k]['grwlen'] == hdr['grwlen'])&
                    (abs(mjds[k] - mjd) <= self.cfg.getfloat('associations',
                        'stdstar_ttol')))]

            element['flat'] = [
                l[k] for k in idx if (
                    (headers[k]['obstype'] == 'FLAT')&
                    (headers[k]['observat'] == hdr['observat'])&
                    (headers[k]['grwlen'] == hdr['grwlen'])&
                    (headers[k]['detector'] == hdr['detector'])&
                    (abs(mjds[k] - mjd) <= self.cfg.getfloat('associations',
                        'flat_ttol')))]

            element['twilight'] = [
                l[k] for k in idx if (
                    (headers[k]['object'] == 'Twilight')&
                    (headers[k]['obstype'] == 'OBJECT')&
                    (headers[k]['observat'] == hdr['observat'])&
                    (headers[k]['detector'] == hdr['detector'])&
                    #(headers[k]['grwlen'] == hdr['grwlen'])&
                    (abs(mjds[k] - mjd) <= self.cfg.getfloat('associations',
                        'twilight_ttol')))]

            element['arc'] = [
                 l[k] for k in idx if (
                    (headers[k]['object'] == 'CuAr')&
                    (headers[k]['obstype'] == 'ARC')&
                    (headers[k]['observat'] == hdr['observat'])&
                    (headers[k]['detector'] == hdr['detector'])&
                    (headers[k]['grwlen'] == hdr['grwlen'])&
                    (abs(mjds[k] - mjd) <= self.cfg.getfloat('associations',
                        'arc_ttol')))]

            element['bias'] = [
                l[k] for k in idx if headers[k]['obstype'] == 'BIAS']

            associated.append(element)

        sci_ims = [i for i in associated if i['obsclass'] == 'science']
        std_ims = [i for i in associated if i['obsclass'] == 'partnerCal']

        for i in std_ims:
            del i['standard_star']
            #
            # THIS SHOULD BE CHANGED TO ACCURATELY TRANSLATE STAR NAMES
            # INTO THE CORRESPONDING NAMES IN CALIBRATION FILES.
            #
            i['stdstar'] = i['object'].lower()

        self.sci = sci_ims
        self.std = std_ims

    def stdstar(self, dic):
        
        cald = self.cfg.get('reduction', 'std_caldir')
        reduce_stdstar(
            rawdir=self.raw_dir,
            rundir=self.run_dir, caldir=cald, starobj=dic['object'],
            stdstar=dic['stdstar'], flat=dic['flat'], arc=dic['arc'],
            twilight=dic['twilight'], starimg=dic['image'],
            bias=dic['bias'])


if __name__ == "__main__":
    import sys
    pip = pipeline(sys.argv[1])
    pip.associate_files()
    print pip.std[0]
    pip.stdstar(pip.std[0])
