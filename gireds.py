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

        self.dry_run = config.getboolean('main', 'dry_run')
        self.fl_over = config.get('reduction', 'fl_over')
        self.fl_vardq = config.get('reduction', 'fl_vardq')

        self.reduction_step = config.get('main', 'reduction_step')
        # Define directory structure
        self.raw_dir = config.get('main', 'raw_dir')
        self.products_dir = config.get('main', 'products_dir')
        self.run_dir = self.products_dir + time.strftime('%Y-%m-%dT%H:%M:%S')

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
        # Open starinfo file and define structured array
        # Talvez seja melhor definir fora dessa funcao
        starinfo_loc = self.cfg.get('third_party', 'starinfo_loc')
        nstar = sum(1 for line in open(starinfo_loc))
        infoname = ['obj', 'std', 'caldir', 'altname']
        infofmt = ['|S25','|S25', '|S25', '|S25']
        starinfo = np.zeros(nstar,  dtype={'names':infoname, 'formats':infofmt})
        with open(starinfo_loc, 'r') as arq:
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
                (headers[i]['obstype'] == 'OBJECT')&
                (headers[i]['object'] != 'Twilight')&
                (headers[i]['obsclass'] != 'acq'))])

        associated = []

        for i, j in enumerate(images):

            # Take greate care when changing this.
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
                l[k] for k in idx if (
                    (headers[k]['obstype'] == 'BIAS')&
                    (headers[k]['observat'] == hdr['observat'])&
                    (headers[k]['detector'] == hdr['detector'])&
                    (
                        (('overscan' in pf.getheader(l[k], ext=1))&
                        (self.fl_over == 'yes')) or
                        (('overscan' not in pf.getheader(l[k], ext=1))&
                        (self.fl_over == 'no'))
                    ))]

            associated.append(element)

        sci_ims = [i for i in associated if i['obsclass'] == 'science']
        std_ims = [i for i in associated if i['obsclass'] == 'partnerCal']

        #for i in std_ims:
        #    del i['standard_star']
        #    #
        #    # THIS SHOULD BE CHANGED TO ACCURATELY TRANSLATE STAR NAMES
        #    # INTO THE CORRESPONDING NAMES IN CALIBRATION FILES.
        #    #
        #    i['stdstar'] = i['object'].lower()

        # Get star info from starinfo.dat
        for i in std_ims:
            starinfo_idx = [j for j,m in enumerate(starinfo['obj']) \
                              if m==i['object']][0]
            i['stdstar'] = starinfo[starinfo_idx]['std']
            i['caldir'] = starinfo[starinfo_idx]['caldir']

        self.sci = sci_ims
        self.std = std_ims

        file('file_associations_sci.dat', 'w').write(repr(sci_ims))
        file('file_associations_std.dat', 'w').write(repr(std_ims))

    def stdstar(self, dic):
        
        lacosd = self.cfg.get('third_party', 'lacos_loc')

        reduce_stdstar(
            rawdir=self.raw_dir,
            rundir=self.run_dir, caldir=dic['caldir'], starobj=dic['object'],
            stdstar=dic['stdstar'], flat=dic['flat'], arc=dic['arc'],
            twilight=dic['twilight'], starimg=dic['image'],
            bias=dic['bias'], overscan=self.fl_over, vardq=self.fl_vardq,
            lacosdir=lacosd, observatory=dic['observatory'])


if __name__ == "__main__":
    import sys
    pip = pipeline(sys.argv[1])

    if pip.reduction_step == '0':
        pip.associate_files()

    if pip.reduction_step == '1':
        r = file('file_associations_sci.dat', 'r').read()
        pip.sci = eval(r)
        r = file('file_associations_std.dat', 'r').read()
        pip.std = eval(r)
        pip.stdstar(pip.std[0])
