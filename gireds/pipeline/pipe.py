import pyfits as pf
import numpy as np
import os
import ConfigParser
import time
import glob
from standard_star import reduce_stdstar
from galaxy import reduce_science
from pyraf import iraf
import argparse
import json
import subprocess


def get_git_hash(git_dir, short=True):
    """
    Gets the Git hash for the current version of the script.

    Paramters
    ---------
    git_dir: string
        Path to the repository root dir.
    short: bool
        Returns only the first 7 characters from the hash.

    Returns
    -------
    ver: string
        Git hash for the current version.
    """

    cwd = os.getcwd()
    os.chdir(git_dir)

    args = ['git', 'rev-parse', '--short', 'HEAD']
    if not short:
        args.remove('--short')

    ver = subprocess.check_output(args).strip('\n')

    os.chdir(cwd)

    return ver


def closest_in_time(images, target):
    """
    Takes a list of images, and returns the one taken closest in time
    to the target image.

    """

    tgt_mjd = pf.getheader(target, ext=1)['mjd-obs']
    mjds = np.array([pf.getheader(i, ext=1)['mjd-obs'] for i in images])

    return images[abs(mjds - tgt_mjd).argsort()[1]]


class GiredsError(Exception):

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


class pipeline():

    """
    GIREDS - Gmos Ifu REDuction Suite

    This class is intended to concentrate and streamline all the of the
    reduction steps for the reduction of Integral Fiedl Unit (IFU) data
    from GMOS.
    """

    def __init__(self, config_file):

        config = ConfigParser.SafeConfigParser()
        cfgnames = config.read(config_file)
        if cfgnames == []:
            raise GiredsError(
                'Config file {:s} not found.'.format(config_file))

        self.cfg = config

        self.gireds_dir = config.get('DEFAULT', 'gireds_dir')
        self.version = get_git_hash(self.gireds_dir)

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
        self.gireds_dir = config.get('main', 'gireds_dir')

        self.all_stars = config.getboolean('associations', 'all_stars')

        if (self.single_step and (self.reduction_step != 0)):
            self.run_dir = config.get('main', 'run_dir')
        else:
            self.run_dir = self.products_dir\
                + time.strftime('%Y-%m-%dT%H:%M:%S')

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
                'object': hdr['object']}

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
                    (abs(mjds[k] - mjd) <=
                        self.cfg.getfloat('associations', 'stdstar_ttol')))]

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
                    (headers_ext1[k]['ccdsum'] == hdr_ext1['ccdsum'])
                    # (headers_ext1[k]['detsec'] == hdr_ext1['detsec'])
                    )]

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
                MaskName = header_flat['maskname']
                if MaskName == "IFU-2":
                    slits = 'both'
                elif MaskName == "IFU-B":
                    slits = 'blue'
                elif MaskName == "IFU-R":
                    slits = 'red'
                i['slits'] = slits

        sci_ims = [i for i in associated if i['obsclass'] == 'science']

        if self.all_stars:
            std_ims = [
                i for i in associated if i['obsclass'] in ['partnerCal',
                                                           'progCal']]
        else:
            used_stds = [i['standard_star'] for i in sci_ims]
            std_ims = [i for i in associated if i['image'] in used_stds]

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

        if not self.dry_run:
            os.chdir(self.run_dir)
            json.dump(
                sci_ims, file('file_associations_sci.dat', 'w'),
                sort_keys=True, indent=4)
            json.dump(
                std_ims, file('file_associations_std.dat', 'w'),
                sort_keys=True, indent=4)

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
            bpm=dic['bpm'], slits=dic['slits'],
            giredsdir=self.gireds_dir,
            fl_gscrrej=self.cfg.getboolean('reduction', 'fl_gscrrej'),
            wltrim_frac=self.cfg.getfloat('reduction', 'wltrim_frac'),
            sens_order=self.cfg.getint('reduction', 'sens_order'),
            sens_function=self.cfg.get('reduction', 'sens_function'))

    def science(self, dic):

        reduce_science(
            rawdir=self.raw_dir, rundir=self.run_dir, flat=dic['flat'],
            arc=dic['arc'], twilight=dic['twilight'], sciimg=dic['image'],
            starimg=dic['standard_star'], bias=dic['bias'],
            overscan=self.fl_over, vardq=self.fl_vardq, lacos=self.lacos_file,
            observatory=dic['observatory'], apply_lacos=self.apply_lacos,
            lacos_xorder=self.cfg.getint('reduction', 'lacos_xorder'),
            lacos_yorder=self.cfg.getint('reduction', 'lacos_yorder'),
            bpm=dic['bpm'], slits=dic['slits'],
            instrument=dic['instrument'], giredsdir=self.gireds_dir,
            fl_gscrrej=self.cfg.getboolean('reduction', 'fl_gscrrej'),
            wltrim_frac=self.cfg.getfloat('reduction', 'wltrim_frac'))


def filecheck(dic, cat):

    for img in dic:

        cal = np.array([True if img[i] != '' else False for i in cat])

        if not cal.all():
            print(
                ('{:20s} {:s}: missing ' + len(cal[~cal]) * '{:s} ')
                .format(img['object'], img['image'], *cat[~cal]))
        else:
            print('{:20s} {:s}: OK'.format(img['object'], img['image']))


def main():    
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--check', help='Checks if the calibration '
                        'exposures required are available in the raw '
                        'directory.', action='store_true')
    parser.add_argument('-v', '--verbose', help='Prints the dictionary of '
                        'file associations.', action='store_true')
    parser.add_argument('config_file', help='Configuration file for GIREDS')
    args = parser.parse_args()
    
    if args.check:
    
        pip = pipeline(args.config_file)
        pip.dry_run = True
        pip.associate_files()
    
        cal_categories = np.array(['bias', 'flat', 'twilight', 'arc'])
    
        filecheck(pip.std, cal_categories)
    
        cal_categories = np.array([
            'bias', 'flat', 'twilight', 'arc', 'standard_star'])
    
        filecheck(pip.sci, cal_categories)
    
        if args.verbose:
            print(json.dumps(pip.std, indent=4))
            print(json.dumps(pip.sci, indent=4))
    
    else:
        iraf.gemini()
        iraf.unlearn('gemini')
    
        iraf.gemtools()
        iraf.unlearn('gemtools')
        pip = pipeline(args.config_file)
    
        ver_stamp = (50 * '#' + '\n' + 'GIREDS version hash: ' + pip.version +
                     '\n' + 50 * '#' + '\n')
    
        logfile = pip.run_dir + '/logfile.log'
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
    
            iraf.printlog(ver_stamp, logfile=logfile, verbose='yes')
    
            iraf.printlog('Starting reduction step 1\n'
                          'on directory {:s}\n'.format(os.getcwd()),
                          logfile=logfile, verbose='yes')
    
            r = file('file_associations_sci.dat', 'r').read()
            pip.sci = eval(r)
            r = file('file_associations_std.dat', 'r').read()
            pip.std = eval(r)
    
            cal_categories = np.array(['bias', 'flat', 'twilight', 'arc'])
    
            for star in pip.std:
    
                cal = np.array([
                    True if star[i] != '' else False for i in cal_categories])
    
                if not cal.all():
                    iraf.printlog(
                        ('ERROR! Image {:s} does not have all the necessary'
                         'calibration files: ' + len(cal[~cal]) * '{:s} ')
                        .format(star['image'], *cal_categories[~cal]),
                        logfile=logfile, verbose='yes')
                    iraf.printlog(
                        'Skipping image {:s}.'.format(star['image']),
                        logfile=logfile, verbose='yes')
                    continue
                else:
                    try:
                        pip.stdstar(star)
                    except Exception as err:
                        iraf.printlog(
                            err.__repr__(), logfile=logfile, verbose='yes')
                        iraf.printlog(
                            'ERROR! An error ocurred when trying to reduce '
                            'the standard star {:s}. Check logfile for more '
                            'information.'.format(star),
                            logfile=logfile, verbose='yes')
    
        if (pip.reduction_step == 2) or\
                ((pip.single_step is False) and (pip.reduction_step >= 2)):
    
            iraf.printlog(ver_stamp, logfile=logfile, verbose='yes')
    
            os.chdir(pip.run_dir)
            iraf.printlog(
                'Starting reduction step 2 on directory {:s}\n'
                .format(os.getcwd()), logfile=logfile, verbose='yes')
    
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
                    iraf.printlog(
                        ('ERROR! Image {:s} does not have all the necessary\n'
                         'calibration files: ' + len(cal[~cal]) * '{:s} ')
                        .format(sci['image'], *cal_categories[~cal]),
                        logfile=logfile, verbose='yes')
                    iraf.printlog(
                        'Skipping image {:s}.'.format(sci['image']),
                        logfile=logfile, verbose='yes')
                    continue
                else:
                    try:
                        pip.science(sci)
                    except Exception as err:
                        iraf.printlog(
                            err.__repr__(), logfile=logfile, verbose='yes')
                        iraf.printlog(
                            'ERROR! An error ocurred when trying to reduce '
                            'the galaxy {:s}. Check logfile for more '
                            'information.'.format(sci),
                            logfile=logfile, verbose='yes')
