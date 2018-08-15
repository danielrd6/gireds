# STDLIB
import argparse
import json
import subprocess
import warnings
import glob
import time
import os
import ConfigParser
from os.path import isfile

# THIRD PARTY
import pkg_resources
from astropy.io import fits
import numpy as np
from pyraf import iraf

# LOCAL
from . import standard_star
from .galaxy import reduce_science
from .merges import merge_cubes
# from distutils.sysconfig import get_python_lib
# from standard_star import reduce_stdstar


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

    tgt_mjd = fits.getheader(target, ext=1)['mjd-obs']
    mjds = np.array([fits.getheader(i, ext=1)['mjd-obs'] for i in images])

    return images[abs(mjds - tgt_mjd).argsort()[0]]


def skipwarn(imageName):

    warnText = 'Skipping alread present image {:s}.'.format(imageName)
    warnings.warn(warnText)
    iraf.printlog('GIREDS: ' + warnText, 'logfile.log', 'yes')

    return


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

        config_defaults = {
            'object_filter': False}
        config = ConfigParser.SafeConfigParser(config_defaults)
        cfgnames = config.read(config_file)
        if cfgnames == []:
            raise GiredsError(
                'Config file {:s} not found.'.format(config_file))

        self.cfg = config

        self.gireds_dir = pkg_resources.resource_filename('gireds', '')
        # self.version = get_git_hash(self.gireds_dir)
        self.version = '0.1.0'

        self.dry_run = config.getboolean('main', 'dry_run')
        self.fl_over = config.get('reduction', 'fl_over')
        self.fl_vardq = config.get('reduction', 'fl_vardq')

        self.reduction_step = config.getint('main', 'reduction_step')
        self.single_step = config.getboolean('main', 'single_step')

        # self.starinfo_file = config.get('associations', 'starinfo_file')
        self.starinfo_file = pkg_resources.resource_filename(
            'gireds', 'data/starinfo.dat')

        self.lacos_file = config.get('third_party', 'lacos_file')
        self.apply_lacos = config.getboolean('reduction', 'apply_lacos')
        # Define directory structure
        self.raw_dir = config.get('main', 'raw_dir')
        self.products_dir = config.get('main', 'products_dir')

        self.all_stars = config.getboolean('associations', 'all_stars')
        self.stored_sens = config.getboolean('associations', 'stored_sensfunc')
        self.object_filter = config.get('associations', 'object_filter')

        # if (self.single_step and (self.reduction_step != 0)):
        #     self.run_dir = config.get('main', 'run_dir')
        # else:
        #     self.run_dir = self.products_dir\
        #         + time.strftime('%Y-%m-%dT%H:%M:%S')

        if config.get('main', 'run_dir') == 'new':
            self.run_dir = self.products_dir\
                + time.strftime('%Y-%m-%dT%H:%M:%S')
        else:
            self.run_dir = config.get('main', 'run_dir')

    def load_storedsens(self):
        """
        Loads the relevant parameters from the previously stored
        sensibility function for later association.
        """

        l = glob.glob(self.gireds_dir + '/data/*.fits')
        l.sort()
        idx = np.arange(len(l))

        headers = [fits.open(i)[0].header for i in l]

        field_names = ['filename', 'observatory', 'instrument', 'detector',
                       'grating', 'filter1', 'maskname']
        types = ['S120'] + ['S60' for i in range(6)]
        hdrkeys = ['observat', 'instrume', 'detector', 'grating', 'filter1',
                   'maskname']

        hdrpars_type = [
            (field_names[i], types[i]) for i in range(len(field_names))]

        stored_sensfunc = np.array([
            ((l[i],) + tuple([headers[i][j] for j in hdrkeys])) for i in idx],
            dtype=hdrpars_type)

        self.stored_sensfunc = stored_sensfunc

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
        starinfo = np.zeros(nstar, dtype={
            'names': infoname, 'formats': infofmt})
        with open(starinfo_file, 'r') as arq:
            for i in range(nstar):
                linelist = arq.readline().split()
                for j in range(len(infoname)):
                    starinfo[i][j] = linelist[j]

        if self.stored_sens:
            self.load_storedsens()

        os.chdir(self.raw_dir)

        l = glob.glob('*.fits')
        l.sort()

        headers = []
        headers_ext1 = []
        for i in l:
            try:
                headers.append(fits.getheader(i, ext=0))
                headers_ext1.append(fits.getheader(i, ext=1))
            except IOError:
                print('IOError reading file {:s}.'.format(i))
                raise SystemExit(0)

        oversc = np.array(
            [('overscan') in i for i in headers_ext1], dtype='bool')

        mjds = np.array([i['mjd-obs'] for i in headers_ext1], dtype='float32')
        idx = np.arange(len(l))

        images = np.array([
            l[i] for i in idx if (
                (headers[i]['obstype'] == 'OBJECT') &
                (headers[i]['object'] != 'Twilight') &
                (headers[i]['obsclass'] != 'acq'))])

        field_names = [
            'filename', 'observatory', 'instrument', 'detector',
            'grating', 'filter1', 'obsclass', 'object', 'obstype',
            'grating_wl', 'overscan', 'mjd', 'ccdsum']
        types = [
            'S120', 'S60', 'S60', 'S60', 'S60', 'S60', 'S60', 'S60', 'S60',
            'float32', 'bool', 'float32', 'S60']
        hdrkeys = [
            'observat', 'instrume', 'detector', 'grating', 'filter1',
            'obsclass', 'object', 'obstype', 'grwlen']

        hdrpars_type = [
            (field_names[i], types[i]) for i in range(len(field_names))]

        hdrpars = np.array([
            ((l[i],) + tuple([headers[i][j] for j in hdrkeys]) +
             (oversc[i],) + (mjds[i],) + (headers_ext1[i]['ccdsum'],))
            for i in idx], dtype=hdrpars_type)

        associated = []

        for i, j in enumerate(images):

            # Take great care when changing this.
            hdr = fits.getheader(j, ext=0)
            hdr_ext1 = fits.getheader(j, ext=1)
            mjd = hdr_ext1['mjd-obs']

            element = {
                'image': j, 'observatory': hdr['observat'],
                'instrument': hdr['instrume'],
                'detector': hdr['detector'], 'grating_wl': hdr['grwlen'],
                'mjd': mjd, 'grating': hdr['grating'],
                'filter1': hdr['filter1'], 'obsclass': hdr['obsclass'],
                'object': hdr['object']}

            if self.stored_sens:
                ssf = self.stored_sensfunc
                element['standard_star'] = ssf['filename'][
                    (ssf['observatory'] == hdr['observat']) &
                    (ssf['detector'] == hdr['detector']) &
                    (ssf['grating'] == hdr['grating']) &
                    (ssf['instrument'] == hdr['instrume']) &
                    (ssf['filter1'] == hdr['filter1']) &
                    (ssf['maskname'] == hdr['maskname'])]
            else:
                element['standard_star'] = hdrpars['filename'][
                    (hdrpars['obstype'] == 'OBJECT') &
                    (np.array([k in ['partnerCal', 'progCal']
                               for k in hdrpars['obsclass']], dtype='bool')) &
                    (hdrpars['object'] != 'Twilight') &
                    (hdrpars['observatory'] == hdr['observat']) &
                    (hdrpars['detector'] == hdr['detector']) &
                    (hdrpars['grating'] == hdr['grating']) &
                    (hdrpars['filter1'] == hdr['filter1']) &
                    (abs(hdrpars['grating_wl'] - hdr['grwlen']) <=
                        self.cfg.getfloat('associations', 'stdstar_wltol')) &
                    (abs(mjds - mjd) <=
                        self.cfg.getfloat('associations', 'stdstar_ttol'))]

            element['flat'] = hdrpars['filename'][
                (hdrpars['obstype'] == 'FLAT') &
                (hdrpars['observatory'] == hdr['observat']) &
                (hdrpars['grating'] == hdr['grating']) &
                (hdrpars['grating_wl'] == hdr['grwlen']) &
                (hdrpars['detector'] == hdr['detector']) &
                (abs(mjds - mjd) <= self.cfg.getfloat('associations',
                                                      'flat_ttol'))]

            element['twilight'] = hdrpars['filename'][
                (hdrpars['object'] == 'Twilight') &
                (hdrpars['obstype'] == 'OBJECT') &
                (hdrpars['observatory'] == hdr['observat']) &
                (hdrpars['detector'] == hdr['detector']) &
                (hdrpars['grating'] == hdr['grating']) &
                (abs(hdrpars['grating_wl'] - hdr['grwlen']) <=
                    self.cfg.getfloat('associations', 'twilight_wltol')) &
                (abs(mjds - mjd) <=
                    self.cfg.getfloat('associations', 'twilight_ttol'))]

            c = 'twilight'
            if len(element[c]) > 1:
                element[c] = closest_in_time(element[c], j)
            elif len(element[c]) == 1:
                element[c] = element[c][0]
            elif len(element[c]) == 0:
                element[c] = ''

            # A flat close to the twilight observation for a better
            # response function.
            if element['twilight']:
                twipars = hdrpars[hdrpars['filename'] == element['twilight']]
                element['twilight_flat'] = hdrpars['filename'][
                    (hdrpars['obstype'] == 'FLAT') &
                    (hdrpars['observatory'] == twipars['observatory']) &
                    (hdrpars['detector'] == twipars['detector']) &
                    (hdrpars['grating'] == twipars['grating']) &
                    (hdrpars['grating_wl'] == twipars['grating_wl']) &
                    (abs(mjds - twipars['mjd']) <= self.cfg.getfloat(
                        'associations', 'twilight_ttol'))]
            else:
                element['twilight_flat'] = np.array([], dtype='S60')

            element['arc'] = hdrpars['filename'][
                # (hdrpars['object'] == 'CuAr') &
                (hdrpars['obstype'] == 'ARC') &
                (hdrpars['observatory'] == hdr['observat']) &
                (hdrpars['detector'] == hdr['detector']) &
                (hdrpars['grating'] == hdr['grating']) &
                (hdrpars['grating_wl'] == hdr['grwlen']) &
                (abs(mjds - mjd) <=
                    self.cfg.getfloat('associations', 'arc_ttol'))]

            element['bias'] = hdrpars['filename'][
                (hdrpars['obstype'] == 'BIAS') &
                (hdrpars['observatory'] == hdr['observat']) &
                (hdrpars['detector'] == hdr['detector']) &
                (abs(mjds - mjd) <=
                    self.cfg.getfloat('associations', 'bias_ttol')) &
                (
                    (hdrpars['overscan'] & (self.fl_over == 'yes')) |
                    (~hdrpars['overscan'] & (self.fl_over == 'no'))
                )]

            im = fits.open(element['image'])
            ishape = np.array(im[1].data.shape, dtype='float32')
            im.close()
            del(im)

            validBiases = np.ones(len(element['bias']), dtype='bool')
            k = 0

            for biasImage in element['bias']:

                bias = fits.open(biasImage)
                bshape = np.array(bias[1].data.shape, dtype='float32')
                bias.close()
                del(bias)

                #
                # Elinates biases if they differ in array size from
                # the science image. Small differences are normal due to
                # the overscan subtraction in processed bias frames.
                #
                if np.any(np.abs(bshape / ishape - 1.0) > 0.10):
                    validBiases[k] = False

                k += 1

            element['bias'] = element['bias'][validBiases]
            del(k)

            element['bpm'] = hdrpars['filename'][
                (hdrpars['obstype'] == 'BPM') &
                (hdrpars['observatory'] == hdr['observat']) &
                (hdrpars['detector'] == hdr['detector']) &
                (hdrpars['ccdsum'] == hdr_ext1['ccdsum'])]

            categories = ['flat', 'bias', 'arc', 'standard_star',
                          'bpm', 'twilight_flat']

            for c in categories:
                if len(element[c]) > 1:
                    element[c] = closest_in_time(element[c], j)
                elif len(element[c]) == 0:
                    element[c] = ''
                elif len(element[c]) == 1:
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

        if self.object_filter:
            objs = self.object_filter.split(',')
            sci_ims = [
                i for i in associated if (
                    (i['obsclass'] == 'science') &
                    (i['object'] in objs))]
        else:
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

            if starinfo[stdstar_idx]['caldir'] == 'gireds_data':
                i['caldir'] = pkg_resources.resource_filename(
                    'gireds', 'data/')
            else:
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
                sci_ims, open('file_associations_sci.dat', 'w'),
                sort_keys=True, indent=4)
            json.dump(
                std_ims, open('file_associations_std.dat', 'w'),
                sort_keys=True, indent=4)

    def stdstar(self, dic):

        standard_star.reduce_stdstar(
            rawdir=self.raw_dir, rundir=self.run_dir, caldir=dic['caldir'],
            starobj=dic['object'], stdstar=dic['stdstar'], flat=dic['flat'],
            arc=dic['arc'], twilight=dic['twilight'],
            twilight_flat=dic['twilight_flat'], starimg=dic['image'],
            bias=dic['bias'], overscan=self.fl_over, vardq=self.fl_vardq,
            lacos=self.lacos_file, observatory=dic['observatory'],
            apply_lacos=self.apply_lacos, instrument=dic['instrument'],
            lacos_xorder=self.cfg.getint('reduction', 'lacos_xorder'),
            lacos_yorder=self.cfg.getint('reduction', 'lacos_yorder'),
            lacos_objlim=self.cfg.getfloat('reduction', 'lacos_objlim'),
            lacos_sigclip=self.cfg.getfloat('reduction', 'lacos_sigclip'),
            bpm=dic['bpm'], slits=dic['slits'],
            fl_gscrrej=self.cfg.getboolean('reduction', 'fl_gscrrej'),
            wltrim_frac=self.cfg.getfloat('reduction', 'wltrim_frac'),
            sens_order=self.cfg.getint('reduction', 'sens_order'),
            sens_function=self.cfg.get('reduction', 'sens_function'),
            apsum_radius=self.cfg.getfloat('reduction', 'apsum_radius'))

    def science(self, dic):

        if self.stored_sens:
            sensfunc = dic['standard_star']
        else:
            sensfunc = 'sens' + dic['standard_star']

        reduce_science(
            rawdir=self.raw_dir, rundir=self.run_dir, flat=dic['flat'],
            arc=dic['arc'], twilight=dic['twilight'],
            twilight_flat=dic['twilight_flat'], sciimg=dic['image'],
            starimg=sensfunc, bias=dic['bias'],
            overscan=self.fl_over, vardq=self.fl_vardq, lacos=self.lacos_file,
            observatory=dic['observatory'], apply_lacos=self.apply_lacos,
            lacos_xorder=self.cfg.getint('reduction', 'lacos_xorder'),
            lacos_yorder=self.cfg.getint('reduction', 'lacos_yorder'),
            lacos_objlim=self.cfg.getfloat('reduction', 'lacos_objlim'),
            lacos_sigclip=self.cfg.getfloat('reduction', 'lacos_sigclip'),
            bpm=dic['bpm'], slits=dic['slits'], instrument=dic['instrument'],
            fl_gscrrej=self.cfg.getboolean('reduction', 'fl_gscrrej'),
            wltrim_frac=self.cfg.getfloat('reduction', 'wltrim_frac'),
            grow_gap=self.cfg.getfloat('reduction', 'grow_gap'))

    def merge(self, sciobj, name, cube_prefix):

        os.chdir(self.run_dir)

        # Read some keywords. Some of them can be read in step 0.
        imgcube = [cube_prefix + sci['image'] for sci in sciobj]
        xoff = [fits.getval(img, ext=0, keyword='xoffset') for img in imgcube]
        yoff = [fits.getval(img, ext=0, keyword='yoffset') for img in imgcube]
        crv3 = [fits.getval(img, ext=1, keyword='crval3') for img in imgcube]
        cd3 = [fits.getval(img, ext=1, keyword='cdelt3') for img in imgcube]
        cd1 = [abs(fits.getval(img, ext=1, keyword='cdelt1'))
               for img in imgcube]

        merge_cubes(
            rawdir=self.raw_dir, rundir=self.run_dir, name=name,
            observatory=sciobj[0]['observatory'], imgcube=imgcube, xoff=xoff,
            yoff=yoff, crval3=crv3, cdelt3=cd3, cdelt1=cd1)


def filecheck(dic, cat):

    objs = np.array([i['object'] for i in dic])
    idx = objs.argsort()

    for k in idx:

        img = dic[k]

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
    parser.add_argument('--incremental', help='Skip already reduced cubes',
                        action='store_true')

    args = parser.parse_args()

    cal_categories_std = np.array([
        'bias', 'flat', 'twilight', 'arc', 'twilight_flat', 'bpm'])

    cal_categories_sci = np.array([
        'bias', 'flat', 'twilight', 'arc', 'standard_star',
        'twilight_flat', 'bpm'])

    if args.check:

        pip = pipeline(args.config_file)
        pip.dry_run = True
        pip.associate_files()

        filecheck(pip.std, cal_categories_std)

        filecheck(pip.sci, cal_categories_sci)

        if args.verbose:
            print(json.dumps(pip.std, indent=4))
            print(json.dumps(pip.sci, indent=4))

    else:
        iraf.gemini()
        iraf.unlearn('gemini')

        iraf.gemtools()
        iraf.unlearn('gemtools')
        pip = pipeline(args.config_file)

        if pip.apply_lacos:
            if pip.cfg.getboolean('reduction', 'fl_gscrrej'):
                cube_prefix = 'dcstexlprg'
            else:
                cube_prefix = 'dcstelprg'
        else:
            if pip.cfg.getboolean('reduction', 'fl_gscrrej'):
                cube_prefix = 'dcstexprg'
            else:
                cube_prefix = 'dcsteprg'

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

            r = open('file_associations_sci.dat', 'r').read()
            pip.sci = eval(r)
            r = open('file_associations_std.dat', 'r').read()
            pip.std = eval(r)

            for star in pip.std:

                cube_file = pip.run_dir + cube_prefix + star['image']
                if args.incremental and isfile(cube_file):
                    print('Skipping already reduced cube {:s}{:s}'.format(
                        cube_prefix, star['image']))
                    continue

                cal = np.array([
                    True if star[i] != '' else False
                    for i in cal_categories_std])

                if not cal.all():
                    iraf.printlog(
                        ('ERROR! Image {:s} does not have all the necessary'
                         'calibration files: ' + len(cal[~cal]) * '{:s} ')
                        .format(star['image'], *cal_categories_std[~cal]),
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

            r = open('file_associations_sci.dat', 'r').read()
            pip.sci = eval(r)
            # r = open('file_associations_std.dat', 'r').read()
            # pip.std = eval(r)

            for sci in pip.sci:

                cube_file = pip.run_dir + cube_prefix + sci['image']
                if args.incremental and isfile(cube_file):
                    print('Skipping already reduced cube {:s}{:s}'.format(
                        cube_prefix, sci['image']))
                    continue

                cal = np.array([
                    True if sci[i] != '' else False
                    for i in cal_categories_sci])

                if not cal.all():
                    iraf.printlog(
                        ('ERROR! Image {:s} does not have all the necessary\n'
                         'calibration files: ' + len(cal[~cal]) * '{:s} ')
                        .format(sci['image'], *cal_categories_sci[~cal]),
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

        if (pip.reduction_step == 3) or\
                ((pip.single_step is False) and (pip.reduction_step >= 3)):

            iraf.printlog(ver_stamp, logfile=logfile, verbose='yes')

            os.chdir(pip.run_dir)
            iraf.printlog(
                'Starting reduction step 3 on directory {:s}\n'
                .format(os.getcwd()), logfile=logfile, verbose='yes')

            r = open('file_associations_sci.dat', 'r').read()
            pip.sci = eval(r)
            # r = open('file_associations_std.dat', 'r').read()
            # pip.std = eval(r)

            # List of objects
            listname = [(sci['object'].lower()).replace(' ', '')
                        for sci in pip.sci]
            sciname = list(set(listname))

            for name in sciname:

                sufix = '_HYPERCUBE.fits'
                cube_file = pip.run_dir + '/' + name + sufix

                if os.path.isfile(cube_file):
                    skipwarn(cube_file)
                    continue

                if args.incremental and isfile(cube_file):
                    print('Skipping already reduced cube {:s}{:s}'.format(
                        name, sufix))
                    continue

                sciobj = [sci for m, sci in enumerate(pip.sci) if
                          listname[m] == name]

                # Prefix may change
                cubes = np.array([
                    True if os.path.isfile(cube_prefix + sci['image'])
                    else False for sci in sciobj])

                if not cubes.all():
                    iraf.printlog(
                        ('ERROR! Object {:s} does not have all the necessary\n'
                         'cube files.')
                        .format(name), logfile=logfile, verbose='yes')
                    iraf.printlog(
                        'Skipping {:s}.'.format(name),
                        logfile=logfile, verbose='yes')
                    continue
                else:
                    try:
                        pip.merge(sciobj, name, cube_prefix)
                    except Exception as err:
                        iraf.printlog(
                            err.__repr__(), logfile=logfile, verbose='yes')
                        iraf.printlog(
                            'ERROR! An error ocurred when trying to merge '
                            'the galaxy {:s}. Check logfile for more '
                            'information.'.format(name),
                            logfile=logfile, verbose='yes')
