import urllib
import json
import subprocess as sp
import pyfits as pf
import argparse
import time
import datetime as dt
import numpy as np


def query_archive(query):
    # Construct the URL. We'll use the jsonsummary service
    url = 'https://archive.gemini.edu/jsonsummary/canonical/'

    # List the OBJECT files taken with GMOS-N on 2010-12-31
    url += query

    # Open the URL and fetch the JSON document text into a string
    u = urllib.urlopen(url)
    jsondoc = u.read()
    u.close()

    # Decode the JSON
    files = json.loads(jsondoc)

    # This is a list of dictionaries each containing info about a file
    total_data_size = 0
    print(
        '{:30s}{:12s}{:12s}{:12s}{:16s}{:8s}{:>10s}'.format(
            'Filename', 'Obs. Class', 'Obs. Type', 'Qa state',
            'Object Name', 'CWL (nm)', 'Disperser'))

    for f in files:
        if f['central_wavelength'] is None:
            f['central_wavelength'] = 0
        else:
            f['central_wavelength'] *= 1e+3

        fields = [
            f['name'],
            f['observation_class'], f['observation_type'],
            f['qa_state'],  f['object'], f['central_wavelength'],
            f['disperser']]

        total_data_size += f['data_size']
        print('{:30s}{:12s}{:12s}{:12s}{:16s}{:8.0f}{:>10s}'.
              format(*fields))

    print('Total data size: {:d}'.format(total_data_size))


def date_span(datestring, spandays):

    date = dt.datetime(int(datestring[:4]),
                       int(datestring[4:6]),
                       int(datestring[6:9]), 0, 0)

    if spandays == 0:
        fmt = '{:04d}{:02d}{:02d}'
        vals = (date.year, date.month, date.day)
    else:
        fmt = '{:04d}{:02d}{:02d}-{:04d}{:02d}{:02d}'
        ld = date - dt.timedelta(days=spandays)
        hd = date + dt.timedelta(days=spandays)
        vals = (ld.year, ld.month, ld.day, hd.year, hd.month, hd.day)

    return (fmt.format(*vals))


def wl_span(wl, span):
    """
    Takes a wavelength coordinate and a radius, and returns a string
    with the wavelength span, already formatted for a query

    Parameters
    ----------
    wl : number
        Wavelength in nanometers

    """

    if span == 0:
        fmt = 'cenwlen={:0.3f}'
        vals = wl * 1e-3
        s = fmt.format(vals)
    else:
        fmt = 'cenwlen={:0.3f}-{:0.3f}'
        vals = (np.array([-span, +span]) + wl) * 1e-3
        s = fmt.format(*vals)

    return s


def download_unpack(query):
    """
    Downloads and unpacks a tar file with bzipped images based on a
    URL query to Gemini's archive.

    Parameters
    ----------
    query : string
        The URL query for Gemini's archive.
    """

    address = 'https://archive.gemini.edu/download/{:s}'.format(query)
    sp.call(['wget', address, '-O', 'gemini_data.tar', '-v'])
    time.sleep(1)
    sp.call(['tar', 'xvf', 'gemini_data.tar'])
    time.sleep(1)
    sp.call(['md5sum', '-c', 'md5sums.txt'])
    time.sleep(3)
    sp.call('bunzip2 -v *bz2', shell=True)


def get_flat(target_image, ttol, wltol):
    """
    Produces a gemini compliant query string based on the pars on
    target_image.

    Parameters
    ----------
    target_image : dictionary or structured array
        The parameters for the target image.
    """

    t = target_image

    pars = ['FLAT',
            'IFS',
            t['maskname'],
            t['instrument'],
            t['binning'],
            date_span(t['filename'][1:9], ttol),
            t['grating'],
            wl_span(t['grating_wl'], wltol),
            'NotFail']

    query = (len(pars)*'/{:s}').format(*pars)

    return query


def get_arc(target_image, ttol):
    """
    Produces a gemini compliant query string based on the pars on
    target_image.

    Parameters
    ----------
    target_image : dictionary or structured array
        The parameters for the target image.
    """

    t = target_image

    pars = ['ARC',
            'IFS',
            t['maskname'],
            t['instrument'],
            t['binning'],
            date_span(t['filename'][1:9], ttol),
            t['grating'],
            wl_span(t['grating_wl'], span=1),
            'NotFail']

    query = (len(pars)*'/{:s}').format(*pars)

    return query


def get_twilight(target_image, ttol, wltol):
    """
    Produces a gemini compliant query string based on the pars on
    target_image.

    Parameters
    ----------
    target_image : dictionary or structured array
        The parameters for the target image.
    """

    t = target_image

    pars = ['Twilight',
            'IFS',
            t['maskname'],
            t['instrument'],
            t['binning'],
            date_span(t['filename'][1:9], ttol),
            t['grating'],
            wl_span(t['grating_wl'], span=wltol),
            'NotFail']

    query = (len(pars)*'/{:s}').format(*pars)

    return query


def get_bias(target_image, ttol):

    t = target_image

    pars = ['PROCESSED_BIAS',
            t['instrument'],
            t['binning'],
            date_span(t['filename'][1:9], ttol)]

    query = (len(pars)*'/{:s}').format(*pars)
    return query


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('image', help='Target image or, in the case of '
                        'options q or d, query string.')
    parser.add_argument('-a', '--arc', help='Get arc', action='store_true')
    parser.add_argument('-f', '--flat', help='Get flat', action='store_true')
    parser.add_argument('-b', '--bias', help='Get bias', action='store_true')
    parser.add_argument('-q', '--query-only', action='store_true',
                        help='Only queries the archive with the given string')
    parser.add_argument('-t', '--twilight',
                        help='Get bias', action='store_true')
    parser.add_argument('-d', '--download', action='store_true',
                        help='Only downloads and unpacks a given query')
    parser.add_argument('-l', '--list-only', action='store_true',
                        help='Only lists files to download')
    parser.add_argument('--twilight-ttol', help="Twilight tolerance in days",
                        nargs='?', default='180')
    parser.add_argument('--twilight-wltol',
                        help="Twilight tolerance in nanometers",
                        nargs='?', default='7')
    parser.add_argument('--bias-ttol', help="Bias tolerance in days",
                        nargs='?', default='30')
    parser.add_argument('--flat-ttol', help="Flat tolerance in days",
                        nargs='?', default='2')
    parser.add_argument('--arc-ttol', help="Flat tolerance in days",
                        nargs='?', default='2')
    parser.add_argument('--flat-wltol', help="Flat tolerance in nanometers",
                        nargs='?', default='6')

    args = parser.parse_args()
    if (not args.download) and (not args.query_only):
        im = pf.open(args.image)

        fields = {'filename': '',
                  'observatory': 'observat',
                  'instrument': 'instrume',
                  'detector': 'detector',
                  'grating': 'grating',
                  'filter1': 'filter1',
                  'obsclass': 'obsclass',
                  'object': 'object',
                  'obstype': 'obstype',
                  'grating_wl': 'grwlen',
                  'maskname': 'maskname',
                  'mjd': 'mjd-obs',
                  'binning': 'ccdsum'}

        hdrpars = {}
        for field in fields:
            if field == 'filename':
                hdrpars[field] = args.image
            elif field == 'mjd':
                hdrpars[field] = im[1].header['mjd-obs']
            elif field == 'binning':
                hdrpars[field] = im[1].header['ccdsum']
            else:
                hdrpars[field] = im[0].header[fields[field]]

        hdrpars['grating'] = hdrpars['grating'].split('+')[0]
        hdrpars['binning'] = hdrpars['binning'].replace(' ', 'x')

        # print(hdrpars)

    if args.download:
        download_unpack(args.image)
    elif args.query_only:
        q = args.image
        query_archive(q)
    else:
        if args.flat:
            q = get_flat(hdrpars, ttol=int(args.flat_ttol),
                         wltol=float(args.flat_wltol))
            print(q)

            query_archive(q)
            if not args.list_only:
                print('Beginning download\n')
                download_unpack(q)

        if args.twilight:
            q = get_twilight(hdrpars, ttol=int(args.twilight_ttol),
                             wltol=float(args.twilight_wltol))
            print(q)

            query_archive(q)
            if not args.list_only:
                print('Beginning download\n')
                download_unpack(q)

        if args.bias:
            q = get_bias(hdrpars, ttol=int(args.bias_ttol))
            print(q)

            query_archive(q)
            if not args.list_only:
                print('Beginning download\n')
                download_unpack(q)

        if args.arc:
            q = get_arc(hdrpars, ttol=int(args.arc_ttol))
            print(q)

            query_archive(q)
            if not args.list_only:
                print('Beginning download\n')
                download_unpack(q)

if __name__ == '__main__':

    main()
