import urllib
import json
import subprocess as sp
import pyfits as pf
import argparse
import time
import datetime as dt


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

    ld = date - dt.timedelta(days=spandays)
    hd = date + dt.timedelta(days=spandays)

    return ('{:04d}{:02d}{:02d}'.format(ld.year, ld.month, ld.day),
            '{:04d}{:02d}{:02d}'.format(hd.year, hd.month, hd.day))


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
    sp.call([address, '-O', 'gemini_data.tar', '-v'])
    time.sleep(1)
    sp.call(['tar', 'xvf', 'gemini_data.tar'])
    time.sleep(1)
    sp.call(['md5sum', '-c', 'md5sums.txt'])
    time.sleep(1)
    sp.call(['bunzip2', '-v', '*bz2'])


def get_flat(target_image):
    """
    Produces a gemini compliant query string based on the pars on
    target_image.

    Parameters
    ----------
    target_image : dictionary or structured array
        The parameters for the target image.
    """

    query = 'query'

    return query


def get_twilight(target_image):

    pass


def get_twilight_flat(target_image):

    pass


def get_bias(target_image, ttol=30):

    t = target_image

    binning = {'1 1': '1x1'}

    dates = date_span(t['filename'][1:9], ttol)

    pars = ['PROCESSED_BIAS',
            t['instrument'],
            binning[t['ccdsum']],
            '{:s}/{:s}'.format(*dates)]

    query = (len(pars)*'/{:s}').format(*pars)
    return query


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('image', help="Target image")
    parser.add_argument('-f', '--flat', help='Get flat', action='store_true')
    parser.add_argument('-b', '--bias', help='Get bias', action='store_true')
    parser.add_argument('-d', '--download', action='store_true',
                        help='Only downloads and unpacks a given query')
    parser.add_argument('-l', '--listonly', action='store_true',
                        help='Only lists files to download')

    args = parser.parse_args()

    if not args.download:
        im = pf.open(args.image)

        field_names = [
            'filename', 'observatory', 'instrument', 'detector',
            'grating', 'filter1', 'obsclass', 'object', 'obstype',
            'grating_wl', 'mjd', 'ccdsum']
        # types = [
        #     'S60', 'S60', 'S60', 'S60', 'S60', 'S60', 'S60', 'S60', 'S60',
        #     'float32', 'float32', 'S60']
        hdrkeys = [
            'observat', 'instrume', 'detector', 'grating', 'filter1',
            'obsclass', 'object', 'obstype', 'grwlen']

        # hdrpars_type = [
        #     (field_names[i], types[i]) for i in range(len(field_names))]

        # hdrpars = np.array(
        #     [((args.image,) + tuple([im[0].header[j] for j in hdrkeys]) +
        #       (im[1].header['mjd-obs'],) + (im[1].header['ccdsum'],))],
        #     dtype=hdrpars_type)

        hdrpars = {}
        for i, j in enumerate(field_names):
            if i == 0:
                hdrpars[j] = args.image
            elif i == 10:
                hdrpars[j] = im[1].header['mjd-obs']
            elif i == 11:
                hdrpars[j] = im[1].header['ccdsum']
            else:
                hdrpars[j] = im[0].header[hdrkeys[i - 1]]

        # print(hdrpars)

    if args.download:
        download_unpack(args.image)
    else:
        if args.flat:
            q = get_flat(hdrpars)

        if args.bias:
            q = get_bias(hdrpars)

            query_archive(q)
            if not args.listonly:
                print('Beginning download\n')
                download_unpack(q)


if __name__ == '__main__':

    main()
