import glob
import sys
import os

from pyraf import iraf

from gireds.utils import auto_apertures


def main():
    iraf.set(stdimage='imtgmos')

    iraf.gemini()
    iraf.gmos()

    # set directories
    iraf.set(rawdir='/dados/gmos/raw')  # raw files
    iraf.set(procdir='/dados/gmos/reduction/products/ngc7213/')  # processed files

    iraf.gmos.logfile = 'logfile.log'
    iraf.gfextract.verbose = 'no'

    iraf.cd('procdir')

    for task in ['gemini', 'gmos', 'gfextract']:
        iraf.unlearn(task)

    flat = 'S20110927S0062'

    for name in glob.glob('database/apeprg' + flat + '*'):
        if os.path.isfile(name):
            print('Removing file {:s}'.format(name))
            os.remove(name)

    if os.path.isfile('eprg' + flat + '.fits'):
        os.remove('eprg' + flat + '.fits')

    grow_gap = 1
    vardq = 'yes'

    ap = auto_apertures.AutoApertures('prg' + flat + '.fits')
    ap.find_dead_beams()
    ap.fix_mdf()

    iraf.delete('eprg' + flat + '.fits')
    extract_args = {'inimage': 'prg' + flat, 'exslits': '*', 'trace': 'yes', 'recenter': 'yes', 'order': 9,
                    't_nsum': 50, 'function': 'chebyshev', 'fl_novl': 'no', 'fl_fulldq': vardq,
                    'fl_gnsskysub': 'no', 'fl_fixnc': 'no', 'fl_fixgaps': 'yes', 'fl_vardq': 'yes',
                    'grow': grow_gap, 'fl_inter': 'no', 'verbose': 'no'}

    iraf.gfextract(**extract_args)
    sys.exit()

    time_out = 0
    while (ap.check_iraf('database/apeprg' + flat) != 0) and (time_out < 5):
        ap.fix_mdf()
        print('Aperture iteration #{:d}.'.format(time_out))
        iraf.delete('eprg' + flat + '.fits')
        iraf.delete('database/apeprg' + flat + '*')

        extract_args['fl_inter'] = 'yes'
        iraf.gfextract(**extract_args)

        time_out += 1


if __name__ == '__main__':
    main()
