import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import spectools as st
import pyfits as pf
import glob
import sys
import os
import subprocess


def plot_apertures(image):
    """
    Plots the aperture numbers of an already identified image.

    Parameters
    ----------
    image: string
        Name of the FLAT image without any prefixes.
        For instance N20130504S0244.fits

    Returns
    -------
    Nothing.
    """

    hdu = pf.open('prg' + image)
    apfile = './database/apeprg' + image.strip('.fits') + '_1'

    b = np.array(
        [i.split()[3:] for i in open(apfile).readlines() if 'begin' in i])

    apid = b[:, 0]
    x = np.array([float(i) for i in b[:, 2]])

    sci_exts = np.array([i for i in range(len(hdu)) if hdu[i].name == 'SCI'])
    data = hdu[sci_exts[len(sci_exts)/2]].data

    profile = np.average(data, 1)

    fig = plt.figure(1)
    ax = fig.add_subplot(111)

    pmax = profile.max()

    ax.plot(np.arange(len(profile))+1, profile/pmax)
    ax.set_ylim(0, 1.1)

    for i, j in enumerate(apid):
        ax.annotate(j, xy=(x[i], 1), ha='center')
        ax.axvline(x[i], alpha=.3)

    plt.show()


def plot_summary(cube_file, savefigs=True, img_format='pdf'):
    """
    Saves a figure for each cube, with average spectra and image.
    """

    hdu = pf.open(cube_file)

    cube = hdu[1].data
    hdr = hdu[0].header

    fig = plt.figure(1, figsize=(12, 6))
    fig.suptitle(
        'Object: {:s}; File: {:s}'.format(hdr['object'], cube_file))

    gs = mpl.gridspec.GridSpec(1, 2, width_ratios=(1, 5))

    axim = plt.subplot(gs[0])
    axspec = plt.subplot(gs[1])

    axim.imshow(np.median(cube, 0), interpolation='none')

    wl = st.get_wl(
        cube_file, hdrext=1, dataext=1, dimension=0, pix0key='crpix3',
        wl0key='crval3', dwlkey='cd3_3')

    axspec.plot(wl, np.median(cube, (1, 2)))
    axspec.set_xlabel(r'Wavelength ($\AA$)')
    axspec.set_ylim(ymin=0)

    if savefigs:
        plt.savefig(
            cube_file.strip('.fits') + '.' + img_format, format=img_format,
            bbox_inches='tight')


def plot_all(products_dir, img_format='pdf'):

    l = glob.glob(products_dir + 'd*fits')
    l.sort()

    for i in l:
        plot_summary(i, savefigs=True)


def build_latex(products_dir, img_format='pdf'):

    l = glob.glob(products_dir + 'd*' + img_format)
    l.sort()

    os.chdir(products_dir)

    latex = open('summary.tex', 'w')

    latex.write(
        '\\documentclass[a4paper]{article}\n\n'
        '\\usepackage{graphicx}\n'
        '\\usepackage{fullpage}\n\n'
        '\\begin{document}\n\n')

    for i in l:
        latex.write(
            '\\includegraphics[width=\\columnwidth]{{{:s}}}\n\n'
            .format(i))

    latex.write(r'\end{document}')
    latex.close()

    subprocess.call(['pdflatex', 'summary.tex'])


if __name__ == '__main__':

    plot_all(sys.argv[1], img_format='pdf')
    build_latex(sys.argv[1], img_format='pdf')
