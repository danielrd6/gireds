import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import spectools as st
import pyfits as pf
import glob
import sys
import os
import subprocess


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

    axim.imshow(np.average(cube, 0), interpolation='none')

    wl = st.get_wl(
        cube_file, hdrext=1, dataext=1, dimension=0, pix0key='crpix3',
        wl0key='crval3', dwlkey='cd3_3')

    axspec.plot(wl, np.average(cube, (1, 2)))
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
