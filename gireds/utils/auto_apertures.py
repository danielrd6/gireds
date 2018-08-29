#!/usr/bin/env python
# stdlib
import argparse

# third party
from astropy import table
from astropy.convolution import convolve, Gaussian1DKernel
from astropy.io import fits
from numpy import ma
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import numpy as np


def vertical_profile(fname, extension, column=None, width=100):

    with fits.open(fname) as hdu:

        if ',' in extension:
            extname, extnum = extension.split(',')
            extnum = int(extnum)
            data = ma.masked_invalid(hdu[extname, extnum].data)

        elif extension.isdigit():
            extnum = int(extension)
            data = ma.masked_invalid(hdu[extnum].data)

        elif not extension.isdigit():
            extname = extension
            data = ma.masked_invalid(hdu[extname].data)

    if column is None:
        column = int(data.shape[1] / 2.0)

    x0 = int(column - (width / 2))
    x1 = int(column + (width / 2))
    p = ma.median(data[:, x0:x1], axis=1)

    return p


def smooth(x, y, over_sample):

    f = interp1d(x, y)
    new_x = np.linspace(x[0], x[-1], x.size * over_sample)
    new_y = f(new_x)

    kernel = Gaussian1DKernel(stddev=over_sample / 2.35)
    c = convolve(new_y, kernel)

    return new_x, c


def find_peaks(x, y, threshold, minflux=None):

    if minflux is None:
        minflux = np.percentile(y, 98.0) / 2.0

    m = (np.abs(np.diff(y[:-1])) < threshold)\
        & (y[:-2] > minflux)\
        & (np.diff(y, 2) < 0)

    return x[:-2][m], y[:-2][m]


def average_neighbours(x, y, threshold):

    new_x = []
    new_y = []

    dx = np.diff(x)

    i = 0
    while i < dx.size:

        if dx[i] < threshold:

            n = 1
            xm = 0
            ym = 0

            while dx[i] < threshold:
                xm += x[i]
                ym += y[i]
                n += 1
                i += 1

                if i >= dx.size:
                    break

            xm += x[i]
            ym += y[i]

            new_x.append(xm / n)
            new_y.append(ym / n)

        else:
            new_x.append(x[i])
            new_y.append(y[i])

        i += 1

    return np.array(new_x), np.array(new_y)


def plot_results(x, p, xp, yp, sx, sy):

    fig = plt.figure(1)
    fig.clf()
    ax = fig.add_subplot(111)

    ax.plot(x, p)
    ax.plot(sx, sy)
    ax.scatter(xp, yp, marker='.', s=50, color='red')

    plt.show()


def read_apertures(fname):

    with open(fname, 'r') as f:
        a = [i for i in f.readlines() if (('begin' in i) or ('title' in i))]

    t = table.Table([
        table.Column(name='column', dtype=float),
        table.Column(name='num', dtype=int),
        table.Column(name='beam_title', dtype='S10'),
    ])

    for i in range(0, len(a), 2):
        t.add_row([
            a[i].split()[-1].strip(),
            a[i].split()[-3].strip(),
            a[i + 1].split()[-1].strip()])

    return t


def find_dead_beams(x):
    
    # First derivative of aperture position
    d = np.diff(x)
    # Typical value for distance between apertures
    md = np.median(d)
    # Typical value for distance between fiber bundles
    mg = d[d > 3. * md].mean()
    # Gap limit
    gap_limit = mg + (3 * d[d > 3. * md].std())

    # Array of gaps indexes, including first and last.
    gaps = np.concatenate(([0], np.where(d > 3 * md)[0] + 1, [len(x)]))

    beams = np.ones((750))

    gap_cases = {
        'first or last bundle': np.array([False, False]),
        'missing left fiber': np.array([True, False]),
        'missing right fiber': np.array([False, True]),
    }

    for k, g in enumerate(gaps[:-1]):

        start, end = g, gaps[k + 1]
        i = j = 0
        model = x[start] + np.arange(50) * md

        gap_distances = np.array(
            [d[(gaps[m]-1).clip(min=0, max=len(d) - 1)] for m in [k, k + 1]])

        while (model[-1] - x[end - 1]) > (md / 2.):

            gap_conform = (gap_distances > gap_limit)

            import pdb; pdb.set_trace()
            if np.all(gap_conform == gap_cases['first or last bundle']):
                if k == 0:
                    beams[(50 * k) + j] = -1
                    j += 1
                if k == 14:
                    beams[(50 * k) + (49 - j)] = -1
            elif np.all(gap_conform == gap_cases['missing left fiber']):
                beams[(50 * k) + j] = -1
                j += 1
            # NOTE: This next test will not work for missing left
            # fibers in the bundle to the right.
            elif np.all(gap_conform == gap_cases['missing right fiber']):
                beams[(50 * k) + (49 - j)] = -1
            model = x[start] + np.arange(50 - j) * md

        # General fix for dead fibers anywhere in the bundle.
        while (j < len(model)) and ((g + i) < len(x)):
            if np.abs(x[g + i] - model[j]) > (md / 2):
                beams[50 * k + j] = -1
                j += 1
            else:
                j += 1
                i += 1

    return beams


def main():

    parser = argparse.ArgumentParser(
        description='Identifies the aperture centers in a GMOS flat field.')
    parser.add_argument(
        'flatfield', action='store', help='GPREPARED GMOS Flat field image.')
    parser.add_argument(
        '-c', '--column', type=float,
        help='Image column for vertical profile.')
    parser.add_argument(
        '-d', '--derivative-threshold', default=20, type=float,
        help='Minimum value of the pixel coordinate derivative that is to be'
        ' identified as a local maximum.')
    parser.add_argument(
        '-e', '--extension', type=str,
        help='Name of the MEF extension in which to perform the aperture'
        ' search.')
    parser.add_argument(
        '-p', '--plot', action='store_true', help='Plots the results.')
    parser.add_argument(
        '-s', '--oversample', default=30, type=int,
        help='Oversampling factor for pixel coordinates.')
    parser.add_argument(
        '-t', '--flux-threshold', type=float,
        help='Flux in ADU below which nothing is considered a valid aperture.')
    parser.add_argument(
        '-w', '--minsep', default=1, type=float,
        help='Minimum separation between adjacent apertures.')
    args = parser.parse_args()

    p = vertical_profile(args.flatfield, args.extension, column=args.column)
    x = np.arange(p.size)

    nx, s = smooth(x, p, over_sample=args.oversample)
    xp, yp = find_peaks(
        nx, s, threshold=args.derivative_threshold,
        minflux=args.flux_threshold)
    avx, avy = average_neighbours(xp, yp, threshold=args.minsep)
    beams = find_dead_beams(avx)
    print('Dead fibers: ' + str(np.where(beams == -1)[0].tolist()))

    print('{:d} apertures found.'.format(avx.size))

    if args.plot:
        plot_results(x, p, avx, avy, nx, s)
