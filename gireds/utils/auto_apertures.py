#!/usr/bin/env python
# stdlib
import argparse

# third party
from astropy.convolution import convolve, Gaussian1DKernel
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from numpy import ma
from scipy.interpolate import interp1d

# local


def vertical_profile(fname):

    hdu = fits.open(fname)
    data = ma.masked_invalid(hdu['sci', 4].data)

    p = data[:, 200:300].mean(axis=1)

    return p


def smooth(x, y, over_sample):

    f = interp1d(x, y)
    new_x = np.linspace(x[0], x[-1], x.size * over_sample)
    new_y = f(new_x)

    kernel = Gaussian1DKernel(stddev=over_sample / 2.35)
    c = convolve(new_y, kernel)

    return new_x, c


def find_peaks(x, y, threshold, minflux):

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


def plot_results(x, p, xp, yp):

    fig = plt.figure(1)
    fig.clf()
    ax = fig.add_subplot(111)

    ax.plot(x, p)
    ax.scatter(xp, yp, marker='.', s=50, color='red')

    plt.show()


def main():

    parser = argparse.ArgumentParser(
        description='Identifies the aperture centers in a GMOS flat field.')
    parser.add_argument(
        '-d', '--derivative-threshold', default=20, type=float,
        help='Minimum value of the pixel coordinate derivative that is to be'
        ' identified as a local maximum.')
    parser.add_argument(
        'flatfield', action='store', help='GPREPARED GMOS Flat field image.')
    parser.add_argument(
        '-f', '--flux-threshold', default=1.5e+4, type=float,
        help='Flux in ADU below which nothing is considered a valid aperture.')
    parser.add_argument(
        '-w', '--minsep', default=1, type=float,
        help='Minimum separation between adjacent apertures.')
    parser.add_argument(
        '-s', '--oversample', default=30, type=int,
        help='Oversampling factor for pixel coordinates.')
    parser.add_argument(
        '-p', '--plot', action='store_true', help='Plots the results.')
    args = parser.parse_args()

    p = vertical_profile(args.flatfield)
    x = np.arange(p.size)

    nx, s = smooth(x, p, over_sample=args.oversample)
    xp, yp = find_peaks(
        nx, s, threshold=args.derivative_threshold,
        minflux=args.flux_threshold)
    avx, avy = average_neighbours(xp, yp, threshold=args.minsep)

    print('{:d} apertures found.'.format(avx.size))

    if args.plot:
        plot_results(x, p, avx, avy)
