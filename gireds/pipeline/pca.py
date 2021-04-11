import copy

import matplotlib.pyplot as plt
import numpy as np
from astropy import table
from astropy.io import fits
from astropy.stats import sigma_clip
from astropy.wcs import WCS
from scipy import signal
from sklearn.decomposition import PCA


class Tomography(object):

    def __init__(self, file_name, extension=('sci', 1)):
        self.file_name = file_name
        self.extension = extension
        with fits.open(file_name) as h:
            self.mdf = table.Table(h[('MDF', 1)].data)
            beam_mask = self.mdf['BEAM'] == -1
            self.source_mask = (self.mdf['BEAM'] == 1)[~beam_mask]
            self.sky_mask = (self.mdf['BEAM'] == 0)[~beam_mask]
            self.original = h[self.extension].data
            self.source = h[self.extension].data[self.source_mask]
            self.sky = h[self.extension].data[self.sky_mask]
        self.tomograms = np.array([])
        self.eigen_spectra = np.array([])
        self.n_components = 0
        self.reconstruct = np.array([])
        self.wcs = WCS()

        self._get_wavelength()

    def _get_wavelength(self):
        self.wcs = WCS(fits.getheader(self.file_name, ext=self.extension), naxis=[1])
        self.wavelength = self.wcs.wcs_pix2world(np.arange(self.source.shape[1]), 0)[0]

    def _show_tomogram(self, ax, component):
        d = self.tomograms[component]
        ax.plot(d)

    def decompose(self, n_components=20):
        p = PCA(n_components=n_components)
        b = self.source.copy().T
        b[np.isnan(b)] = 0.0
        self.eigen_spectra = p.fit_transform(b).T
        self.tomograms = (getattr(p, 'components_') + getattr(p, 'mean_'))
        self.reconstruct = p.inverse_transform(self.eigen_spectra.T).T

    def plot(self, x, y):
        fig = plt.figure(1)
        fig.clf()
        ax = fig.add_subplot(111)
        ax.plot(self.wavelength, self.source[:, y, x])
        ax.plot(self.wavelength, self.reconstruct[:, y, x])
        plt.show()

    def tomography(self, component):
        fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10, 6), constrained_layout=True)
        self._show_tomogram(ax[0], component)

        ax[1].plot(self.wavelength, self.eigen_spectra[component])

        plt.show()

    def clear(self, components, spaxel):
        """
        Shows the result of removing the given components from the
        reconstructed data.

        Parameters
        ----------
        components : iterable
            List of eigen spectra to eliminate.
        spaxel : (x, y)
            Coordinates of the spectrum to plot.

        Returns
        -------
        None
        """
        d = copy.deepcopy(self.source)
        if len(components) > 0:
            for i in components:
                d -= np.tensordot(self.eigen_spectra[i], self.tomograms[i], axes=0)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(self.wavelength, self.source[:, spaxel[1], spaxel[0]])
        ax.plot(self.wavelength, d[:, spaxel[1], spaxel[0]])
        plt.show()

    def power_spectrum(self, component):
        y = self.tomograms[component]
        p = signal.periodogram(y)

        fig, ax = plt.subplots(nrows=2, ncols=1)
        self._show_tomogram(ax[0], component)
        ax[1].plot(p[0] * self.tomograms.shape[1], p[1])
        plt.show()

    def remove_cosmic_rays(self, sigma_threshold=6.0, clipping_iterations=3):
        """
        Removes cosmic rays or other spurious events based on anomalous
        contributions from specific eigen spectra.

        Parameters
        ----------
        sigma_threshold : float
            Threshold in standard deviations for selecting anomalous contributions.
        clipping_iterations : int
            Number of clipping iterations for the sigma clip algorithm.

        Returns
        -------
        None

        Spectra containing cosmic rays that were not removed in previous steps
        of the reduction process are identified based on their contribution to
        the variance.

        For instance, suppose there is a single spectrum with a extraordinarily
        high value in a couple of pixels, and that no other spectrum has this
        characteristic. The result from the PCA analysis will show that there is
        a large contribution from a single eigen-vector to that particular spectrum.
        This is a good candidate for a cosmic ray event, but if we were to remove
        the entire eigen vector from that spectrum, we would risk removing also
        some other effect which might be real, since eigen spectra usually
        represent several aspects of the data at the same time.

        To avoid modifying the data unnecessarily we perform an additional
        sigma clipping, but this time in the eigen spectrum, to identify those
        pixels which are causing the problem. Only those pixels which match
        the same sigma clipping criteria will actually be removed from the data,
        while the rest is left untouched.

        See Also
        --------
        sklearn.decomposition.PCA, astropy.stats.sigma_clip
        """
        for i, tom in enumerate(self.tomograms):
            x = sigma_clip(tom, sigma=sigma_threshold, iters=clipping_iterations).mask
            if np.any(x):
                tom[~x] = 0.0
                es = self.eigen_spectra[i].copy()
                m = sigma_clip(es, sigma=sigma_threshold, iters=clipping_iterations).mask
                es[~m] = 0
                self.source -= tom[:, np.newaxis] * es[np.newaxis, :]
            else:
                continue

    def write(self, f_name, overwrite=False):
        with fits.open(self.file_name) as h:
            d = self.original.copy()
            d[self.source_mask] = self.source
            h[self.extension].data = d
            h.writeto(f_name, overwrite=overwrite)
