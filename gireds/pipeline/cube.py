import copy

import image_registration
import matplotlib.pyplot as plt
import numpy as np
from astropy.stats import sigma_clip
from astropy import table
from astropy import wcs
from astropy.io import fits
from astropy.modeling import fitting
from astropy.coordinates import Angle
from matplotlib.patches import RegularPolygon
from numpy import ma
from scipy import ndimage
from scipy.interpolate import griddata
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation
from scipy.optimize import minimize

from gireds.pipeline.models import DifferentialRefraction


def get_points(mdf, sampling=0.02, cushion=0.0):
    object_fibers = mdf[mdf['BEAM'] == 1]

    limits = []
    for dimension in 'XY':
        limits.append(object_fibers['{:s}INST'.format(dimension)].min() - cushion)
        limits.append(object_fibers['{:s}INST'.format(dimension)].max() + cushion)

    coords = [np.arange(limits[2 * _], limits[(2 * _) + 1], sampling) for _ in range(2)]
    # noinspection PyUnresolvedReferences
    x, y = [_.flatten() for _ in np.meshgrid(*coords)]
    shape = tuple([len(coords[_]) for _ in [1, 0]])

    return np.vstack((x, y)).T, shape


class CubeBuilder:
    def __init__(self, file_name, sampling=0.1):
        self.file_name = file_name
        with fits.open(file_name) as f:
            self.science = ma.masked_invalid(f['SCI'].data)
            self.mdf = table.Table(f['MDF'].data)
            self.variance = ma.masked_invalid(f['VAR'].data) if 'VAR' in f else None
            self.data_quality = ma.masked_invalid(f['DQ'].data) if 'DQ' in f else None
            self.wavelength_keys = {'crval': f['SCI'].header['crval1'], 'cd': f['SCI'].header['CD1_1']}
            self.air_mass = f['primary'].header['airmass']
            self.temperature = f['primary'].header['tambient']
            self.pressure = f['primary'].header['pressure']
            self.parallactic_angle = self._get_parallactic_angle(f['primary'].header)
            self.instrument_angle = np.deg2rad(f['primary'].header['iaa'])
            self.position_angle = np.deg2rad(f['primary'].header['pa'])
            w = wcs.WCS(f['sci'].header, naxis=[1])

        self.n_wavelength = self.science.shape[1]
        self.wavelength = w.wcs_pix2world(np.arange(self.n_wavelength), 0)[0]
        self.sampling = sampling

        self.flux_density()
        self.atmospheric_shift = None
        self.refraction_angle = 0.0

    @staticmethod
    def _get_parallactic_angle(header):
        """
        Calculates parallactic angle based on header information.

        Parameters
        ----------
        header : astropy.io.fits.header.Header
            Header of the input image.

        Returns
        -------
        pa : float
            Parallactic angle in radians.

        Notes
        -----
        This equation is based on the one presented in Astronomical
        Algorithms 2nd ed. (Jean Meeus, 1998).
        """
        latitude = {'gemini-north': '19:49:25.7016', 'gemini-south': '-30:14:26.700'}
        lat = Angle(latitude[header['OBSERVAT'].lower()], unit='degree').radian
        dec = Angle(header['DEC'], unit='degree').radian
        ha = Angle(header['HA'], unit='hourangle').radian
        d = (np.tan(lat) * np.cos(dec)) - (np.sin(dec) * np.cos(ha))
        pa = np.arctan(np.sin(ha) / d)
        return pa

    def angle_representation(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.arrow(0, 0, 0, 1)
        ax.text(0, 1, 'N')
        for i, j in enumerate(['refraction_angle', 'position_angle', 'instrument_angle', 'parallactic_angle']):
            t = getattr(self, j) + (np.pi / 2.0)
            x, y = np.cos(t), np.sin(t)
            ax.arrow(0, 0, x, y, width=0.07, color='C{:d}'.format(i), alpha=0.5)
            ax.text(x / 2.0, y / 2.0, j.replace('_angle', ''), rotation=np.rad2deg(t), ha='center', va='center')
        w = 1.2
        ax.set_title(self.file_name.split('/')[-1])
        ax.set_xlim(-w, w)
        ax.set_ylim(-w, w)
        ax.grid()
        ax.set_aspect('equal')
        plt.show()

    def flux_density(self):
        """
        Converts from flux per fiber to flux per arcsec squared.

        Returns
        -------
        None
        """
        lenslet_radius = 0.09
        lenslet_area = 3.0 * np.sqrt(3.0) * np.square(lenslet_radius) / 2.0
        correction_factor = 1.0 / lenslet_area

        self.science *= correction_factor
        if self.variance is not None:
            self.variance *= correction_factor

    @staticmethod
    def _check_registration(reference, images):
        xo, yo = [], []
        for i in images:
            x_off, y_off, x_err, y_err = image_registration.chi2_shift(reference, i)
            xo.append(x_off)
            yo.append(y_off)
        xo = np.array(xo)
        yo = np.array(yo)
        return xo, yo

    def _debug_plots(self, data, planes):
        fig, ax = plt.subplots(nrows=2, ncols=3, sharex='col', sharey='row')

        x_off = - np.cos(self.refraction_angle) * self.atmospheric_shift(planes) / self.sampling
        y_off = - np.sin(self.refraction_angle) * self.atmospheric_shift(planes) / self.sampling

        ax = ax.flatten()
        s = data[0].shape
        ext = np.array([-s[1], s[1], -s[0], s[0]]) / 2.0 * self.sampling
        for i, j in enumerate(np.linspace(0, len(planes) - 1, 6).astype(int)):
            ax[i].imshow(ndimage.shift(data[j], (-y_off[j], -x_off[j])), origin='lower', extent=ext)
            ax[i].grid()
            ax[i].set_title(int(planes[j]))
        fig.tight_layout()
        plt.show()
        plt.close(fig)

    @staticmethod
    def _offsets_rotation(x, y):
        z = np.zeros_like(x)
        v0 = np.column_stack([x, y, z])
        offsets = np.linalg.norm(v0, axis=1) * np.sign(x)

        if np.median(np.diff(offsets)) > 0.0:
            flip = True
            offsets *= -1.0
        else:
            flip = False

        v1 = np.column_stack([offsets, z, z])

        t = np.arctan2(y.copy(), x.copy())
        t[t < 0] += np.pi
        mt = np.median(t)

        def res(p, v_0, v_1):
            rot = Rotation.from_euler('z', p[0])
            r = np.sum(np.square(v_0 - rot.apply(v_1)))
            return r

        fit = minimize(res, np.array([mt]), args=(v0, v1), bounds=[[0.0, 2.0 * np.pi]], method='slsqp')

        angle = float(fit.x)
        if flip:
            angle += np.pi

        return offsets, angle

    def fit_refraction_function(self, steps=10, plot=False, sample=None, debug=False):
        """
        Fits a refraction function using a 3rd order Legendre
        Polynomial to the x and y pixel offsets caused by
        atmospheric refraction.

        Parameters
        ----------
        steps : int
            Number of segments of the data cube to consider. The
            larger this number, the finer the detail in fitting
            offsets.
        plot : bool
            Plots the function and the corresponding data points.
        sample : tuple, (w_0, w_1)
            Wavelength interval to be considered in the fit.
        debug : bool
            Plots debugging graphs to confirm that the shifts
            provided are actually matching the reference.

        Returns
        -------
        None
        """
        data = copy.deepcopy(self.science)

        d = np.array([ma.median(_, axis=0) for _ in np.array_split(data, steps)])
        planes = np.array([((_[-1] + _[0]) / 2) for _ in np.array_split(self.wavelength, steps)])

        for i, j in enumerate(d):
            d[i] /= j.max()

        md = d[int(len(d) / 2.0)]
        mid_point = planes[int(len(d) / 2.0)]

        if sample is None:
            sample = self.wavelength[[0, -1]]

        sample_mask = (planes >= sample[0]) & (planes <= sample[1])
        d = d[sample_mask]
        planes = planes[sample_mask]

        x_off, y_off = self._check_registration(reference=md, images=d)
        x_off *= self.sampling
        y_off *= self.sampling

        offsets, angle = self._offsets_rotation(x_off, y_off)
        self.refraction_angle = angle

        if debug:
            print(self.file_name)
            print('OFFSETS')
            print(offsets)

        model = DifferentialRefraction(temperature=self.temperature, pressure=self.pressure, air_mass=self.air_mass,
                                       wl_0=mid_point)
        model.wl_0.fixed = True
        fitter = fitting.FittingWithOutlierRemoval(fitting.SLSQPLSQFitter(), sigma_clip, niter=3, sigma=3.0)
        rejected, shift = fitter(model, planes, offsets, acc=1e-12)

        if plot:
            fig, ax = plt.subplots(1, 1, sharex='col')

            ax.scatter(planes, offsets, c=planes)
            ax.set_ylabel('Differential refraction (arcsec)')
            ax.set_xlabel('Wavelength')
            ax.plot(self.wavelength, shift(self.wavelength))
            pars = [getattr(shift, _).value for _ in ['air_mass', 'wl_0']]
            pars.append(np.rad2deg(angle))
            ax.set_title('secz = {:.2f}; wl_0 = {:.0f}; angle = {:.2f}'.format(*pars))
            ax.grid()

            plt.show()

        self.atmospheric_shift = shift

        if debug:
            self._debug_plots(d, planes)

    def _get_sources(self):
        source = ['science']
        if self.variance is not None:
            source.append('variance')
        if self.data_quality is not None:
            source.append('data_quality')
        return source

    @staticmethod
    def roll_and_pad(data, shift):
        shift = tuple([int(np.round(_)) for _ in shift])
        d = np.roll(copy.deepcopy(data), shift=shift)
        for i, s in enumerate(shift):
            sli = [slice(None), slice(None)]
            if s >= 0:
                sli[i] = slice(0, s)
            elif s < 0:
                sli[i] = slice(s, None)
            else:
                raise RuntimeError('This shift is very strange indeed.')
            d[tuple(sli)] = 16
        return d

    def fix_atmospheric_refraction(self):

        assert self.atmospheric_shift is not None, \
            'Atmospheric shift is not defined. Please run the method fit_refraction_function prior to this method.'

        x_shift = - np.cos(self.refraction_angle) * self.atmospheric_shift(self.wavelength) / self.sampling
        y_shift = - np.sin(self.refraction_angle) * self.atmospheric_shift(self.wavelength) / self.sampling

        for s in self._get_sources():
            data = copy.deepcopy(getattr(self, s))
            data[data.mask] = 0.0

            if s == 'data_quality':
                for i, j in enumerate(x_shift):
                    data[i] = self.roll_and_pad(data[i], (-y_shift[i], -x_shift[i]))
            else:
                for i, j in enumerate(x_shift):
                    data[i] = ndimage.shift(data[i], (-y_shift[i], -x_shift[i]), mode='constant', cval=0.0)

            setattr(self, s, ma.masked_invalid(data))

    def build_cube(self):
        sampling = self.sampling
        n_wavelength = self.n_wavelength

        points, shape = get_points(mdf=self.mdf, sampling=sampling)
        cube = np.zeros((n_wavelength,) + shape)

        beam_mask = self.mdf['BEAM'] == 1
        x = self.mdf['XINST'][beam_mask].data
        y = self.mdf['YINST'][beam_mask].data
        grid_coordinates = np.vstack([x, y]).T

        for s in self._get_sources():
            print('Building cube for {:s}.'.format(s))
            data = getattr(self, s)
            k = 0
            step_size = int(n_wavelength / 10)
            for plane in range(data.shape[1]):
                if plane % step_size == 0:
                    print('{:d}%'.format(k))
                    k += 10
                values = np.array(data[self.mdf[beam_mask]['APID'].data - 1, plane])
                method = 'nearest' if (s == 'data_quality') else 'linear'
                grid = griddata(grid_coordinates, values=values, xi=points, method=method)
                cube[plane] = grid.reshape(shape)
            print(' OK!')

            setattr(self, s, ma.masked_invalid(cube))

    def set_wcs(self, header):
        """
        Adds WCS keywords for the data cube in the header.
        Parameters
        ----------
        header : astropy.io.fits.Header
            Header instance on which to write the WCS keywords.

        Returns
        -------

        """
        for i in range(1, 4):
            header['CRPIX{:d}'.format(i)] = 1.0
            header['CRVAL{:d}'.format(i)] = 0.0

        header['CRVAL3'] = self.wavelength_keys['crval']
        header['CD1_1'] = -self.sampling
        header['CDELT1'] = -self.sampling
        header['CD2_2'] = -self.sampling
        header['CDELT2'] = -self.sampling
        header['CD3_3'] = self.wavelength_keys['cd']
        header['CDELT3'] = self.wavelength_keys['cd']

    @staticmethod
    def copy_header(old, new):
        keys = ['object', 'ccdsum', 'gain', 'gainmult', 'rdnoise', 'radecsys', 'equinox', 'mjd-obs', 'wavtran',
                'exptime', 'airmass', 'bunit', 'fluxscal']
        for card in old.cards:
            if card[0].lower() in keys:
                new.append(card)

    def write(self, output, overwrite=False):
        new = fits.HDUList()

        with fits.open(self.file_name) as old:
            new.append(old['PRIMARY'])

            if self.atmospheric_shift is not None:
                new['PRIMARY'].header.append(('ATMCORR', 'inverse function', 'Atmospheric refraction correction.'))

            extension_names = {'science': 'SCI', 'variance': 'VAR', 'data_quality': 'DQ'}

            for key in self._get_sources():
                name = extension_names[key]
                data = getattr(self, key)
                if isinstance(data, ma.MaskedArray):
                    data = data.data
                if name in old:
                    new.append(fits.ImageHDU(data=data, name=name))
                    self.set_wcs(new[name].header)
                    self.copy_header(old[name].header, new[name].header)
                if key == 'science':
                    new['SCI'].header['BUNIT'] = 'erg/cm2/s/A/arcsec2'

            new.writeto(output, overwrite=overwrite)


class Combine:
    def __init__(self, input_files):
        self.input_files = input_files
        self.n_images = len(input_files)
        self.offsets = np.array([])

        self.science = None
        self.variance = None
        self.data_quality = None
        self.std_dev = None

    def _get_wavelength_end_points(self):
        wep = []
        for name in self.input_files:
            with fits.open(name, mode='readonly') as f:
                w = wcs.WCS(f['sci'].header, naxis=[3])
                wep.append(w.wcs_pix2world([0, f['sci'].data.shape[0]], 0))
        self.wavelength_end_points = np.array(wep)

    def get_offsets(self):
        offsets = []
        for name in self.input_files:
            with fits.open(name, mode='readonly') as f:
                observatory = f['primary'].header['observat'].lower()
                x_off = f['primary'].header['xoffset'] / f['sci'].header['cdelt1']
                y_off = f['primary'].header['yoffset'] / f['sci'].header['cdelt2']
                z_off = f['sci'].header['crval3'] / f['sci'].header['cd3_3']
                if observatory == 'gemini-north':
                    x_off *= -1
                    y_off *= +1
                offsets.append([z_off, y_off, x_off])
        self.offsets = np.round(np.array(offsets) - np.min(offsets, axis=0), decimals=0).astype(int)

    def get_padding(self, name):
        index = self.input_files.index(name)
        offsets = self.offsets
        extremes = {'min': offsets.min(axis=0), 'max': offsets.max(axis=0)}
        pad = []
        current = offsets[index]
        for i in range(offsets.shape[1]):
            p = (abs(current[i] - extremes['min'][i]), abs(current[i] - extremes['max'][i]))
            pad.append(p)
        return pad

    @staticmethod
    def crop_data(data):
        new_data = data[:]
        for axis in range(3):
            min_length = np.min([_.shape[axis] for _ in data])
            for i, j in enumerate(new_data):
                s = [slice(None), slice(None), slice(None)]
                s[axis] = slice(0, min_length)
                new_data[i] = new_data[i][tuple(s)]
        return new_data

    def _combine_data(self, extension, method='mean', mask_data=False):
        data = []
        flags = []

        for name in self.input_files:
            with fits.open(name, 'readonly') as f:
                if extension not in f:
                    return None
                padding = self.get_padding(name)
                data.append(np.pad(f[extension].data, padding, mode='constant', constant_values=0.0))
                if mask_data and ('dq' in f):
                    # The first bit is for generally bad pixels, including regions that are not illuminated.
                    flags.append(np.pad(f['dq'].data, padding, mode='constant', constant_values=1))
                else:
                    flags.append(np.zeros_like(data[-1]).astype('int16'))
        data = self.crop_data(data)
        flags = self.crop_data(flags)
        data = ma.array(data=data, mask=np.array(flags) > 0)
        data.mask |= data == 0

        data = self._match_fluxes(data)

        result = getattr(ma, method)(data, axis=0)
        return result

    @staticmethod
    def _match_fluxes(data, steps=20):
        x = np.arange(data.shape[1])
        s = np.array([(_[1] + _[0]) / 2.0 for _ in np.array_split(x, steps)])

        def get_cf(im1, im2):
            mask = (~im1.mask) & (~im2.mask)
            return ma.median(im1[mask]) / ma.median(im2[mask])

        for i, d in enumerate(data[:-1]):
            a = np.array_split(data[i], steps, axis=0)
            b = np.array_split(data[i + 1], steps, axis=0)
            cf = np.array([get_cf(m, n) for (m, n) in zip(a, b)])

            correction = interp1d(s, cf, assume_sorted=True, kind='cubic', bounds_error=False, fill_value='extrapolate')
            data[i + 1] *= correction(x)[:, np.newaxis, np.newaxis]

        return data

    def _combine_data_quality(self, extension='dq'):
        data = []
        for name in self.input_files:
            with fits.open(name, 'readonly') as f:
                if extension not in f:
                    return None
                padding = self.get_padding(name)
                data.append(np.pad(f[extension].data, padding, mode='constant', constant_values=1).astype('int16'))
        data = np.array(self.crop_data(data))

        r = copy.deepcopy(data[0])
        for i in range(1, len(self.input_files)):
            r |= data[i]
        result = ma.array(r, mask=None)
        return result

    def combine(self):
        self.get_offsets()
        self.science = self._combine_data('sci', method='mean', mask_data=True)
        self.std_dev = self._combine_data('sci', method='std', mask_data=True)

        var = self._combine_data('var', method='mean', mask_data=True)
        if var is not None:
            var /= (len(self.input_files) ** 2)
        self.variance = var
        self.data_quality = self._combine_data_quality()

    def write(self, output, overwrite=False):
        new = fits.HDUList()
        with fits.open(self.input_files[0]) as old:
            new.append(old['PRIMARY'])

            for i, j in enumerate(self.input_files):
                new['primary'].header['IMCB{:04d}'.format(i)] = self.input_files[i]

            extension_names = {'science': 'SCI', 'variance': 'VAR', 'data_quality': 'DQ'}
            for key in ['science', 'variance', 'data_quality']:
                data = getattr(self, key)
                if data is None:
                    continue
                else:
                    name = extension_names[key]
                    hdu = fits.ImageHDU(data=data.data, header=old[name].header, name=name)
                    for i in range(3):
                        hdu.header['naxis{:d}'.format(i + 1)] = data.shape[2 - i]
                    new.append(hdu)

            new.writeto(output, overwrite=overwrite)


class RawCube:
    def __init__(self, file_name):
        with fits.open(file_name) as f:
            self.data = ma.masked_invalid(f['sci'].data)
            self.mdf = table.Table(f['mdf'].data)

    @staticmethod
    def hexagon(x0, y0, radius):
        h = RegularPolygon((x0, y0), numVertices=6, radius=radius, orientation=np.deg2rad(30))
        return h

    def plot_image(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)

        points, shape = get_points(mdf=self.mdf, sampling=0.02)

        image = np.zeros(shape)
        for row in self.mdf[self.mdf['BEAM'] == 1]:
            s = np.median(self.data[row['APID'] - 1])
            h = self.hexagon(x0=row['XINST'], y0=row['YINST'], radius=0.09)

            mask = h.contains_points(points).reshape(shape)
            image[mask] = s

        ax.imshow(image, origin='lower', cmap='plasma')

        plt.show()
