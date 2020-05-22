import copy

import matplotlib.pyplot as plt
import numpy as np
from astropy import table
from astropy.io import fits
from astropy.stats import sigma_clip
from matplotlib.patches import RegularPolygon
from numpy import ma
from scipy import ndimage
from scipy.interpolate import griddata
from scipy.optimize import minimize


def get_points(mdf, sampling=0.02, cushion=0.0):
    object_fibers = mdf[mdf['BEAM'] == 1]

    limits = []
    for dimension in 'XY':
        limits.append(object_fibers['{:s}INST'.format(dimension)].min() - cushion)
        limits.append(object_fibers['{:s}INST'.format(dimension)].max() + cushion)

    coords = [np.arange(limits[2 * _], limits[(2 * _) + 1], sampling) for _ in range(2)]
    x, y = [_.flatten() for _ in np.meshgrid(*coords)]
    shape = tuple([len(coords[_]) for _ in [1, 0]])

    return np.vstack((x, y)).T, shape


class CubeBuilder:
    def __init__(self, file_name):
        with fits.open(file_name) as f:
            self.data = ma.masked_invalid(f['SCI'].data)
            self.mdf = table.Table(f['MDF'].data)

        self.flux_density()
        self.data_cube = None

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

        self.data *= correction_factor

    def get_mean_spectrum(self):
        y, x = np.indices(self.data_cube.shape[1:])
        y0, x0 = ndimage.center_of_mass(self.data_cube.sum(0))
        r = np.sqrt(np.square(x - x0) + np.square(y - y0))

        spectrum = self.data_cube[:, r < 10].sum(1)
        spectrum /= ma.median(spectrum)

        return spectrum, (y0, x0)

    def fit_refraction_function(self, steps=10, degree=3, plot=False, n_iterate=5, sigma_threshold=3):
        mean_spectrum, x0 = self.get_mean_spectrum()
        data = copy.deepcopy(self.data_cube)
        data /= mean_spectrum[:, np.newaxis, np.newaxis]

        total_planes = np.arange(data.shape[0])
        d = np.array([_.sum(0) for _ in np.array_split(data, steps)])
        planes = np.array([((_[-1] + _[0]) / 2) for _ in np.array_split(np.arange(data.shape[0]), steps)])
        centers = np.array([ndimage.center_of_mass(_) for _ in d])
        centers = [x0[_] - centers[:, _] for _ in range(2)]

        shift = []

        for direction in range(2):
            mask = np.zeros_like(planes, dtype=bool)
            for i in range(n_iterate):
                fit = np.polyval(np.polyfit(planes[~mask], centers[direction][~mask], deg=degree), planes)
                mask = sigma_clip(centers[direction] - fit, sigma=sigma_threshold, iters=1).mask
            fit = np.polyval(np.polyfit(planes[~mask], centers[direction][~mask], deg=degree), total_planes)
            shift.append(fit)

        if plot:
            fig = plt.figure()

            for i in range(2):
                ax = fig.add_subplot(1, 2, i + 1)
                ax.scatter(planes, centers[i], c=planes)
                ax.plot(total_planes, shift[i])

            plt.show()

        return shift[::-1]

    def fix_atmospheric_refraction(self):
        x_shift, y_shift = self.fit_refraction_function(steps=20, degree=3, plot=True, sigma_threshold=2, n_iterate=3)
        data = copy.deepcopy(self.data_cube.data)
        data[self.data_cube.mask] = 0.0

        for i, j in enumerate(x_shift):
            data[i] = ndimage.shift(data[i], (y_shift[i], x_shift[i]), mode='constant', cval=0.0)

        self.data_cube = ma.masked_invalid(data)

    def build_cube(self, sampling=0.1):
        n_wavelength = self.data.shape[1]
        points, shape = get_points(mdf=self.mdf, sampling=sampling)
        cube = np.zeros((n_wavelength,) + shape)

        beam_mask = self.mdf['BEAM'] == 1
        x = self.mdf['XINST'][beam_mask].data
        y = self.mdf['YINST'][beam_mask].data
        grid_coordinates = np.vstack([x, y]).T

        print('Building cube')
        k = 0
        step_size = int(n_wavelength / 10)
        for plane in range(self.data.shape[1]):
            if plane % step_size == 0:
                print('{:d}%'.format(k))
                k += 10
            values = np.array(self.data[self.mdf[beam_mask]['APID'].data - 1, plane])
            grid = griddata(grid_coordinates, values=values, xi=points, method='linear')
            cube[plane] = grid.reshape(shape)
        print(' OK!')

        self.data_cube = ma.masked_invalid(cube)

    def write(self, output, overwrite=False):
        fits.writeto(output, data=self.data_cube.data, overwrite=overwrite)


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
