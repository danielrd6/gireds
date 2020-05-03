import copy
import inspect

import matplotlib.pyplot as plt
import numpy as np
from astropy import table
from astropy.convolution import convolve, Gaussian1DKernel
from astropy.io import fits
from numpy import ma
from scipy import signal
from scipy.interpolate import interp1d


def vertical_profile(data, column=None, width=10):
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


def plot_results(x, p, xp, yp):
    fig = plt.figure(1)
    fig.clf()
    ax = fig.add_subplot(111)

    ax.plot(x, p)
    ax.scatter(xp, yp, marker='.', s=50, color='red')

    plt.show()


def read_apertures(fname):
    with open(fname, 'r') as f:
        a = [i for i in f.readlines() if (('begin' in i) or ('title' in i))]

    t = table.Table([
        table.Column(name='line', dtype=float),
        table.Column(name='num', dtype=int),
        table.Column(name='bundle', dtype='S10'),
        table.Column(name='fiber', dtype=int),
    ])

    for i in range(0, len(a), 2):
        t.add_row([
            a[i].split()[-1].strip(),
            a[i].split()[-3].strip(),
            a[i + 1].split()[-1].split('_')[0].strip(),
            a[i + 1].split()[-1].split('_')[1].strip()])

    return t


def fix_missing(apertures, idx):
    for i in ['bundle', 'fiber']:
        apertures[i][idx:-1] = apertures[i][(idx + 1):]
    apertures.remove_row(-1)

    return


def peaks_to_distances(fun):
    def wrapper(peak_positions):
        distance = np.diff(peak_positions)
        median_distance = np.median(distance)
        fun_args = inspect.getargspec(fun)
        if 'distance' in fun_args.args:
            return fun(peak_positions, distance, median_distance)
        else:
            return fun(peak_positions, median_distance)

    return wrapper


class AutoApertures:
    def __init__(self, flat_field, over_sample=10, flux_threshold=30, min_sep=2):
        self.over_sample = over_sample
        self.file_name = flat_field
        self.flux_threshold = flux_threshold
        self.min_sep = min_sep
        with fits.open(flat_field) as hdu:
            self.mdf = table.Table(hdu['mdf'].data)
            self.data = [_.data for _ in hdu if _.name == 'SCI']
            self.total_sections = len(self.data)
            self.section_shape = self.data[0].shape
            self.total_columns = self.total_sections * self.section_shape[1]

        self.image = np.column_stack(self.data)

        fibers_per_slit = 750

        self.n_beams = len(self.mdf)

        if self.n_beams == fibers_per_slit:
            self.slits = 1
            self.column = [3000]
        elif self.n_beams == (2 * fibers_per_slit):
            self.slits = 2
            self.column = [1700, 5000]
        else:
            raise RuntimeError('Could not infer the slit mask from the length of the MDF table.')

        self.dead_beams = []

    def _remove_last(self, last_fiber=750, n=3):
        c = 0
        i = last_fiber - 1
        while c < n:
            if self.mdf[i]['BEAM'] == 1:
                self.mdf[i]['BEAM'] = -1
                c += 1
            i -= 1

    def fix_mdf(self, remove_last=(0, 0)):
        self.mdf['BEAM'] = np.ones(self.slits * 750)
        for beam in self.dead_beams:
            self.mdf[beam]['BEAM'] = -1

        # Take a couple more out
        self._remove_last(last_fiber=750, n=remove_last[0])
        if self.slits == 2:
            self._remove_last(last_fiber=1500, n=remove_last[1])

        with fits.open(self.file_name, mode='update') as hdu_list:
            table_hdu = fits.table_to_hdu(self.mdf)
            table_hdu.name = 'MDF'
            hdu_list['mdf'] = table_hdu

    def write_mdf_file(self, output_name):
        hdu = fits.table_to_hdu(self.mdf)
        hdu.name = 'MDF'
        new = fits.HDUList()
        new.append(hdu)
        new.writeto(output_name, overwrite=True)

    def find_peaks(self, column):
        profile = vertical_profile(self.image, column=column)
        x = np.arange(profile.size)

        new_x, smooth_profile = smooth(x, profile, over_sample=self.over_sample)
        peak_indices, bogus = signal.find_peaks(smooth_profile, height=np.percentile(profile, self.flux_threshold))
        peak_coordinates = new_x[peak_indices]
        return peak_coordinates

    def get_dead_beams(self):
        for i in range(self.slits):
            peak_coordinates = self.find_peaks(self.column[i])
            beams = self.find_dead_beams(peak_coordinates)
            dead = np.where(beams == -1)[0].tolist()
            self.dead_beams += [_ + (i * 750) for _ in dead]
        print('Dead beams: ', self.dead_beams)

    @staticmethod
    def _peak_stats(peak_positions):
        distance = np.diff(peak_positions)
        fiber_distance = np.median(distance)
        gap_distance = np.median(distance[distance > (3.0 * fiber_distance)])
        gap_threshold = gap_distance + (2.0 * distance[distance > (3.0 * fiber_distance)].std())
        return distance, fiber_distance, gap_threshold

    @staticmethod
    def _find_gaps(distance, fiber_distance):
        gaps = np.where(distance > (3.0 * fiber_distance))[0] + 1
        gaps = np.concatenate([[0], gaps, [len(distance) + 1]]).astype('int16')
        return gaps

    def find_dead_beams(self, peak_positions):
        original_peak_positions = copy.deepcopy(peak_positions)
        expected_fibers = 750
        expected_fibers_per_bundle = 50

        distance, fiber_distance, gap_threshold = self._peak_stats(peak_positions)

        # Array of gaps indexes, including first and last.
        gap_locations = self._find_gaps(distance, fiber_distance)

        beams = np.ones(expected_fibers)
        bundle_index = 0
        while (len(peak_positions) < 750) and (bundle_index < 15):
            fibers_in_bundle = (gap_locations[bundle_index + 1] - gap_locations[bundle_index])
            bundle_positions = peak_positions[gap_locations[bundle_index]:gap_locations[bundle_index + 1]]

            if fibers_in_bundle < expected_fibers_per_bundle:
                if self._has_missing_fiber(bundle_positions):
                    bundle_positions = self.middle_fix(bundle_positions)
                elif bundle_index == 0:
                    right_gap = distance[gap_locations[bundle_index + 1] - 1]
                    if right_gap > gap_threshold:
                        bundle_positions = self.right_fix(bundle_positions)
                    else:
                        bundle_positions = self.left_fix(bundle_positions)
                elif bundle_index == 14:
                    left_gap = distance[gap_locations[bundle_index] - 1]
                    if left_gap > gap_threshold:
                        bundle_positions = self.left_fix(bundle_positions)
                    else:
                        bundle_positions = self.right_fix(bundle_positions)
                else:
                    left_gap = distance[gap_locations[bundle_index] - 1]
                    if left_gap > gap_threshold:
                        bundle_positions = self.left_fix(bundle_positions)
                    else:
                        bundle_positions = self.right_fix(bundle_positions)

                peak_positions = np.concatenate([
                    peak_positions[:gap_locations[bundle_index]], bundle_positions,
                    peak_positions[gap_locations[bundle_index + 1]:]])
                distance, fiber_distance, gap_threshold = self._peak_stats(peak_positions)
                gap_locations = self._find_gaps(distance, fiber_distance)

            else:
                print('Bundle {:d} is complete.'.format(bundle_index))
                bundle_index += 1

        for i in range(beams.size):
            if peak_positions[i] not in original_peak_positions:
                beams[i] = -1

        return beams

    @staticmethod
    def _has_missing_fiber(positions):
        distance = np.diff(positions)
        missing = np.any(distance > (np.median(distance) * 1.5))
        return missing

    @staticmethod
    @peaks_to_distances
    def right_fix(peak_positions, median_distance):
        peak_positions = np.concatenate([peak_positions, peak_positions[-1] + [median_distance]])
        return peak_positions

    @staticmethod
    @peaks_to_distances
    def left_fix(peak_positions, median_distance):
        peak_positions = np.concatenate([peak_positions[0] - [median_distance], peak_positions])
        return peak_positions

    @staticmethod
    @peaks_to_distances
    def middle_fix(peak_positions, distance, median_distance):
        dead_fibers = []
        for i, peak in enumerate(peak_positions[:-1]):
            if distance[i] > (median_distance * 1.5):
                dead_fibers.append(i)
                peak_positions = np.concatenate(
                    [peak_positions[:i + 1], peak_positions[i] + [median_distance], peak_positions[i + 1:]])
                break
        return peak_positions


def main(flat_field, over_sample=10, flux_threshold=30, min_sep=2):
    a = AutoApertures(flat_field, flux_threshold=flux_threshold, over_sample=over_sample, min_sep=min_sep)
    a.get_dead_beams()
    a.fix_mdf()
