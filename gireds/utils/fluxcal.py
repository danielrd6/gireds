import numpy as np
import pyfits as pf
import ifscube.spectools as st
from scipy.optimize import minimize
from scipy.interpolate import interp1d


class stdstar():
    """
    A class for the data output of IRAF's standard task.
    """

    def __init__(self, std):

        self.title = std[0].split()
        self.exptime, self.airmass = [float(self.title[i]) for i in [3, 4]]

        self.wl, self.flambda, self.bandpass, self.adu =\
            np.loadtxt(std, skiprows=1, unpack=True)

        self.c = 2.5 * np.log10(
            self.adu / self.exptime / self.bandpass / self.flambda)

        return

    def extinction(self, extfile):
        """
        Not working yet!
        """

        # e = np.loadtxt(extfile)
        # ext = interp1d(e[:, 0], e[:, 1])
        # c = 2.5 * (np.log10(obscounts / exptime / bp / flambda)) + \
        #            airmass * ext(wl)

        pass

    def interp(self, bounds_error=False, fill_value=np.nan):

        self.adu_interp = interp1d(
            self.wl, self.adu, bounds_error=bounds_error,
            fill_value=fill_value)
        self.flambda_interp = interp1d(
            self.wl, self.flambda, bounds_error=bounds_error,
            fill_value=fill_value)
        self.c_interp = interp1d(
            self.wl, self.c, bounds_error=bounds_error,
            fill_value=fill_value)

        return


def sensitivity(observed, reference, extinction=None, fnuzero=3.68e-20):
    """
    Function designed fit a sensitivity function based on the
    comparison between the osbervation of a standard star and its
    flux calibrated spectrum.
    """

    h = pf.getheader(observed, ext=1)
    exptime, airmass = h['exptime'], h['airmass']
    obswl = st.get_wl(observed, hdrext=1, dataext=1)
    refwl, mag, bp = np.loadtxt(reference, unpack=True)

    trim = (refwl - bp[0] > obswl[0]) & (refwl + bp[-1] < obswl[-1])
    refwl, mag, bp = refwl[trim], mag[trim], bp[trim]

    obsdata = pf.getdata(observed, ext=1)

    obsbp = np.array([
        sum(obsdata[(obswl > refwl[i] - bp[i]/2.) & (obswl < refwl[i] +
                    bp[i]/2.)]) for i in range(len(refwl))])

    flambda = fnuzero * 10 ** (-0.4 * mag) * 2.99792458e+18 / refwl ** 2

    if extinction is None:
        c = 2.5 * np.log10(obsbp / exptime / bp / flambda)
    else:
        e = np.loadtxt(extinction)
        ext = interp1d(e[:, 0], e[:, 1])
        c = 2.5 * (np.log10(obsbp / exptime / bp / flambda)) + \
            airmass * ext(refwl)

    return refwl, obsbp, c, flambda


def sensfunc(std, extinction=None, fnuzero=3.68e-20, mask=None,
             shift=False):
    """
    Fits a sensibility function to a table provided by IRAF's
    standard task.
    """

    with open(std, 'r') as f:
        fl = f.readlines()

    idx = [i for i in range(len(fl)) if '.fits' in fl[i]]
    stars = []
    
    for i, j in enumerate(idx):
        if j != idx[-1]:
            stars.append(stdstar(fl[j:idx[i + 1]]))
        else:
            stars.append(stdstar(fl[j:]))

    if len(stars) > 1:
        for i in stars:
            i.interp()

        x = np.unique(np.concatenate([i.wl for i in stars]))
        y = np.nanmean([s.adu_interp(x) for s in stars], 0)

    else:
        x = stars[0].wl
        y = stars[0].adu


    return x, y
