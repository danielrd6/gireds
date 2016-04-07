import numpy as np
import pyfits as pf
import spectools as st
from scipy.optimize import minimize
from scipy.interpolate import interp1d


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

    flambda = fnuzero * 10 ** (-0.4 * mag) * 2.99792458e+18 / refwl**2

    if extinction is None:
        c = 2.5 * np.log10(obsbp / exptime / bp / flambda)
    else:
        e = np.loadtxt(extinction)
        ext = interp1d(e[:, 0], e[:, 1])
        c = 2.5 * (np.log10(obsbp / exptime / bp / flambda)) + \
            airmass * ext(refwl)

    return refwl, obsbp, c, flambda
