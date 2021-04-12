import numpy as np
from astropy.modeling import Fittable1DModel, Parameter


class DifferentialRefraction(Fittable1DModel):
    """
    See Filippenko, A. 1982 (PASP 94, 715).

    Wavelengths must be in microns!
    """
    n_inputs = 1
    n_outputs = 1

    temperature = Parameter(default=7.0, min=-60.0, max=40.0)
    pressure = Parameter(default=500.0, min=100.0, max=800.0)
    water_vapour = Parameter(default=8.0, min=0.0, max=800.0)
    air_mass = Parameter(default=1.5, min=1.0, max=5.0)
    wl_0 = Parameter(default=5000.0)

    @staticmethod
    def evaluate(x, temperature, pressure, water_vapour, air_mass, wl_0):

        def n_lambda(wavelength):
            r = np.square(1.0 / wavelength)
            a = 29498.1 / (146.0 - r)
            b = 255.4 / (41.0 - r)
            n_sea_level = 1e-6 * (64.328 + a + b)

            a = 1.0 + (1.049 - (0.0157 * temperature)) * 1e-6 * pressure
            b = 720.883 * (1.0 + 0.003661 * temperature)
            n_tp = n_sea_level * pressure * (a / b)

            a = 0.0624 - (0.000680 * r)
            b = 1.0 + (0.003661 * temperature)
            water_factor = water_vapour * (a / b) * 1e-6

            return n_tp - water_factor

        k = np.tan(np.arcsin(1 / air_mass))
        delta_r = 206265.0 * (n_lambda(x * 1e-4) - n_lambda(wl_0 * 1e-4)) * k
        return delta_r
