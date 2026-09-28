
import numpy as np
from astropy import units as u

from .aws_utils import csv2qtable


class RomanWFIPhotometricParameters:
    """
    Class to handle Roman WFI parameters related to photometry.
    """

    def __init__(self, config: dict | None = None):
        """Initialize RomanWFIPhotometricParameters with default data files for
        zero points, filter parameters, thermal backgrounds, and zodiacal light.
        source: https://github.com/RomanSpaceTelescope/roman-technical-information

        Parameters
        ----------
        config : dict or None
            Configuration dictionary containing paths to data directories. If None, values are loaded from the .env file.
            Required keys: ['RDMT_SPIRE_LDATA_DIR']

        """
        self.config = config
        prefix="roman_technical_information/"
        # Tables with parameters for different filters
        self.zero_points = csv2qtable(
            self.config["RDMT_SPIRE_LDATA_DIR"], prefix + "zero_points_20260401.ecsv"
        )
        self.filter_params = csv2qtable(
            self.config["RDMT_SPIRE_LDATA_DIR"], prefix + "filter_parameters.ecsv"
        )
        self.thermal_bkg = csv2qtable(
            self.config["RDMT_SPIRE_LDATA_DIR"], prefix + "internal_thermal_backgrounds.ecsv"
        )
        self.zodiacal_light = csv2qtable(
            self.config["RDMT_SPIRE_LDATA_DIR"], prefix + "zodiacal_light.ecsv"
        )

        # Dictionary to store the properties for one particular optical filter
        self._properties = {"optical_filter": ""}

        # Check that all loaded data files have the correct units
        self.check_units()

    def check_units(self):
        """Check and ensure that all loaded data files have the correct units"""
        df = self.zero_points
        assert df["z_r"].unit == u.mag
        assert df["sigma_z"].unit == u.mag
        assert df["alpha_minr"].unit == u.ct

        df = self.filter_params
        assert df["center_psf_n_eff_pixel"].unit == u.pixel
        assert df["center_psf_peak_flux"].unit is None

        df = self.thermal_bkg
        assert df["rate"].unit == (u.ct / (u.pix * u.s))

        df = self.zodiacal_light
        assert df["rate"].unit == (u.ct / (u.pix * u.s))

    def _setup(self, optical_filter: str):
        """Setup the RomanWFIPhotometricParameters for the specified optical filter.

        Parameters
        ----------
        optical_filter : str
            The optical filter to use in lowercase for the Roman WFI properties.
        """
        if self._properties["optical_filter"] != optical_filter:
            self._properties = {"optical_filter": optical_filter}
            self._properties["zp"] = self._get_val(
                self.zero_points, "z_r", optical_filter
            )
            self._properties["n_eff"] = self._get_val(
                self.filter_params, "center_psf_n_eff_pixel", optical_filter
            )
            self._properties["f_peak"] = self._get_val(
                self.filter_params, "center_psf_peak_flux", optical_filter
            )
            self._properties["f_thermal"] = self._get_val(
                self.thermal_bkg, "rate", optical_filter
            )
            self._properties["f_min_zodi"] = self._get_val(
                self.zodiacal_light, "rate", optical_filter
            )

    def get_psf_fwhm(self, optical_filter: str):
        """Get the full-width at half-maximum (FWHM) of the PSF for the current optical filter."""
        return self._get_val(self.filter_params, "psf_fwhm", optical_filter)


    @staticmethod
    def ab_magnitude(flux):
        """Convert a flux to AB magnitude.

        Parameters
        ----------
        flux : astropy.units.Quantity
            The flux to convert to AB magnitude.

        Returns
        -------
        astropy.units.Quantity
            The AB magnitude corresponding to the input flux.
        """
        # return -2.5 * np.log10(flux.to_value('nJy')) + 31.4 * u.mag
        return (-2.5 * np.log10(flux.to_value("Jy")) + 8.9) * u.mag

    def _get_val(self, df, column, optical_filter):
        """Get the value from the DataFrame for the specified column and optical filter."""
        cond = np.char.lower(df["filter"]) == optical_filter
        if cond.sum() == 0:
            raise RuntimeError(
                f"SourceCatalogMonitor: filter '{optical_filter}' not found in parameter table"
            )
        return df[column][cond][0]

    @staticmethod
    def _saturation_limit_mag(c_sat, t_exp, zp, f_peak):
        """Calculate the saturation magnitude for the given parameters.

        Parameters
        ----------
        c_sat : float [ct / (pix s)]
            Saturation count level.
        t_exp : float [s]
            Exposure time in seconds.
        zp : astropy.units.Quantity [mag]
            Zero point magnitude.
        f_peak : astropy.units.Quantity [ct / (pix s)]
            Peak flux of the PSF.

        Returns
        -------
        astropy.units.Quantity
            Saturation magnitude.
        """
        alpha_sat = c_sat / (f_peak * t_exp * u.pix)
        return zp - 2.5 * np.log10(alpha_sat.value) * u.mag

    @staticmethod
    def _faint_limit_mag(snr, t_exp, zp, n_eff, f_bkgd):
        """faint limit flux for a given snr based on a quadratic equation
        for f_source_limit obtained by rearranging the SNR equation:
        snr=f_source_limit*t_exp / np.sqrt((n_eff * f_bkgd + f_source_limit)*t_exp)

        Parameters
        ----------
        snr : float
            Signal-to-noise ratio.
        t_exp : float [s]
            Exposure time in seconds.
        zp : astropy.units.Quantity [mag]
            Zero point magnitude.
        n_eff : float [pix]
            Effective number of pixels.
        f_bkgd : astropy.units.Quantity [ct / (pix s)]
            Background flux.

        Returns
        -------
        astropy.units.Quantity [mag]
            Faint limit magnitude.
        """
        snr2 = snr**2
        term_b = snr2 / t_exp
        term_c = (snr2 * n_eff * f_bkgd / u.ct) / t_exp
        f_source_limit = (term_b + np.sqrt(term_b**2 + 4.0 * term_c)) / 2.0
        m_faint = zp - 2.5 * np.log10(f_source_limit.value) * u.mag
        return m_faint

    def get_psf_flux_error_theory(self, optical_filter, psf_flux, t_exp):
        """Compute the theoretical PSF flux error based on poisson statistics
        related to source and background flux.
        flux_err = np.sqrt(counts)/t_exp
                 = np.sqrt(t_exp * (n_eff * f_bkgd + f_src))/t_exp

        Parameters
        ----------
        optical_filter : str
            Name of the optical filter.
        psf_flux : astropy.units.Quantity [ct / s]
            PSF flux of the source.
        t_exp : astropy.units.Quantity [s]
            Exposure time.

        Returns
        -------
        astropy.units.Quantity [ct / s]
            Theoretical PSF flux error.
        """
        self._setup(optical_filter)

        f_bkgd = 2.0 * self._properties["f_min_zodi"] + self._properties["f_thermal"]
        zp = self._properties["zp"]
        n_eff = self._properties["n_eff"]
        # Conversion factor from cts to nJy based on zero point magnitude
        kappa = 10 ** ((31.4 - zp.value) / 2.5) * (u.nJy / (u.ct / u.s))
        f_src = psf_flux / kappa

        counts = t_exp * (n_eff * f_bkgd + f_src)
        counts = np.clip(counts, 0.0, None)
        psf_flux_err = u.ct * np.sqrt(counts.value) / t_exp
        psf_flux_err = psf_flux_err * kappa
        psf_flux_err = np.where(psf_flux_err == 0.0, np.nan, psf_flux_err)

        if np.ndim(psf_flux) == 0:
            return psf_flux_err.item()
        return psf_flux_err

    def get_msat(self, optical_filter, t_exp):
        """Get the saturation magnitude for the given filter and exposure time.

        Parameters
        ----------
        optical_filter : str
            Name of the optical filter.
        t_exp : astropy.units.Quantity [s]
            Exposure time.

        Returns
        -------
        astropy.units.Quantity [mag]
            Saturation magnitude.
        """
        self._setup(optical_filter)
        return self._saturation_limit_mag(
            120000.0 * u.ct, t_exp, self._properties["zp"], self._properties["f_peak"]
        )

    def get_mfaint(self, optical_filter, t_exp):
        """Get the faint magnitude limit for the given filter and exposure time.

        Parameters
        ----------
        optical_filter : str
            Name of the optical filter.
        t_exp : astropy.units.Quantity [s]
            Exposure time.

        Returns
        -------
        astropy.units.Quantity [mag]
            Faint magnitude limit.
        """
        self._setup(optical_filter)
        f_bkgd = 2.0 * self._properties["f_min_zodi"] + self._properties["f_thermal"]
        return self._faint_limit_mag(
            50.0, t_exp, self._properties["zp"], self._properties["n_eff"], f_bkgd
        )
