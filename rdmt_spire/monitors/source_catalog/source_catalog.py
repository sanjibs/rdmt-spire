import logging
import os

import asdf
import numpy as np
import pandas as pd
from astropy.table import Table

from rdmt_spire.utilities import aws_utils

from ...constants.source_catalog_constants import (
    SOURCE_CATALOG_PROPERTIES,
    SOURCE_CATALOG_STATISTICS,
)
from ..monitor_base import BaseMonitor

logger = logging.getLogger(__name__)

# Get clarity on handling errors. For example, if the filter is not found in the calibration files.
# Should it return an empty dictionary and log the error?
# Also, what is the expected behavior of the monitor if the input file (catalog.parquet) is missing?
# Should it return an empty dictionary and log the error?

class SourceCatalogMonitor(BaseMonitor):
    """
    Source Catalog Monitor derived from class BaseMonitor.
    Reads the source catalog (parquet format) created from an L2 file.
    Bins point sources by brightness and tracks properties:
    sharpness, roundness1, ellipticity, flux_frac_radius_50, flux ratios,
    and flux error ratio.
    """

    def __init__(self, asdf_file: asdf.AsdfFile, config: dict[str, str] | str | os.PathLike,):
        super().__init__(asdf_file)
        self.monitor_name = "source_catalog"
        self.config = config
        self.log.append(f"{self.monitor_name}: initialized")
        # Define properties and statistics for the metrics
        self.properties = SOURCE_CATALOG_PROPERTIES.copy()
        self.statistics = SOURCE_CATALOG_STATISTICS.copy()
        # Expected values for each property for a given filter, table pandas.DataFrame
        self.expected_props = self._load_data_file("expected_photometric_properties.ecsv")

    def _load_data_file(self, filename: str) -> pd.DataFrame:
        """
        Loads a local calibration or parameter file under the data/ directory.
        """        
        path = os.path.join(os.path.dirname(__file__), 'data', filename)
        if filename.endswith('.ecsv'):
            with open(path, 'r') as f:
                content = f.read()
            df = Table.read(content.replace('\t', ' '), format="ascii.ecsv").to_pandas()
        else:
            df = Table.read(path).to_pandas()
        
        # clean up filter column to avoid mismatch caused by different naming conventions in files
        if 'filter' in df.columns:
            df['filter'] = df['filter'].str.strip().str.lower()
        return df

    def calculate_metrics(self):
        """
        Calculates source catalog metrics binned by magnitude.
        Adds median, STD, and NMAD metrics for 8 properties across bright and faint bins.
        """

        # Do we need a  Fallback logic for tests to return dummy values
        # if self.asdf_file.uri is None:

        # Should we use (try,except,raise) clause to log the error
        # except (KeyError, TypeError) as e:
        #    raise RuntimeError(f"SourceCatalogMonitor: failed to retrieve parameter from asdf file: {e}")
        
        # 2. Parse needed values from the ASDF file.
        filename = self.asdf_file["roman"]["meta"]["filename"]
        optical_filter = self.asdf_file["roman"]["meta"]["instrument"]["optical_element"].strip().lower()
        t_exp = self.asdf_file["roman"]["meta"]["exposure"]["exposure_time"]


        # 3. Load calibration parameters
        zero_points = self._load_data_file("zero_points_20260401.ecsv")
        filter_params = self._load_data_file("filter_parameters.ecsv")
        thermal_bkg = self._load_data_file("internal_thermal_backgrounds.ecsv")
        zodiacal_light = self._load_data_file("zodiacal_light.ecsv")

        # Lookup parameters helper
        def get_val(df, col, filter_val):
            rows = df[df['filter'] == filter_val]
            if rows.empty:
                raise RuntimeError(f"SourceCatalogMonitor: filter '{filter_val}' not found in parameter table")
            return rows[col].values[0]

        zp = get_val(zero_points, 'Z_R', optical_filter)
        n_eff = get_val(filter_params, 'center_PSF_n_eff_pixel', optical_filter)
        f_peak = get_val(filter_params, 'center_PSF_peak_flux', optical_filter)
        f_thermal = get_val(thermal_bkg, 'rate', optical_filter)
        f_min_zodi = get_val(zodiacal_light, 'rate', optical_filter)


        # 4. Construct parquet catalog file path and load it
        file_object = aws_utils.load_file_object(self.config["RDMT_SPIRE_L4_DIR"], filename.replace('_cal.asdf', '_cat.parquet'))
        df = pd.read_parquet(file_object)

        # 5. Point source selection
        if 'is_extended' not in df.columns:
            raise RuntimeError("SourceCatalogMonitor: 'is_extended' column missing from source catalog")
        df_pts = df[~df['is_extended']].copy()

        # 6. Calculate magnitude bin boundaries
        # Saturation mag: m_sat = Zp - 2.5 * log10(alpha_sat)
        # alpha_sat = c_sat / (f_peak * t_exp), where c_sat = 120,000
        def saturation_limit_mag(c_sat, t_exp, zp, f_peak):
            alpha_sat = c_sat / (f_peak * t_exp)
            return zp - 2.5 * np.log10(alpha_sat)
        m_sat = saturation_limit_mag(120000.0, t_exp, zp, f_peak)

        # Faint limit from SNR = 50 quadratic equation
        def faint_limit_mag(snr, t_exp, zp, n_eff, f_bkgd):
            snr2 = snr ** 2
            term_b = snr2 / t_exp
            term_c = (snr2 * n_eff * f_bkgd) / t_exp
            f_source_limit = (term_b + np.sqrt(term_b**2 + 4.0 * term_c)) / 2.0
            m_faint = zp - 2.5 * np.log10(f_source_limit)
            return m_faint

        # Compute theoretical PSF flux error for each source
        def psf_flux_error_theory(psf_flux, t_exp, zp, n_eff, f_bkgd):
            kappa = 10**((31.4 - zp) / 2.5)
            f_src = psf_flux / kappa
            term_err = t_exp * (n_eff * f_bkgd + f_src)
            term_err_clipped = np.clip(term_err, 0.0, None)
            sigma_f = np.sqrt(term_err_clipped) / t_exp
            psf_flux_err = sigma_f * kappa
            psf_flux_err[psf_flux_err==0.0] = np.nan
            return psf_flux_err

        f_bkgd = 2.0 * f_min_zodi + f_thermal
        m_faint = faint_limit_mag(50.0, t_exp, zp, n_eff, f_bkgd)

        m_mid = (m_sat + m_faint) / 2.0

        self.log.append(f"m_sat: {m_sat:.4f}, m_mid: {m_mid:.4f}, m_faint: {m_faint:.4f}")

        # 7. Subdivide sources
        mab = -2.5 * np.log10(df_pts['psf_flux']) + 31.4
        cond_bright = (mab > m_sat) & (mab <= m_mid)
        cond_faint = (mab > m_mid) & (mab < m_faint)

        # self.append_data("num_sources_bright", np.sum(cond_bright), "")
        # self.append_data("num_sources_faint", np.sum(cond_faint), "")

        # 8. Compute property arrays helper

        def compute_properties(df_sub):
            if df_sub.empty:
                return {}

            props = {}
            props["sharpness"] = df_sub["sharpness"].values
            props["roundness1"] = df_sub["roundness1"].values
            props["ellipticity"] = df_sub["ellipticity"].values
            props["flux_frac_radius_50"] = df_sub["fluxfrac_radius_50"].values

            # Division by zero handled by replacing with NaN
            props["flux_ratio_aper01_aper02"] = (df_sub["aper02_flux"] / df_sub["aper01_flux"].replace(0, np.nan)).values
            props["flux_ratio_aper02_aper04"] = (df_sub["aper04_flux"] / df_sub["aper02_flux"].replace(0, np.nan)).values
            props["flux_ratio_aper04_aper08"] = (df_sub["aper08_flux"] / df_sub["aper04_flux"].replace(0, np.nan)).values

            # PSF flux error
            flux_err_theory = psf_flux_error_theory(df_sub["psf_flux"].values, t_exp, zp, n_eff, f_bkgd)
            props["flux_err_ratio_psf_theory"] = df_sub["psf_flux_err"].values / flux_err_theory

            # # Theoretical PSF flux error
            # f_src = df_sub["psf_flux"].values / kappa
            # term_err = t_exp * (n_eff * f_bkgd + f_src)
            # # to avoid invalid value error in sqrt
            # term_err_clipped = np.clip(term_err, 0.0, None)
            # sigma_f = np.sqrt(term_err_clipped) / t_exp
            # psf_flux_err_theory = sigma_f * kappa
            # # to avoid invalid value error
            # psf_flux_err_theory[psf_flux_err_theory==0.0] = np.nan
            # props["flux_err_ratio_psf_theory"] = df_sub["psf_flux_err"].values / psf_flux_err_theory
            return props

        df_props = compute_properties(df_pts)

        # 9. Calculate and append median, stddev, and p16, p84 metrics
        for prop in self.properties:
            if prop.endswith("bright"):
                vals = df_props[prop.rstrip('_bright')][cond_bright]
            else:
                vals = df_props[prop.rstrip('_faint')][cond_faint]
            vals=vals[np.isfinite(vals)]
            # can be set higher to have robust statistics
            if len(vals) <= 1:
                percentile_vals = [np.nan, np.nan, np.nan, np.nan, np.nan]
                mean_val = np.nan
                std_val = np.nan
            else:
                mean_val = float(np.mean(vals))
                std_val = float(np.std(vals))
                percentile_vals = np.percentile(vals, [2.275, 15.86, 50, 84.14, 97.725])

            self.append_data(f"{prop}_n_sources", len(vals), "")
            self.append_data(f"{prop}_median", percentile_vals[2], "")
            self.append_data(f"{prop}_dispersion_p68", (percentile_vals[3]-percentile_vals[1])/2.0, "")
            self.append_data(f"{prop}_dispersion_p95", (percentile_vals[4]-percentile_vals[0])/4.0, "")
            self.append_data(f"{prop}_mean", mean_val, "")
            self.append_data(f"{prop}_std", std_val, "")

    def _update_metric_evaluation(self, metric_name, error, expected_props, optical_filter):
        """
        Update the metric evaluation status to False if the metric value falls outside the expected range
        of a metric based on its expected properties and error else leave the status unchanged.

        Parameters
        ----------
        metric_name : str
            Name of the metric to update.
        error : float
            The estimated error for the metric.
        expected_props : pandas.DataFrame
            Table containing the expected properties for the given filter.
        optical_filter : str
            The optical filter being used. e.g. f062.
        """
        data_value = self.get_data(metric_name)
        if np.isfinite(data_value):
            expected_properties=expected_props[expected_props['property_name'] == f"{metric_name}_{optical_filter}"]
            if expected_properties.empty:
                raise RuntimeError(f"SourceCatalogMonitor: filter '{metric_name}_{optical_filter}' not found in expected properties table")
            min = expected_properties['min'].values[0] + 3*error
            max = expected_properties['max'].values[0] - 3*error
            if (data_value<min) | (data_value>max):
                self.add_evaluation(metric_name, False)


    def evaluate_metrics(self):
        """
        Evaluate metrics against validity (checking they are not None and finite).
        When number of sources does not meet the minimum requirement, n_sources is evaluated and set to False, 
        but the evaluation for median and dispersion metrics is skipped.
        """

        optical_filter = self.asdf_file["roman"]["meta"]["instrument"]["optical_element"].strip().lower()


        for property in self.properties:
            # First we evaluate the basic validity of each metric and set evalutation status to 
            # False if it is not valid and True if it is valid
            for stat_name in self.statistics:
                metric_name = f"{property}_{stat_name}"
                data_value = self.get_data(metric_name)
                is_valid = self.is_valid_metric(metric_name, data_value)
                if is_valid:
                    self.add_evaluation(metric_name, True)

            # For a few specific properties, e.g., median and dispersion, 
            # we do a more rigorous evaluation of the metrics and 
            # set the evalutation status to False if it fails certain criterion

            # These are needed for the rigorous evaluation of the median and dispersion metrics
            dispersion = self.get_data(property+"_dispersion_p68")
            n_sources = self.get_data(property+"_n_sources")

            if n_sources >= 10:
                # median and dispersion metrics are evaluted only if there are enough sources
                if np.isfinite(dispersion):
                    # formula for standard error of the median
                    error=dispersion/np.sqrt(n_sources)
                    self._update_metric_evaluation(f"{property}_median", error, self.expected_props, optical_filter)

                    # formula for std deviation of std deviation
                    error=dispersion/np.sqrt(2*(n_sources-1))
                    self._update_metric_evaluation(f"{property}_dispersion_p68", error, self.expected_props, optical_filter)
            else:   
                # Not enough sources to evaluate the metric
                self.add_evaluation(f"{property}_n_sources", False)


