"""Photometric monitoring metrics for Roman WFI source catalogs.

This module defines :class:`PhotoAstroMonitor`, which derives summary metrics
for point-source photometry from a source catalog produced for a Roman L2 image.
"""

import logging

import asdf
import numpy as np
from astropy import units as u
from astropy.table import Table

from ...constants import photoastro_constants
from ...utilities import aws_utils
from ...utilities.phot.photoastro_utils import crossmatch_with_gaia
from ...utilities.property import Property
from ...utilities.techinfo import RomanWFIPhotometricParameters
from ..monitor_base import BaseMonitor
from .aperture_psf import PhotometryCatalogPipeline

logger = logging.getLogger(__name__)

# Get clarity on handling errors. For example, if the filter is not found in the calibration files.
# Should it return an empty dictionary and log the error?
# Also, what is the expected behavior of the monitor if the input file (catalog.parquet) is missing?
# Should it return an empty dictionary and log the error?


class PhotoAstroMonitor(BaseMonitor):
    """Monitor photometric quality metrics for a Roman WFI catalog.

    The monitor reads the catalog generated for an L2 image, selects point
    sources, bins them by magnitude, and computes summary statistics for
    aperture ratios, encircled-energy radii, and PSF-flux error metrics.
    """

    def __init__(
        self,
        asdf_file: asdf.AsdfFile,
        config: dict[str, str],
    ) -> None:
        super().__init__(asdf_file)
        self.monitor_name = "photoastro"
        self.config = config
        self.log.append(f"{self.monitor_name}: initialized")
        # Define properties and statistics for the metrics
        self.properties = [prop_name for prop_name, *rest in photoastro_constants.PROPERTIES]


    def calculate_metrics(self) -> None:
        """Compute monitor metrics from the source catalog and store them.

        The metrics are generated for bright and faint magnitude bins and stored
        on the monitor instance via :meth:`append_data`.
        """
        # Parse needed values from the ASDF file.
        optical_filter = self.asdf_file["roman"]["meta"]["instrument"]["optical_element"].strip().lower()
        t_exp = self.asdf_file["roman"]["meta"]["exposure"]["exposure_time"] * u.s

        # Create load the source catalog
        df_cat = PhotometryCatalogPipeline(self.asdf_file, self.config).catalog


        # Calculate bright and faint magnitude limits
        wfi_properties = RomanWFIPhotometricParameters(self.config)
        m_faint = wfi_properties.get_mfaint(optical_filter, t_exp)
        m_sat = wfi_properties.get_msat(optical_filter, t_exp)
        m_mid = (m_sat + m_faint) / 2.0
        self.log.append(f"m_sat: {m_sat:.4f}, m_mid: {m_mid:.4f}, m_faint: {m_faint:.4f}")

        # Subdivide sources into bright and faint magnitude bins
        mab = wfi_properties.ab_magnitude(df_cat["psf_flux"])
        cond_bright = (mab > m_sat) & (mab <= m_mid)
        cond_faint = (mab > m_mid) & (mab < m_faint)

        # Define statistics/metrics for each property in each magnitude bin
        self.prop_list = []
        for prop_name, prop_unit, stat_style, outlier_thresholds in photoastro_constants.PROPERTIES:
            for suffix1 in photoastro_constants.SUFFIX1:
                self.prop_list.append(Property(prop_name, stat_style, suffix1=suffix1, outlier_thresholds=outlier_thresholds))


        # Initialize dictionary to hold data for each property in each magnitude bin
        df = {}
        for cond, mag_bin in [(cond_bright, "_bright"), (cond_faint, "_faint")]:
            df[f"ee25_radius{mag_bin}"] = df_cat["ee25_radius"][cond]
            df[f"ee50_radius{mag_bin}"] = df_cat["ee50_radius"][cond]
            df[f"ee75_radius{mag_bin}"] = df_cat["ee75_radius"][cond]

            # Division by zero handled by replacing with NaN
            df[f"flux_ratio_aper02_aper01_custom{mag_bin}"] = df_cat["aper02_flux"][cond] / np.where(
                df_cat["aper01_flux"][cond] == 0, np.nan, df_cat["aper01_flux"][cond]
            )
            df[f"flux_ratio_aper04_aper02_custom{mag_bin}"] = df_cat["aper04_flux"][cond] / np.where(
                df_cat["aper02_flux"][cond] == 0, np.nan, df_cat["aper02_flux"][cond]
            )
            df[f"flux_ratio_aper08_aper04_custom{mag_bin}"] = df_cat["aper08_flux"][cond] / np.where(
                df_cat["aper04_flux"][cond] == 0, np.nan, df_cat["aper04_flux"][cond]
            )

            # PSF flux error
            flux_err_theory = wfi_properties.get_psf_flux_error_theory(optical_filter, df_cat["psf_flux"][cond], t_exp)
            df[f"flux_err_ratio_psf_theory_custom{mag_bin}"] = df_cat["psf_flux_err"][cond] / flux_err_theory

            # Cross-match with Gaia to get angular separation
            if "angsep_gaia_custom" in self.properties:
                cat = Table({"x_psf": df_cat["x_psf"][cond], "y_psf": df_cat["y_psf"][cond]})
                matched_cat = crossmatch_with_gaia(self.asdf_file, cat)
                df[f"angsep_gaia_custom{mag_bin}"] = matched_cat["angsep_gaia"]

        # Compute statistics for each property in each magnitude bin
        for prop in self.prop_list:
            prop.compute(df[f"{prop.prop_name}{prop.suffix1}"])
            for key, card in prop.cards.items():
                self.append_data(card.data_name, card.data_value, card.data_unit)

    def evaluate_metrics(self) -> None:
        """Evaluate each metric against the configured threshold tables."""

        # Load table of metric thresholds with columns (metric_name, min, max)
        optical_filter = self.asdf_file["roman"]["meta"]["instrument"]["optical_element"].strip().lower()
        filename = "metric_thresholds/photometric_metric_thresholds.ecsv"
        metric_thresholds = aws_utils.csv2qtable(self.config["RDMT_SPIRE_LDATA_DIR"], filename)

        # Evaluate each property against the metric thresholds
        for prop in self.prop_list:
            prop.evaluate(metric_thresholds, suffix2="_" + optical_filter)
            for key, card in prop.cards.items():
                self.add_evaluation(card.data_name, card.evaluation_value)
