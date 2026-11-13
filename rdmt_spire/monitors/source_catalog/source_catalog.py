import logging
import os

import asdf
import numpy as np
from astropy import units as u
from astropy.table import QTable

from ...constants import source_catalog_constants
from ...utilities import aws_utils
from ...utilities.phot.photoastro_utils import crossmatch_with_gaia
from ...utilities.property import Property
from ...utilities.techinfo import RomanWFIPhotometricParameters
from ..monitor_base import BaseMonitor

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class SourceCatalogMonitor(BaseMonitor):
    """
    Photometry Monitor derived from class BaseMonitor.
    Reads the source catalog (parquet format) created from an L2 file.
    Bins point sources by brightness and tracks properties:
    sharpness, roundness1, ellipticity, flux_frac_radius_50, flux ratios,
    and flux error ratio.
    """

    def __init__(self, asdf_file: asdf.AsdfFile, config: dict[str, str] | str | os.PathLike):
        super().__init__(asdf_file)
        self.monitor_name = "source_catalog"
        self.config = config
        # Define properties and statistics for the metrics
        self.properties = [prop_name for prop_name, *rest in source_catalog_constants.PROPERTIES]


    def calculate_metrics(self):
        """
        Calculates source catalog metrics binned by magnitude.
        Adds median, STD, and NMAD metrics for 8 properties across bright and faint bins.
        """
        # Parse needed values from the ASDF file.
        filename = self.asdf_file["roman"]["meta"]["filename"]
        optical_filter = self.asdf_file["roman"]["meta"]["instrument"]["optical_element"].strip().lower()
        t_exp = self.asdf_file["roman"]["meta"]["exposure"]["exposure_time"] * u.s

        if self.config.get("L4_ARCHIVE_OBJECT_KEY", "") == "":
            l4_object_key = filename.replace('_cal.asdf', '_cat.parquet')
        else:
            l4_object_key = self.config["L4_ARCHIVE_OBJECT_KEY"]

        # 4. Construct parquet catalog file path and load it
        logger.info(f"SourceCatalogMonitor: Reading {self.config['L4_ARCHIVE_DIR']} {l4_object_key}")
        file_object = aws_utils.load_file_object(self.config["L4_ARCHIVE_DIR"], l4_object_key)
        logger.info("SourceCatalogMonitor: Loading source catalog from parquet file")
        df_cat = QTable.read(file_object, format="parquet")
        logger.info(f"SourceCatalogMonitor: Loaded source catalog for file of length {len(df_cat)}")

        # 5. Point source selection
        if 'is_extended' not in df_cat.colnames:
            raise RuntimeError("SourceCatalogMonitor: 'is_extended' column missing from source catalog")
        df_cat = df_cat[~df_cat["is_extended"]].copy()

        # Calculate bright and faint magnitude limits
        wfi_properties = RomanWFIPhotometricParameters(self.config)
        m_faint = wfi_properties.get_mfaint(optical_filter, t_exp)
        m_sat = wfi_properties.get_msat(optical_filter, t_exp)
        m_mid = (m_sat + m_faint) / 2.0

        logger.info(f"SourceCatalogMonitor: Magnitudes - m_sat: {m_sat:.4f}, m_mid: {m_mid:.4f}, m_faint: {m_faint:.4f}")

        # Subdivide sources into bright and faint magnitude bins
        mab = wfi_properties.ab_magnitude(df_cat["psf_flux"])
        cond_bright = (mab > m_sat) & (mab <= m_mid)
        cond_faint = (mab > m_mid) & (mab < m_faint)

        self.prop_list = []
        for prop_name, prop_unit, stat_style, outlier_thresholds in source_catalog_constants.PROPERTIES:
            for suffix1 in source_catalog_constants.SUFFIX1:
                self.prop_list.append(Property(prop_name, stat_style, suffix1=suffix1, outlier_thresholds=outlier_thresholds))

        # Initialize dictionary to hold data for each property in each magnitude bin
        df = {}
        for cond, mag_bin in [(cond_bright, "_bright"), (cond_faint, "_faint")]:
            df[f"sharpness{mag_bin}"] = df_cat["sharpness"][cond]
            df[f"roundness1{mag_bin}"] = df_cat["roundness1"][cond]
            df[f"ellipticity{mag_bin}"] = df_cat["ellipticity"][cond]
            df[f"fluxfrac_radius_50{mag_bin}"] = df_cat["fluxfrac_radius_50"][cond]

            # Division by zero handled by replacing with NaN
            temp = np.where(df_cat["aper01_flux"][cond] == 0, np.nan, df_cat["aper01_flux"][cond])
            df[f"flux_ratio_aper02_aper01{mag_bin}"] = df_cat["aper02_flux"][cond] / temp
            temp = np.where(df_cat["aper02_flux"][cond] == 0, np.nan, df_cat["aper02_flux"][cond])
            df[f"flux_ratio_aper04_aper02{mag_bin}"] = df_cat["aper04_flux"][cond] / temp
            temp = np.where(df_cat["aper04_flux"][cond] == 0, np.nan, df_cat["aper04_flux"][cond])
            df[f"flux_ratio_aper08_aper04{mag_bin}"] = df_cat["aper08_flux"][cond] / temp

            # PSF flux error
            flux_err_theory = wfi_properties.get_psf_flux_error_theory(optical_filter, df_cat["psf_flux"][cond], t_exp)
            df[f"flux_err_ratio_psf_theory{mag_bin}"] = df_cat["psf_flux_err"][cond] / flux_err_theory

            # Cross-match with Gaia to get angular separation
            if "angsep_gaia" in self.properties:
                selected_rows = np.flatnonzero(cond)
                invalid_pixels = ~np.isfinite(df_cat["x_psf"][cond].value) | ~np.isfinite(df_cat["y_psf"][cond].value)
    
                if np.any(invalid_pixels):
                    # These indices refer to df_cat after the is_extended filter.
                    bad_rows = selected_rows[invalid_pixels][:10]
                    diagnostic_columns = [
                        name for name in ("id", "label", "x_psf", "y_psf", "psf_flags", "psf_flux", "psf_flux_err")
                        if name in df_cat.colnames
                    ]
                    logger.warning(
                        "SourceCatalogMonitor: bin=%s first 10 invalid position rows "
                        "(point-source catalog indices=%s):\n%s",
                        mag_bin, bad_rows.tolist(), df_cat[diagnostic_columns][bad_rows],
                    )
                    cond[invalid_pixels] = False # Mark invalid pixel positions as False in the selection condition
                cat = QTable({"x_psf": df_cat["x_psf"][cond], "y_psf": df_cat["y_psf"][cond]})
                matched_cat = crossmatch_with_gaia(self.asdf_file, cat)
                df[f"angsep_gaia{mag_bin}"] = matched_cat["angsep_gaia"]

        # Compute statistics for each property in each magnitude bin
        for prop in self.prop_list:
            logger.info(f"SourceCatalogMonitor: computing property: {prop}")
            prop.compute(df[f"{prop.prop_name}{prop.suffix1}"])
            
            logger.info(f"SourceCatalogMonitor: appending datacards for {prop}")
            for key, card in prop.cards.items():
                self.append_data(card.data_name, card.data_value, card.data_unit)

    def evaluate_metrics(self) -> None:
        """Evaluate each metric against the configured threshold tables."""

        # Load table of metric thresholds with columns (metric_name, min, max)
        optical_filter = self.asdf_file["roman"]["meta"]["instrument"]["optical_element"].strip().lower()
        filename = "metric_thresholds/source_catalog_metric_thresholds.ecsv"
        metric_thresholds = aws_utils.csv2qtable(self.config["BUNDLED_DATA_DIR"], filename)

        # Evaluate each property against the metric thresholds
        for prop in self.prop_list:
            prop.evaluate(metric_thresholds, suffix2="_" + optical_filter)
            for key, card in prop.cards.items():
                self.add_evaluation(card.data_name, card.evaluation_value)
