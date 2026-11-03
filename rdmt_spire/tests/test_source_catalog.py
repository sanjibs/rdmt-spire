import os

import asdf
import numpy as np
import pandas as pd
import pytest
from astropy.table import Table

from ..constants.source_catalog_constants import (
    SOURCE_CATALOG_PROPERTIES,
)

# from dotenv import dotenv_values
from ..monitors.source_catalog import SourceCatalogMonitor


def test_source_catalog_monitor(tmp_path):
    """
    Test the SourceCatalogMonitor using a generated catalog parquet file with known values
    to verify that the monitor correctly groups sources, computes the physical limits,
    and recovers the expected metric statistics.
    """
    # Useful for testing real data files
    # config = dotenv_values(".env")
    # datadir = config['TEST_L4_DATADIR']

    # 1. Define temporary data directory and paths
    datadir = str(tmp_path)
    filename = "mock_file_cal.asdf"
    parquet_filename = "mock_file_cat.parquet"
    parquet_path = os.path.join(datadir, parquet_filename)

    # 2. Build a valid ASDF object referencing our mock filename
    af = asdf.AsdfFile()
    af.tree["roman"] = {
        "meta": {
            "filename": filename,
            "instrument": {
                "optical_element": "F087"
            },
            "exposure": {
                "exposure_time": 294.10974
            }
        }
    }

    # 3. Create mock parquet catalog data with 4 known point sources:
    # - 2 sources in the bright bin (magnitudes ~20.0 and ~21.0)
    # - 2 sources in the faint bin (magnitudes ~22.0 and ~23.0)
    #
    # We pre-calculate exact values for the properties so we can assert on statistics:
    #
    # Bright bin (sharpness=[0.2, 0.4], roundness1=[-0.1, 0.3], ellipticity=[0.1, 0.3], fluxfrac_radius_50=[0.5, 0.7])
    # - sharpness: median=0.3, rms = sqrt((0.2^2 + 0.4^2)/2) = sqrt(0.1) ~ 0.3162, nmad = 1.4826 * median(|0.2|, |0.4|) = 0.44478
    # - roundness1: median=0.1, rms = sqrt(0.05) ~ 0.2236, nmad = 1.4826 * 0.2 = 0.29652
    # - ellipticity (expected=1.0): median=0.2, rms = sqrt(((0.1-1.0)^2 + (0.3-1.0)^2)/2) = sqrt((0.81 + 0.49)/2) = sqrt(0.65) ~ 0.8062, nmad = 1.4826 * median(|-0.9|, |-0.7|) = 1.18608
    #
    # Faint bin (sharpness=[0.1, 0.3], roundness1=[-0.2, 0.4])
    # - sharpness: median=0.2, rms = sqrt(0.05) ~ 0.2236, nmad = 1.4826 * 0.2 = 0.29652
    # - roundness1: median=0.1, rms = sqrt(0.1) ~ 0.3162, nmad = 1.4826 * 0.3 = 0.44478

    data = {
        'is_extended': [False, False, False, False],
        # psf_flux values corresponding to magnitudes 20.0, 21.0, 22.0, 23.0
        'psf_flux': [36307.805, 14454.398, 5754.399, 2290.868],
        # psf_flux_err chosen to make ratio psf_flux_err/psf_flux_err_theory equal to [2.0, 3.0, 4.0, 6.0]
        'psf_flux_err': [233.1558, 221.6841, 188.6324, 183.4854],
        'sharpness': [0.2, 0.4, 0.1, 0.3],
        'roundness1': [-0.1, 0.3, -0.2, 0.4],
        'ellipticity': [0.1, 0.3, 0.2, 0.4],
        'fluxfrac_radius_50': [0.5, 0.7, 0.4, 0.6],
        'aper01_flux': [1000.0, 1000.0, 1000.0, 1000.0],
        'aper02_flux': [2000.0, 3000.0, 1500.0, 2500.0],
        'aper04_flux': [3000.0, 6000.0, 2250.0, 5000.0],
        'aper08_flux': [4000.0, 12000.0, 3375.0, 10000.0]
    }
    df = pd.DataFrame(data)
    df.to_parquet(parquet_path)

    # 4. Instantiate and run the monitor
    monitor = SourceCatalogMonitor(af, config={"RDMT_SPIRE_L4_DIR": datadir})
    monitor.run()

    # Define properties to calculate dynamically
    properties = SOURCE_CATALOG_PROPERTIES

    # 5. Load expected values from expected_photometric_properties.ecsv
    expected_path = os.path.join(os.path.dirname(__file__), '..', 'monitors', 'source_catalog', 'data', 'expected_photometric_properties.ecsv')
    expected_props = Table.read(expected_path, format="ascii.ecsv").to_pandas()
    for prop in properties:
        for metric in ['median', 'dispersion_p68']:
            for filter in ['f062', 'f087','f106', 'f129', 'f146', 'f158', 'f184', 'f213']:
                assert f"{prop}_{metric}_{filter}" in expected_props['property_name'].values


    # Pre-calculated values for the 4 sources corresponding to the generated mock data
    source_properties = {
        "sharpness": [0.2, 0.4, 0.1, 0.3],
        "roundness1": [-0.1, 0.3, -0.2, 0.4],
        "ellipticity": [0.1, 0.3, 0.2, 0.4],
        "flux_frac_radius_50": [0.5, 0.7, 0.4, 0.6],
        "flux_ratio_aper01_aper02": [2.0, 3.0, 1.5, 2.5],
        "flux_ratio_aper02_aper04": [1.5, 2.0, 1.5, 2.0],
        "flux_ratio_aper04_aper08": [4.0/3.0, 2.0, 1.5, 2.0],
        "flux_err_ratio_psf_theory": [2.0, 3.0, 4.0, 6.0]
    }


    # Verify each computed property dynamically
    for prop in properties:
        if prop.endswith("_bright"):
            # Bright bin values are indices [0, 1]
            vals = np.array(source_properties[prop.rstrip("_bright")][:2])
        else:
            # Faint bin values are indices [2, 3]
            vals = np.array(source_properties[prop.rstrip("_faint")][2:])

        percentile_vals = np.percentile(vals, [2.275, 15.86, 50, 84.14, 97.725])
        assert monitor.get_data(f"{prop}_n_sources") == 2
        assert monitor.get_data(f"{prop}_median") == pytest.approx(percentile_vals[2], rel=1e-3)
        assert monitor.get_data(f"{prop}_mean") == pytest.approx(np.mean(vals), rel=1e-3)
        assert monitor.get_data(f"{prop}_std") == pytest.approx(np.std(vals), rel=1e-3)
        assert monitor.get_data(f"{prop}_dispersion_p68") == pytest.approx((percentile_vals[3] - percentile_vals[1])/2.0, rel=1e-3)
        assert monitor.get_data(f"{prop}_dispersion_p95") == pytest.approx((percentile_vals[4] - percentile_vals[0])/4.0, rel=1e-3)

    # Check all card evaluations are True
    data_cards = monitor.get_data_card('all')
    for card in data_cards:
        # only n_sources> 10 are set to True, others are False
        if 'n_sources' in card.data_name: 
            assert card.evaluation_value is False
        else:
            assert card.evaluation_value is True
