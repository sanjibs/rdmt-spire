import os

import asdf
import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from ..constants.codes import StatusCodes
from ..constants.source_catalog_constants import (
    SOURCE_CATALOG_PROPERTIES,
)
from ..db_tables.sci_tables import L2ScienceResultsTable
from ..manager import MonitorManager


def test_source_catalog_integration(tmp_path):
    """
    Integration test for SourceCatalogMonitor with MonitorManager and database archiving.
    """
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

    # 3. Create mock parquet catalog data with 4 known point sources
    data = {
        'is_extended': [False, False, False, False],
        'psf_flux': [36307.805, 14454.398, 5754.399, 2290.868],
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

    # 4. Initialize MonitorManager
    monitor_config = {
        "source_catalog": {
            "RDMT_SPIRE_L4_DIR": datadir,
        }
    }
    manager = MonitorManager(af, "source_catalog", monitor_config=monitor_config)

    # 5. Process the monitor
    manager.process()
    assert manager.statusCode == StatusCodes.SUCCESS
    assert len(manager.monitor_objects) == 1
    assert manager.monitor_objects[0].monitor_name == "source_catalog"

    # 6. Initialize in-memory SQLite database session
    engine = create_engine("sqlite:///:memory:")
    session_local = sessionmaker(bind=engine)
    L2ScienceResultsTable.__table__.create(bind=engine)
    session = session_local()

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


    try:
        # 7. Archive the results
        reprocess_number = 0
        manager.archive(session, filename, reprocess_number, L2ScienceResultsTable)

        # 8. Retrieve the stored row and assert values
        row = session.get(L2ScienceResultsTable, (filename, reprocess_number))
        assert row is not None
        for prop in SOURCE_CATALOG_PROPERTIES:
            if prop.endswith("_bright"):
                # Bright bin values are indices [0, 1]
                vals = np.array(source_properties[prop.rstrip("_bright")][:2])
            else:
                # Faint bin values are indices [2, 3]
                vals = np.array(source_properties[prop.rstrip("_faint")][2:])

            percentile_vals = np.percentile(vals, [2.275, 15.86, 50, 84.14, 97.725])
            assert getattr(row, f"{prop}_n_sources") == 2
            assert getattr(row, f"{prop}_median") == pytest.approx(percentile_vals[2], rel=1e-3)
            assert getattr(row, f"{prop}_mean") == pytest.approx(np.mean(vals), rel=1e-3)
            assert getattr(row, f"{prop}_std") == pytest.approx(np.std(vals), rel=1e-3)
            assert getattr(row, f"{prop}_dispersion_p68") == pytest.approx((percentile_vals[3] - percentile_vals[1])/2.0, rel=1e-3)
            assert getattr(row, f"{prop}_dispersion_p95") == pytest.approx((percentile_vals[4] - percentile_vals[0])/4.0, rel=1e-3)
            assert not getattr(row, f"{prop}_n_sources_eval")
            assert getattr(row, f"{prop}_median")
            assert getattr(row, f"{prop}_mean")

        # 9. Verify database verification works
        # Check that comparing the row against itself passes
        assert row._verify(row, return_result=True) is True

    finally:
        session.close()
