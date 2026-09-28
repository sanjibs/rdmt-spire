import os

import asdf
import numpy as np
import pytest
from astropy import units as u
from astropy.table import QTable
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from ..constants.codes import StatusCodes
from ..db_tables.sci_tables import L2ScienceResultsTable
from ..manager import MonitorManager
from ..monitors.source_catalog import source_catalog
from ..utilities import aws_utils

#pytestmark = pytest.mark.skip(reason="Skipping source catalog integration test")


def test_source_catalog_integration(tmp_path, monkeypatch):
    """
    Integration test for SourceCatalogMonitor with MonitorManager and database archiving.
    """
    # 1. Define temporary data directory and paths
    l4_dir = str(tmp_path)
    filename = "mock_file_cal.asdf"
    parquet_filename = "mock_file_cat.parquet"
    parquet_path = os.path.join(l4_dir, parquet_filename)

    # 2. Build a valid ASDF object referencing our mock filename
    af = asdf.AsdfFile()
    af.tree["roman"] = {
        "meta": {
            "filename": filename,
            "instrument": {"optical_element": "F087"},
            "exposure": {"exposure_time": 294.10974},
        }
    }

    # 3. Create mock parquet catalog data with 4 known point sources
    table = QTable()
    table["is_extended"] = np.array([False, False, False, False])
    table["psf_flux"] = np.array([36307.805, 14454.398, 5754.399, 2290.868])*u.nJy
    table["psf_flux_err"] = np.array([233.1558, 221.6841, 188.6324, 183.4854])*u.nJy
    table["sharpness"] = np.array([0.2, 0.4, 0.1, 0.3])
    table["roundness1"] = np.array([-0.1, 0.3, -0.2, 0.4])
    table["ellipticity"] = np.array([0.1, 0.3, 0.2, 0.4])
    table["fluxfrac_radius_50"] = np.array([0.5, 0.7, 0.4, 0.6])*u.arcsec
    table["aper01_flux"] = np.array([1000.0, 1000.0, 1000.0, 1000.0])*u.nJy
    table["aper02_flux"] = np.array([2000.0, 3000.0, 1500.0, 2500.0])*u.nJy
    table["aper04_flux"] = np.array([3000.0, 6000.0, 2250.0, 5000.0])*u.nJy
    table["aper08_flux"] = np.array([4000.0, 12000.0, 3375.0, 10000.0])*u.nJy
    table["x_psf"] = np.arange(4) * u.pix
    table["y_psf"] = np.arange(4) * u.pix
    table.write(parquet_path, format="parquet")

    def fake_crossmatch(asdf_file, catalog):
        matched = catalog.copy()
        matched["angsep_gaia"] = np.full(len(catalog), 0.1) * u.arcsec
        return matched

    monkeypatch.setattr(source_catalog, "crossmatch_with_gaia", fake_crossmatch)

    # 4. Initialize MonitorManager
    config = aws_utils.get_monitor_config()
    config["RDMT_SPIRE_L4_DIR"]= str(l4_dir)
    monitor_config = {"source_catalog": config}
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

    try:
        # 7. Archive the results
        reprocess_number = 0
        manager.archive(session, filename, reprocess_number, L2ScienceResultsTable)

        # 8. Retrieve the stored row and assert values
        row = session.get(L2ScienceResultsTable, (filename, reprocess_number))
        assert row is not None
        for card in manager.monitor_objects[0].get_data_card("all"):
            assert getattr(row, card.data_name) == pytest.approx(card.data_value)
            assert getattr(row, f"{card.data_name}_eval") == card.evaluation_value

        # 9. Verify database verification works
        # Check that comparing the row against itself passes
        assert row._verify(row, return_result=True) is True

    finally:
        session.close()
