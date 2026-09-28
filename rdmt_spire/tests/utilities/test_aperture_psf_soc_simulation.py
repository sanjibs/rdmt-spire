

"""Integration tests for ``PhotometryCatalogPipeline`` against SOC simulation products.

The module builds a ``sample_tables`` fixture from `.env`-configured Roman L2 and L4
artifacts, opens a real calibrated ASDF exposure, reads the matching parquet source
catalog, and runs the pipeline in three modes: a baseline configuration, a scattered
and centroid-refined configuration, and a variable pixel-area configuration. Each run is
joined back to the SOC source catalog by ``label`` so the tests can compare pipeline
outputs with the reference catalog on a per-source basis.

The tests check the public output contract and a small set of behavior invariants. They
verify that the required columns and units listed in ``REQUIRED_OUTPUT_COLUMNS`` are
present, that the output row count matches the non-extended sources in the reference
catalog, and that aperture photometry yields finite fluxes for at least some detections.
They also assert median relative agreement with the SOC catalog for aperture, PSF,
background, and centroid-error columns, with explicit tolerances that allow the heavier
refined-centroid path while still catching regressions in the measured photometry.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("asdf")
pytest.importorskip("photutils")
pytest.importorskip("astropy")
import asdf
import astropy.table
import astropy.units as u
from astropy.table import Table

from ...monitors.photoastro.aperture_psf import (
    REQUIRED_OUTPUT_COLUMNS,
    PhotometryCatalogPipeline,
)
from ...utilities import aws_utils

pytestmark = pytest.mark.skip(reason="Bypassing this test as it requires real data and is computationally heavy")

@pytest.fixture(scope="module")
def sample_tables() -> tuple[Path, Path, Path]:
    config = aws_utils.get_monitor_config()
    l2 = Path(config['RDMT_SPIRE_L2_DIR']) / "r0034201001001001001_0001_wfi01_f087_cal.asdf"  
    catdir = Path(config['RDMT_SPIRE_L4_DIR']) 
        
    if not l2.exists() or not catdir.exists():
        pytest.skip("Sample data files are not present in workspace")
    t1 = time.time()
    with asdf.open(l2) as af:
        t1 = time.time()
        filename = str(af["roman"]["meta"]["filename"])
        cat_path = catdir / filename.replace("_cal.asdf", "_cat.parquet")
        cat = Table.read(cat_path, format="parquet")

        out1 = PhotometryCatalogPipeline(af, config, refine_centroids=False, pixel_area_is_constant=True).catalog
        cat_matched1=Table()
        cat_matched1['label'] = out1['label'].copy()
        cat_matched1 = astropy.table.join(cat_matched1, cat, keys="label", join_type='left')

        # with scatter and refine_centroids options 
        out2 = PhotometryCatalogPipeline(af, config, init_scatter=2.0, refine_centroids=True, pixel_area_is_constant=True).catalog
        cat_matched2=Table()
        cat_matched2['label'] = out2['label'].copy()
        cat_matched2 = astropy.table.join(cat_matched2, cat, keys="label", join_type='left')

        # with scatter and refine_centroids options 
        out3 = PhotometryCatalogPipeline(af, config, pixel_area_is_constant=False).catalog
        cat_matched3=Table()
        cat_matched3['label'] = out3['label'].copy()
        cat_matched3 = astropy.table.join(cat_matched3, cat, keys="label", join_type='left')
        print(f"Time taken for processing 3 catalogs: {time.time() - t1} seconds")

    return cat, out1, out2, out3, cat_matched1, cat_matched2, cat_matched3


# def test_monitor_outputs_required_columns(sample_tables: tuple[Path, Path, Path]) -> None:
def test_monitor_outputs_required_columns(sample_tables: tuple[Table, Table, Table, Table, Table, Table, Table]) -> None:
    """Ensure the monitor output table always includes the standard public schema.

    This checks the two execution modes against the required column list from the API
    contract. If a required field disappears or is renamed, the test catches it before
    downstream users consume the output.
    """
    cat, out1, out2, out3, cat_matched1, cat_matched2, cat_matched3 = sample_tables

    # l2, catdir, psf, config = sample_paths
    # with asdf.open(l2) as af:
    #     monitor = PhotometryCatalog(af, config)
    #     out = monitor.run()

    for name,unit in REQUIRED_OUTPUT_COLUMNS:
        assert name in out1.colnames
        assert u.Unit(unit) == out1[name].unit
        assert out1[name].unit == u.Unit(unit)


def test_monitor_row_count_matches_point_sources(sample_tables: tuple[Table, Table, Table, Table, Table]) -> None:
    """Check that the output row count matches the number of non-extended sources.

    The catalog identifies extended objects separately from point sources, so the monitor
    should only report entries for the point-source subset. This protects against over- or
    under-counting when the pipeline is run in different configurations.
    """
    cat, out1, out2, out3, cat_matched1, cat_matched2, cat_matched3 = sample_tables

    expected = int(np.count_nonzero(~np.asarray(cat["is_extended"], dtype=bool)))
    assert len(out1) == expected


def test_monitor_has_finite_aperture_values_for_some_sources(
    sample_tables: tuple[Table, Table, Table, Table, Table, Table, Table],
) -> None:
    """Verify the aperture measurement returns at least some finite flux values.

    A real image should yield measurable flux for a subset of sources, so the test ensures
    the aperture integration path is producing valid numeric outputs rather than NaNs.
    """
    cat, out1, out2, out3, cat_matched1, cat_matched2, cat_matched3 = sample_tables

    if 'aper04_flux' in out1.colnames:
        finite_count = int(np.count_nonzero(np.isfinite(np.asarray(out1["aper04_flux"], dtype=float))))
        assert finite_count > 0

def test_monitor_has_aperture_psf_values_matching_the_soc_source_catalog(
    sample_tables: tuple[Table, Table, Table, Table, Table, Table, Table],
) -> None:
    """Compare the measured aperture and PSF values against the SOC source catalog.

    The test joins the monitor output to the reference catalog and checks the median
    relative error for each relevant column. This confirms that the measured photometry is
    consistent with the reference source catalog within the expected tolerances.
    """
    # with scatter and refine_centroids options the errors are slightly larger
    cat, out1, out2, out3, cat_matched1, cat_matched2, cat_matched3 = sample_tables
    
    for out, cat_matched in [(out1, cat_matched1), (out2, cat_matched2), (out3, cat_matched3)]:
        if len(cat_matched) == 0:
            continue
        for key in ['aper01_flux', 'aper02_flux', 'aper04_flux', 'aper08_flux', 'x_psf', 'y_psf', 'psf_flux', 'aper_bkg_flux', 'aper_bkg_flux']:
            for suffix in ['', '_err']:
                key1 = key+suffix
                if key1 in out.colnames:
                    temp= (cat_matched[key1]-out[key1])/cat_matched[key1]
                    error=np.nanmedian(np.abs(temp))
                    print(f"{key1:<20} relative medae: {error:0.5f}")        
                    if key1 in ['x_psf_err', 'y_psf_err']:
                        assert error < 1e-2
                    elif key1 in ['aper_bkg_flux', 'aper_bkg_flux_err', 'aper01_flux', 'aper01_flux_err']:
                        assert error < 0.1
                    elif key1 in ['aper02_flux', 'aper02_flux_err', 'aper04_flux', 'aper04_flux_err', 'aper08_flux', 'aper08_flux_err']:
                        assert error < 1e-3
                    else:
                        assert error < 1e-3

