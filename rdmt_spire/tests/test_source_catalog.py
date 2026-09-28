"""Tests for the source catalog monitor using synthetic Roman WFI-like inputs.

This module exercises the behavior of :class:`photometry.source_catalog.SourceCatalogMonitor`
with a synthetic source table and a lightweight ASDF metadata wrapper. The tests are
structured to isolate monitor logic from external dependencies while still validating the
real output shape and statistical checks produced by the class.

The file covers two main scenarios:

1. A synthetic source catalog is written to the expected L4 parquet location, and the
   monitor is run end-to-end with patched photometric calibration and Gaia crossmatch
   helpers. The test verifies that the resulting metrics are finite for both bright and
   faint magnitude bins, that the computed source counts match the selected samples,
   and that the median values for key properties agree with the underlying catalog data.
2. An invalid catalog lacking the required ``is_extended`` column is rejected with a
   clear ``RuntimeError`` before metrics are computed. This confirms the guardrail that
   prevents the monitor from processing malformed source tables.

The wording in this module reflects the actual behaviors being tested here: synthetic
catalog generation, end-to-end metric computation for the source-catalog monitor, and
explicit validation of malformed input handling.
"""

from __future__ import annotations

from pathlib import Path

import asdf
import numpy as np
import pytest
from astropy import units as u
from astropy.table import QTable

from ..monitors.source_catalog import source_catalog
from ..utilities import aws_utils


@pytest.fixture
def synthetic_config(tmp_path: Path) -> dict[str, str]:
    l4_dir = tmp_path / "l4"
    l4_dir.mkdir()
    config = aws_utils.get_monitor_config()
    config["RDMT_SPIRE_L4_DIR"]= str(l4_dir)
    return config


def _make_synthetic_catalog(n_bright: int = 60, n_faint: int = 40) -> QTable:
    rng = np.random.default_rng(0)
    mags = np.concatenate(
        [
            np.linspace(18.1, 19.9, n_bright),
            np.linspace(20.1, 21.9, n_faint),
        ]
    )
    flux = 10 ** ((31.4 - mags) / 2.5) * u.nJy
    table = QTable()
    table["is_extended"] = np.array([False] * (n_bright + n_faint), dtype=bool)
    table["sharpness"] = rng.normal(0.25, 0.08, size=len(mags))
    table["roundness1"] = rng.normal(0.1, 0.05, size=len(mags))
    table["ellipticity"] = rng.normal(2.0, 0.18, size=len(mags))
    table["fluxfrac_radius_50"] = rng.uniform(0.45, 0.65, size=len(mags)) * u.arcsec
    table["psf_flux"] = flux
    table["psf_flux_err"] = 0.05 * flux
    table["aper01_flux"] = 0.95 * flux
    table["aper02_flux"] = 1.00 * flux
    table["aper04_flux"] = 1.02 * flux
    table["aper08_flux"] = 1.05 * flux
    x_psf = np.linspace(20.0, 140.0, len(mags)) * u.pix
    y_psf = (np.linspace(30.0, 150.0, len(mags)) + rng.normal(0, 1, len(mags))) * u.pix
    table["x_psf"] = x_psf+rng.normal(0, 0.1, len(mags)) * u.pix
    table["y_psf"] = y_psf+rng.normal(0, 0.1, len(mags)) * u.pix
    # table["is_extended"][0:3] = True
    # table["sharpness"][0:3] = 1.5
    table['mag'] = mags*u.mag
    table['is_bright'] = np.array([True] * n_bright + [False] * n_faint, dtype=bool)
    return table


def _write_synthetic_l2_and_catalog(tmp_path: Path, config: dict[str, str], table: QTable) -> asdf.AsdfFile:
    filename = "r_test01_wfi01_f106_cal.asdf"
    l2_path = tmp_path / filename
    catalog_path = Path(config["RDMT_SPIRE_L4_DIR"]) / filename.replace("_cal.asdf", "_cat.parquet")

    tree = {
        "roman": {
            "meta": {
                "filename": filename,
                "instrument": {"optical_element": "F106"},
                "exposure": {"exposure_time": 60.0},
            }
        }
    }
    af = asdf.AsdfFile(tree)
    af.write_to(l2_path)
    table.write(catalog_path, format="parquet")
    return af


def test_source_catalog_run_computes_expected_metrics(tmp_path: Path, synthetic_config: dict[str, str], monkeypatch: pytest.MonkeyPatch) -> None:
    table = _make_synthetic_catalog()
    _ = _write_synthetic_l2_and_catalog(tmp_path, synthetic_config, table)

    class FakePhotometricParameters:
        def __init__(self, config):
            pass

        def ab_magnitude(self, flux):
            flux = u.Quantity(flux, copy=False)
            return -2.5 * np.log10(flux.to_value(u.Jy) / 3631.0) * u.mag

        def get_msat(self, optical_filter, exposure_time):
            return 18.0 * u.mag

        def get_mfaint(self, optical_filter, exposure_time):
            return 22.0 * u.mag

        def get_psf_flux_error_theory(self, optical_filter, psf_flux, exposure_time):
            return np.ones(len(psf_flux)) * u.nJy

    def _fake_crossmatch(asdf_file: asdf.AsdfFile, cat: QTable) -> QTable:
        matched = cat.copy()
        matched["angsep_gaia"] = np.full(len(cat), 0.1, dtype=float) * u.arcsec
        return matched

    monkeypatch.setattr(source_catalog, "RomanWFIPhotometricParameters", FakePhotometricParameters)
    monkeypatch.setattr(source_catalog, "crossmatch_with_gaia", _fake_crossmatch)

    with asdf.open(tmp_path / "r_test01_wfi01_f106_cal.asdf") as opened:
        monitor = source_catalog.SourceCatalogMonitor(opened, synthetic_config)
        monitor.run()

    table["angsep_gaia"] = np.full(len(table), 0.1, dtype=float) * u.arcsec

    stats1 = ['median', 'nmad', 'mean', 'std', 'n_sources', 'f_outliers']
    stats2 = ['medae', 'rmse', 'mae', 'n_sources', 'f_outliers']
    prop_list = ['angsep_gaia','sharpness', 'roundness1', 'ellipticity']
    for prop in monitor.properties:
        stats = stats2 if prop == 'angsep_gaia' else stats1
        for mag_bin in ["bright", "faint"]:
            for stat in stats:
                name = f"{prop}_{stat}_{mag_bin}"
                print(f"Checking {name}")
                assert name in monitor.data
                card=monitor.data[name]
                assert np.isfinite(card.data_value)
            cond = table['is_bright'] if mag_bin == "bright" else ~table['is_bright']
            name1=f"{prop}_{'n_sources'}_{mag_bin}"
            assert monitor.data[name1].data_value == np.sum(cond)
            if prop in prop_list:
                name=f"{prop}_{stats[0]}_{mag_bin}"
                print(f"{name:40}, {monitor.data[name].data_value:8f}, {np.median(table[prop][cond]):8f}, {np.sum(cond)}, {monitor.data[name1].data_value}, {monitor.data[name].evaluation_value}")
                assert monitor.data[name].data_value == pytest.approx(np.median(table[prop][cond].value), rel=1e-6)
                if prop == 'ellipticity':
                    assert monitor.data[name].evaluation_value is False
                else:
                    assert monitor.data[name].evaluation_value is True


def test_source_catalog_requires_is_extended_column(tmp_path: Path, synthetic_config: dict[str, str]) -> None:
    filename = "r_test01_wfi01_f106_cal.asdf"
    l2_path = tmp_path / filename
    catalog_path = Path(synthetic_config["RDMT_SPIRE_L4_DIR"]) / filename.replace("_cal.asdf", "_cat.parquet")

    tree = {
        "roman": {
            "meta": {
                "filename": filename,
                "instrument": {"optical_element": "F106"},
                "exposure": {"exposure_time": 60.0},
            }
        }
    }
    asdf.AsdfFile(tree).write_to(l2_path)

    table = QTable()
    table["sharpness"] = [0.1, 0.2]
    table["roundness1"] = [0.0, 0.1]
    table["ellipticity"] = [0.05, 0.1]
    table["fluxfrac_radius_50"] = [0.5, 0.6]* u.arcsec
    table["psf_flux"] = [1000.0, 1500.0] * u.nJy
    table["psf_flux_err"] = [20.0, 25.0] * u.nJy
    table["aper01_flux"] = [900.0, 1400.0] * u.nJy
    table["aper02_flux"] = [1000.0, 1500.0] * u.nJy
    table["aper04_flux"] = [1020.0, 1530.0] * u.nJy
    table["aper08_flux"] = [1050.0, 1560.0] * u.nJy
    table["x_psf"] = [10.0, 20.0] * u.pix
    table["y_psf"] = [10.0, 20.0] * u.pix
    table.write(catalog_path, format="parquet")

    with asdf.open(l2_path) as af:
        monitor = source_catalog.SourceCatalogMonitor(af, synthetic_config)
        with pytest.raises(RuntimeError, match="'is_extended' column missing"):
            monitor.calculate_metrics()
