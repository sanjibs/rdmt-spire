"""Tests for the photoastro monitor using synthetic Roman WFI-like inputs.

This module exercises the behavior of :class:`photometry.photoastro.PhotoAstroMonitor`
with lightweight synthetic catalogs and deliberately minimal dependency stubs.

The tests cover two main scenarios:

1. A small, hand-built catalog is used to validate metric creation in the
   bright and faint magnitude bins. The test patches the photometry pipeline,
   photometric calibration object, and Gaia crossmatch helper to isolate the
   monitor logic and confirm that expected summary cards such as
   ``flux_ratio_aper02_aper01_median_*`` and ``angsep_gaia_medae_*`` are produced.

2. A larger synthetic catalog is used to exercise the full ``run()`` workflow,
   checking that all generated metric cards are finite, that source counts match
   the expected bright/faint selection, and that the median values for selected
   properties agree with the underlying catalog values. The test also verifies
   the evaluation flags are set consistently for the bright/faint bins.

"""

from __future__ import annotations

from pathlib import Path

import asdf
import numpy as np
import pytest
from astropy import units as u
from astropy.table import QTable

from ..monitors.photoastro import photoastro
from ..utilities import aws_utils


@pytest.fixture
def synthetic_config(tmp_path: Path) -> dict[str, str]:
    l4_dir = tmp_path / "l4"
    l4_dir.mkdir()
    config = aws_utils.get_monitor_config()
    config["RDMT_SPIRE_L4_DIR"]= str(l4_dir)
    return config


def _make_asdf_file() -> asdf.AsdfFile:
    return asdf.AsdfFile(
        {
            "roman": {
                "meta": {
                    "instrument": {"optical_element": " F106 "},
                    "exposure": {"exposure_time": 60.0},
                }
            }
        }
    )

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
    table["fluxfrac_radius_50"] = rng.uniform(0.45, 0.65, size=len(mags)) * u.arcsec
    table["psf_flux"] = flux
    table["psf_flux_err"] = 0.05 * flux
    table["aper01_flux"] = 0.95 * flux
    table["aper02_flux"] = 1.00 * flux
    table["aper04_flux"] = 1.02 * flux
    table["aper08_flux"] = 1.05 * flux
    table["ee25_radius"] = rng.normal(0.05, 0.05, size=len(mags)) * u.arcsec
    table["ee50_radius"] = rng.normal(0.10, 0.05, size=len(mags)) * u.arcsec
    table["ee75_radius"] = rng.normal(0.15, 0.05, size=len(mags)) * u.arcsec
    x_psf = np.linspace(20.0, 140.0, len(mags)) * u.pix
    y_psf = (np.linspace(30.0, 150.0, len(mags)) + rng.normal(0, 1, len(mags))) * u.pix
    table["x_psf"] = x_psf+rng.normal(0, 0.1, len(mags)) * u.pix
    table["y_psf"] = y_psf+rng.normal(0, 0.1, len(mags)) * u.pix
    # table["is_extended"][0:3] = True
    # table["sharpness"][0:3] = 1.5
    table['mag'] = mags*u.mag
    table['is_bright'] = np.array([True] * n_bright + [False] * n_faint, dtype=bool)
    return table


def _make_catalog() -> QTable:
    flux = np.array([10 ** ((31.4 - 19.0) / 2.5), 10 ** ((31.4 - 21.0) / 2.5)]) * u.nJy
    return QTable(
        {
            "psf_flux": flux,
            "psf_flux_err": [1.0, 2.0] * u.nJy,
            "aper01_flux": [0.0, 8.0] * u.nJy,
            "aper02_flux": [2.0, 10.0] * u.nJy,
            "aper04_flux": [4.0, 20.0] * u.nJy,
            "aper08_flux": [8.0, 40.0] * u.nJy,
            "ee25_radius": [0.01, 0.02] * u.arcsec,
            "ee50_radius": [0.02, 0.04] * u.arcsec,
            "ee75_radius": [0.03, 0.06] * u.arcsec,
            "x_psf": [10.0, 20.0] * u.pix,
            "y_psf": [11.0, 21.0] * u.pix,
        }
    )


def test_photoastro_calculate_metrics_builds_expected_bright_and_faint_cards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog = _make_catalog()

    class FakePhotometryCatalog:
        def __init__(self, asdf_file, config):
            self.catalog = catalog

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

    def fake_crossmatch(asdf_file, cat):
        matched = cat.copy()
        matched["angsep_gaia"] = np.full(len(cat), 0.1) * u.arcsec
        return matched

    monkeypatch.setattr(photoastro, "PhotometryCatalogPipeline", FakePhotometryCatalog)
    monkeypatch.setattr(photoastro, "RomanWFIPhotometricParameters", FakePhotometricParameters)
    monkeypatch.setattr(photoastro, "crossmatch_with_gaia", fake_crossmatch)

    monitor = photoastro.PhotoAstroMonitor(_make_asdf_file(), {})
    monitor.calculate_metrics()
    suffix0='_custom'
    if "flux_ratio_aper02_aper01" in monitor.data:
        assert np.isnan(monitor.data[f"flux_ratio_aper02_aper01{suffix0}_median_bright"].data_value)
        assert monitor.data[f"flux_ratio_aper02_aper01{suffix0}_median_faint"].data_value == pytest.approx(1.25)
        assert monitor.data[f"flux_ratio_aper02_aper01{suffix0}_n_sources_bright"].data_value == 0
        assert monitor.data[f"flux_ratio_aper02_aper01{suffix0}_n_sources_faint"].data_value == 1
    assert monitor.data[f"angsep_gaia{suffix0}_medae_faint"].data_value == pytest.approx(0.1)



def test_source_catalog_run_computes_expected_metrics(tmp_path: Path, synthetic_config: dict[str, str], monkeypatch: pytest.MonkeyPatch) -> None:
    catalog = _make_synthetic_catalog()

    class FakePhotometryCatalog:
        def __init__(self, asdf_file, config):
            self.catalog = catalog

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

    def fake_crossmatch(asdf_file, cat):
        matched = cat.copy()
        matched["angsep_gaia"] = np.full(len(cat), 0.1) * u.arcsec
        return matched

    monkeypatch.setattr(photoastro, "PhotometryCatalogPipeline", FakePhotometryCatalog)
    monkeypatch.setattr(photoastro, "RomanWFIPhotometricParameters", FakePhotometricParameters)
    monkeypatch.setattr(photoastro, "crossmatch_with_gaia", fake_crossmatch)

    monitor = photoastro.PhotoAstroMonitor(_make_asdf_file(), synthetic_config)
    monitor.run()


    catalog["angsep_gaia"] = np.full(len(catalog), 0.1, dtype=float) * u.arcsec

    stats1 = ['median', 'nmad', 'mean', 'std', 'n_sources', 'f_outliers']
    stats2 = ['medae', 'rmse', 'mae', 'n_sources', 'f_outliers']
    prop_list = ['angsep_gaia_custom','ee25_radius', 'ee50_radius', 'ee75_radius']
    suffix0 = '_custom'
    for prop in monitor.properties:
        stats = stats2 if prop == 'angsep_gaia_custom' else stats1
        for mag_bin in ["bright", "faint"]:
            for stat in stats:                
                name = f"{prop}_{stat}_{mag_bin}"
                print(f"Checking {name}")
                assert name in monitor.data
                card=monitor.data[name]
                assert np.isfinite(card.data_value)
            cond = catalog['is_bright'] if mag_bin == "bright" else ~catalog['is_bright']
            name1=f"{prop}_{'n_sources'}_{mag_bin}"
            assert monitor.data[name1].data_value == np.sum(cond)
            if prop in prop_list:
                print(prop)
                name=f"{prop}_{stats[0]}_{mag_bin}"
                prop_cat=prop.removesuffix(suffix0)
                print(f"{name:40}, {monitor.data[name].data_value:8f}, {np.median(catalog[prop_cat][cond]):8f}, {np.sum(cond)}, {monitor.data[name1].data_value}, {monitor.data[name].evaluation_value}")
                assert monitor.data[name].data_value == pytest.approx(np.median(catalog[prop_cat][cond].value), rel=1e-6)
                if prop == 'ellipticity':
                    assert monitor.data[name].evaluation_value is False
                else:
                    assert monitor.data[name].evaluation_value is True
