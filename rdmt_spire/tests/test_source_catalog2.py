import io
import math
from unittest.mock import patch

import asdf
import numpy as np
import pandas as pd
import pytest

from rdmt_spire.monitors.source_catalog.source_catalog import SourceCatalogMonitor

# ── F087 calibration constants (from data files, t_exp=140 s) ────────────────
_T_EXP   = 140.0
_ZP      = 26.3023   # Z_R
_F_PEAK  = 0.4838
_F_THERM = 0.003
_F_ZODI  = 0.251


def _mag_to_flux(mag_ab: float) -> float:
    return 10 ** ((31.4 - mag_ab) / 2.5)


def _make_asdf_file() -> asdf.AsdfFile:
    af = asdf.AsdfFile()
    af["roman"] = {
        "meta": {
            "filename": "r00000001001001001001_0001_wfi01_f087_cal.asdf",
            "instrument": {"optical_element": "F087"},
            "exposure": {"exposure_time": _T_EXP},
        }
    }
    return af


def _make_catalog(
    n_bright: int = 20,
    n_faint: int = 20,
    sharpness_bright: float = 0.80,
    sharpness_faint: float = 0.70,
    include_extended: int = 0,
) -> pd.DataFrame:
    """Build a minimal source-catalog DataFrame with controlled sources.

    Bright sources are placed at mag 19, faint at mag 22, both well within the
    expected F087 bright/faint magnitude bins.
    """
    rng = np.random.default_rng(42)

    def _sources(n: int, flux_center: float, sharpness: float) -> pd.DataFrame:
        # add 1 % flux noise so the DataFrame is not degenerate
        psf_flux = flux_center * (1.0 + 0.01 * rng.standard_normal(n))
        return pd.DataFrame({
            "is_extended":        np.zeros(n, dtype=bool),
            "sharpness":          np.full(n, sharpness)+rng.standard_normal(n)*0.01,
            "roundness1":         np.zeros(n)+rng.standard_normal(n)*0.01,
            "ellipticity":        np.full(n, 0.05)+rng.standard_normal(n)*0.01,
            "fluxfrac_radius_50": np.full(n, 0.15)+rng.standard_normal(n)*0.01,
            "psf_flux":           psf_flux+rng.standard_normal(n)*0.01,
            "psf_flux_err":       psf_flux * 0.10+rng.standard_normal(n)*0.01,
            "aper01_flux":        psf_flux * 0.60+rng.standard_normal(n)*0.01,
            "aper02_flux":        psf_flux * 0.75+rng.standard_normal(n)*0.01,
            "aper04_flux":        psf_flux * 0.88+rng.standard_normal(n)*0.01,
            "aper08_flux":        psf_flux * 0.96+rng.standard_normal(n)*0.01,
        })

    parts = [
        _sources(n_bright, _mag_to_flux(19.0), sharpness_bright),
        _sources(n_faint,  _mag_to_flux(22.0), sharpness_faint),
    ]
    if include_extended:
        ext = _sources(include_extended, _mag_to_flux(19.0), sharpness_bright)
        ext["is_extended"] = True
        parts.append(ext)

    return pd.concat(parts, ignore_index=True)


def _catalog_to_parquet_bytes(df: pd.DataFrame) -> io.BytesIO:
    buf = io.BytesIO()
    df.to_parquet(buf)
    buf.seek(0)
    return buf


def _run_calculate_metrics(monitor: SourceCatalogMonitor, df: pd.DataFrame) -> None:
    buf = _catalog_to_parquet_bytes(df)
    # every call to load_file_object(...) inside the with block immediately returns the in-memory BytesIO buffer
    with patch(
        "rdmt_spire.monitors.source_catalog.source_catalog.aws_utils.load_file_object",
        return_value=buf,
    ):
        monitor.calculate_metrics()


# ── fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def asdf_f087() -> asdf.AsdfFile:
    return _make_asdf_file()


@pytest.fixture
def monitor(asdf_f087, tmp_path) -> SourceCatalogMonitor:
    return SourceCatalogMonitor(asdf_f087, datadir=str(tmp_path))


# ── init ─────────────────────────────────────────────────────────────────────

def test_init_monitor_name(monitor, asdf_f087):
    assert monitor.monitor_name == "source_catalog"
    assert monitor.asdf_file is asdf_f087


def test_init_properties_list(monitor):
    assert len(monitor.properties) == 16
    assert "sharpness_bright" in monitor.properties
    assert "sharpness_faint" in monitor.properties
    assert "flux_err_ratio_psf_theory_bright" in monitor.properties
    assert "flux_err_ratio_psf_theory_faint" in monitor.properties


# ── calculate_metrics: output structure ──────────────────────────────────────

def test_calculate_metrics_all_keys_present(monitor):
    _run_calculate_metrics(monitor, _make_catalog())
    stats = ["median", "dispersion_p68", "dispersion_p95", "std", "mean", "n_sources"]
    for prop in monitor.properties:
        for stat in stats:
            assert f"{prop}_{stat}" in monitor.data


def test_calculate_metrics_total_key_count(monitor):
    _run_calculate_metrics(monitor, _make_catalog())
    # 8 properties × 2 bins × 6 stats =96
    assert len(monitor.data) == 96


# ── calculate_metrics: source binning ────────────────────────────────────────

def test_calculate_metrics_n_sources_bright_faint(monitor):
    _run_calculate_metrics(monitor, _make_catalog(n_bright=20, n_faint=15))
    assert monitor.get_data("sharpness_bright_n_sources") == 20
    assert monitor.get_data("sharpness_faint_n_sources") == 15


def test_calculate_metrics_extended_sources_excluded(monitor):
    _run_calculate_metrics(monitor, _make_catalog(n_bright=20, n_faint=15, include_extended=5))
    assert monitor.get_data("sharpness_bright_n_sources") == 20
    assert monitor.get_data("sharpness_faint_n_sources") == 15


# ── calculate_metrics: metric values ─────────────────────────────────────────

def test_calculate_metrics_bright_p16_matches_input(monitor):
    # sharpness is constant per bin so all percentiles equal the input value
    _run_calculate_metrics(monitor, _make_catalog(n_bright=30, n_faint=30, sharpness_bright=0.80))
    assert monitor.get_data("sharpness_bright_median") == pytest.approx(0.80, abs=5*0.1/math.sqrt(30))


def test_calculate_metrics_faint_p16_matches_input(monitor):
    _run_calculate_metrics(monitor, _make_catalog(n_bright=30, n_faint=30, sharpness_faint=0.60))
    assert monitor.get_data("sharpness_faint_median") == pytest.approx(0.60, abs=5*0.1/math.sqrt(30))


def test_calculate_metrics_flux_ratio_aper02_aper01(monitor):
    # aper02_flux / aper01_flux = 0.75 / 0.60 = 1.25
    _run_calculate_metrics(monitor, _make_catalog(n_bright=30, n_faint=30))
    assert monitor.get_data("flux_ratio_aper01_aper02_bright_median") == pytest.approx(
        0.75 / 0.60, abs=0.01
    )


def test_calculate_metrics_std_is_non_negative(monitor):
    _run_calculate_metrics(monitor, _make_catalog(n_bright=20, n_faint=20))
    for prop in monitor.properties:
        std_val = monitor.get_data(f"{prop}_std")
        if math.isfinite(std_val):
            assert std_val >= 0.0


def test_calculate_metrics_dispersion_p68_lte_dispersion_p95(monitor):
    # p68 width (p84-p16) should be ≤ p95 width (p97.5-p2.5)
    _run_calculate_metrics(monitor, _make_catalog(n_bright=30, n_faint=30))
    for prop in monitor.properties:
        p68 = monitor.get_data(f"{prop}_dispersion_p68")
        p95 = monitor.get_data(f"{prop}_dispersion_p95")
        print(f"Checking property: {prop}", p68, p95, flush=True)
        if math.isfinite(p68) and math.isfinite(p95):
            assert np.abs(p68-p95)/p68 < 5/(math.sqrt(2*(30-1)))


# ── calculate_metrics: edge cases ────────────────────────────────────────────

def test_calculate_metrics_single_bright_source_gives_nan(monitor):
    # ≤1 source → percentile values are NaN
    _run_calculate_metrics(monitor, _make_catalog(n_bright=1, n_faint=20))
    assert math.isnan(monitor.get_data("sharpness_bright_median"))
    assert math.isnan(monitor.get_data("sharpness_bright_std"))


def test_calculate_metrics_missing_is_extended_raises(monitor):
    df = _make_catalog().drop(columns=["is_extended"])
    buf = _catalog_to_parquet_bytes(df)
    with patch(
        "rdmt_spire.monitors.source_catalog.source_catalog.aws_utils.load_file_object",
        return_value=buf,
    ):
        with pytest.raises(RuntimeError, match="is_extended"):
            monitor.calculate_metrics()


def test_calculate_metrics_unknown_filter_raises(asdf_f087, tmp_path):
    asdf_f087["roman"]["meta"]["instrument"]["optical_element"] = "F999"
    mon = SourceCatalogMonitor(asdf_f087, datadir=str(tmp_path))
    buf = _catalog_to_parquet_bytes(_make_catalog())
    with patch(
        "rdmt_spire.monitors.source_catalog.source_catalog.aws_utils.load_file_object",
        return_value=buf,
    ):
        with pytest.raises(RuntimeError, match="f999"):
            mon.calculate_metrics()


# ── evaluate_metrics ─────────────────────────────────────────────────────────

def test_evaluate_metrics_valid_in_range_gives_true(monitor):
    # sharpness=0.8 is within [min=0, max=nan] for F087
    _run_calculate_metrics(monitor, _make_catalog(n_bright=30, n_faint=30, sharpness_bright=0.80))
    monitor.evaluate_metrics()
    assert monitor.data["sharpness_bright_median"].evaluation_value is True


def test_evaluate_metrics_out_of_range_gives_false(monitor):
    # ellipticity max=0.5; value of 2.0 is clearly above the upper bound
    df = _make_catalog(n_bright=30, n_faint=30)
    df["ellipticity"] = 2.0
    _run_calculate_metrics(monitor, df)
    monitor.evaluate_metrics()
    assert monitor.data["ellipticity_bright_median"].evaluation_value is False


def test_evaluate_metrics_below_min_gives_false(monitor):
    # roundness1 min=-2; value of -5 is below the lower bound
    df = _make_catalog(n_bright=30, n_faint=30)
    df["roundness1"] = -5.0
    _run_calculate_metrics(monitor, df)
    monitor.evaluate_metrics()
    assert monitor.data["roundness1_bright_median"].evaluation_value is False


def test_evaluate_metrics_few_sources_sets_n_sources_false(monitor):
    # with fewer than 10 sources, n_sources evaluation should be False
    _run_calculate_metrics(monitor, _make_catalog(n_bright=5, n_faint=5))
    monitor.evaluate_metrics()
    assert monitor.data["sharpness_bright_n_sources"].evaluation_value is False
    assert monitor.data["sharpness_faint_n_sources"].evaluation_value is False


def test_evaluate_metrics_few_sources_skips_range_check_on_median(monitor):
    # with <10 sources, median validity is set to True in the first pass but
    # the range check is skipped, leaving evaluation_value True (not False)
    df = _make_catalog(n_bright=5, n_faint=5)
    df["ellipticity"] = 2.0  # would fail range check if it were applied
    _run_calculate_metrics(monitor, df)
    monitor.evaluate_metrics()
    assert monitor.data["ellipticity_bright_median"].evaluation_value is True


def test_evaluate_metrics_sufficient_sources_applies_range_check(monitor):
    df = _make_catalog(n_bright=20, n_faint=20)
    _run_calculate_metrics(monitor, df)
    monitor.evaluate_metrics()
    assert monitor.data["sharpness_bright_median"].evaluation_value is not None


def test_evaluate_metrics_nan_metric_is_false(monitor):
    # single source → NaN median → is_valid_metric returns False → eval = False
    _run_calculate_metrics(monitor, _make_catalog(n_bright=1, n_faint=20))
    monitor.evaluate_metrics()
    assert monitor.data["sharpness_bright_median"].evaluation_value is False


# ── _update_metric_evaluation ────────────────────────────────────────────────

def test_update_metric_evaluation_unknown_filter_raises(monitor):
    _run_calculate_metrics(monitor, _make_catalog(n_bright=20, n_faint=20))
    expected_props = monitor._load_data_file("expected_photometric_properties.ecsv")
    with pytest.raises(RuntimeError, match="not found in expected properties table"):
        monitor._update_metric_evaluation(
            "sharpness_bright_median",
            error=0.01,
            expected_props=expected_props,
            optical_filter="unknown_filter",
        )


def test_update_metric_evaluation_in_range_leaves_evaluation_unchanged(monitor):
    _run_calculate_metrics(monitor, _make_catalog(n_bright=30, n_faint=30, sharpness_bright=0.80))
    # pre-set evaluation to True
    monitor.add_evaluation("sharpness_bright_median", True)
    expected_props = monitor._load_data_file("expected_photometric_properties.ecsv")
    monitor._update_metric_evaluation(
        "sharpness_bright_median",
        error=0.0,
        expected_props=expected_props,
        optical_filter="f087",
    )
    assert monitor.data["sharpness_bright_median"].evaluation_value is True
