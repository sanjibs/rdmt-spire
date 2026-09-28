"""Unit tests for ``RomanWFIPhotometricParameters`` lookups and photometric models.

The tests use a dotenv-backed ``photometric_params`` fixture to load the Roman WFI
technical-information tables, then select the ``f106`` filter and construct synthetic
exposure times, count limits, signal-to-noise values, background rates, and scalar or
array PSF fluxes. No monkeypatching or minimal stubs are used; expected results are
computed directly from the analytical equations exercised by the implementation.

The coverage checks filter-specific zero-point, effective-pixel, peak-flux, thermal,
and zodiacal lookups; saturation and faint-limit magnitudes; zero-background and
background-dominated faint-limit regimes; and scalar and vector PSF flux-error theory.
Assertions verify lookup identity, agreement with closed-form values, correct magnitude
and flux units, and elementwise numerical agreement for array inputs.
"""

from __future__ import annotations

import numpy as np
import pytest
from astropy import units as u

from ...utilities import aws_utils
from ...utilities.techinfo import RomanWFIPhotometricParameters


@pytest.fixture
def photometric_params() -> RomanWFIPhotometricParameters:
    config = aws_utils.get_monitor_config()
    return RomanWFIPhotometricParameters(config)


def test_roman_wfi_photometric_parameters_match_reference_formulae(
    photometric_params: RomanWFIPhotometricParameters,
) -> None:
    """Verify the photometric parameter lookup matches the underlying analytical formulae.

    This test reads the filter-specific zero point, effective pixel count, peak flux, and
    background rates from the Roman parameter tables and checks them against the same
    equations used to define saturation and faint-limit magnitudes. It guards against unit,
    normalization, or lookup mistakes in the photometric model.
    """
    filt = "f106"
    t_exp = 1.0 * u.s

    photometric_params._setup(filt)
    zp = photometric_params._properties["zp"]
    n_eff = photometric_params._properties["n_eff"]
    f_peak = photometric_params._properties["f_peak"]
    f_thermal = photometric_params._properties["f_thermal"]
    f_zodi = photometric_params._properties["f_min_zodi"]

    assert photometric_params._get_val(photometric_params.zero_points, "z_r", filt) == zp
    assert photometric_params._get_val(photometric_params.filter_params, "center_psf_n_eff_pixel", filt) == n_eff
    assert photometric_params._get_val(photometric_params.filter_params, "center_psf_peak_flux", filt) == f_peak
    assert photometric_params._get_val(photometric_params.thermal_bkg, "rate", filt) == f_thermal
    assert photometric_params._get_val(photometric_params.zodiacal_light, "rate", filt) == f_zodi

    c_sat = 120000.0 * u.ct
    alpha_sat = c_sat / (f_peak * t_exp)
    expected_msat = zp - 2.5 * np.log10(alpha_sat.value) * u.mag
    assert photometric_params._saturation_limit_mag(c_sat, t_exp, zp, f_peak).to_value(u.mag) == pytest.approx(
        expected_msat.to_value(u.mag)
    )
    assert photometric_params.get_msat(filt, t_exp).to_value(u.mag) == pytest.approx(expected_msat.to_value(u.mag))

    snr = 50.0
    f_bkgd = 2.0 * f_zodi + f_thermal
    term_b = snr**2 / t_exp
    term_c = (snr**2 * n_eff * f_bkgd / u.ct) / t_exp
    f_source_limit = (term_b + np.sqrt(term_b**2 + 4.0 * term_c)) / 2.0
    expected_mfaint = zp - 2.5 * np.log10(f_source_limit.value) * u.mag
    assert photometric_params._faint_limit_mag(snr, t_exp, zp, n_eff, f_bkgd).to_value(u.mag) == pytest.approx(
        expected_mfaint.to_value(u.mag)
    )
    assert photometric_params.get_mfaint(filt, t_exp).to_value(u.mag) == pytest.approx(expected_mfaint.to_value(u.mag))


def test_faint_limit_matches_analytical_regimes(
    photometric_params: RomanWFIPhotometricParameters,
) -> None:
    """Check the faint-limit calculation in both zero-background and background-dominated limits.

    The signal-to-noise relation can be solved analytically in two simplifying regimes,
    and this test compares the implementation to those closed-form expectations. It ensures
    the numerical code handles both limiting cases without drifting away from theory.
    """
    filt = "f106"
    t_exp = 10.0 * u.s
    snr = 50.0
    photometric_params._setup(filt)
    zp = photometric_params._properties["zp"]
    n_eff = photometric_params._properties["n_eff"]

    # Starting from the SNR relation
    #   snr = f_source / sqrt((n_eff * f_bkgd + f_source) / t_exp),
    # the limiting source rate is the positive root of
    #   f_source^2 - (snr^2 / t_exp) f_source - (snr^2 * n_eff * f_bkgd / t_exp) = 0.
    # In the zero-background limit, f_source_limit = snr^2 / t_exp.
    f_bkgd = 0.0 * (u.ct / (u.pix * u.s))
    expected_flux = (snr**2 / t_exp).value  # counts s^-1 in the implementation convention
    expected_mag = zp - 2.5 * np.log10(expected_flux) * u.mag
    actual_mag = photometric_params._faint_limit_mag(snr, t_exp, zp, n_eff, f_bkgd)
    assert actual_mag.to_value(u.mag) == pytest.approx(expected_mag.to_value(u.mag))

    # In the background-dominated limit, n_eff * f_bkgd >> f_source, so the root
    # asymptotes to f_source_limit ~= snr * sqrt((n_eff * f_bkgd / u.ct) / t_exp).
    f_bkgd = 100000.0 * (u.ct / (u.pix * u.s))
    expected_flux = (snr * np.sqrt((n_eff * f_bkgd / u.ct) / t_exp)).value
    expected_mag = zp - 2.5 * np.log10(expected_flux) * u.mag
    actual_mag = photometric_params._faint_limit_mag(snr, t_exp, zp, n_eff, f_bkgd)
    assert actual_mag.to_value(u.mag) == pytest.approx(expected_mag.to_value(u.mag), rel=5e-3)


def test_psf_flux_error_theory_matches_closed_form_expression(
    photometric_params: RomanWFIPhotometricParameters,
) -> None:
    """Validate the PSF flux uncertainty against the closed-form noise model.

    The theory for the PSF flux error is derived from the source and background terms in
    the SNR equation, and this test compares both scalar and array inputs to that analytic
    expectation. It is a direct check that the implementation is using the correct units and
    noise model.
    """
    filt = "f106"
    t_exp = 1.0 * u.s
    photometric_params._setup(filt)

    zp = photometric_params._properties["zp"]
    n_eff = photometric_params._properties["n_eff"]
    f_bkgd = 2.0 * photometric_params._properties["f_min_zodi"] + photometric_params._properties["f_thermal"]

    psf_flux = 1.0 * u.uJy
    kappa = 10 ** ((31.4 - zp.value) / 2.5) * (u.nJy / (u.ct / u.s))
    f_src = psf_flux / kappa
    term_err = t_exp * (n_eff * f_bkgd + f_src)
    sigma_f = u.ct * np.sqrt(term_err.value) / t_exp
    expected_scalar = (sigma_f * kappa).to(u.nJy)

    actual_scalar = photometric_params.get_psf_flux_error_theory(filt, psf_flux, t_exp)
    assert actual_scalar.unit == u.nJy
    assert actual_scalar.to_value(u.nJy) == pytest.approx(expected_scalar.to_value(u.nJy))

    psf_flux_array = np.array([0.1, 1.0, 10.0]) * u.uJy
    expected_array = (u.ct * np.sqrt((t_exp * (n_eff * f_bkgd + (psf_flux_array / kappa))).value) / t_exp * kappa).to(u.nJy)
    actual_array = photometric_params.get_psf_flux_error_theory(filt, psf_flux_array, t_exp)
    assert actual_array.unit == u.nJy
    assert np.allclose(actual_array.to_value(u.nJy), expected_array.to_value(u.nJy), rtol=1e-12, atol=1e-12)
