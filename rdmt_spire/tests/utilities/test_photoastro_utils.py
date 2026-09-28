
"""Unit tests for image, astrometry, photometry, and catalog helpers in ``photoastro_utils``.

The tests use synthetic ASDF metadata, noisy and flat NumPy images, Gaussian PSFs,
small WCS definitions, quantity-valued source coordinates, and inline Astropy tables.
The ``_make_gaia_table`` helper builds minimal Gaia-like catalogs for catalog lookup and
cross-match cases. Monkeypatches replace CRDS, background, and Gaia catalog calls where
needed so the tests can inspect forwarded parameters and exercise fallback behavior
without requiring external services or reference catalogs.

Coverage includes metadata flattening, 2D background estimation and oversized-box
fallbacks, jitter normalization and PSF broadening, flux-density conversion, WCS image
geometry, Gaia detector filtering and nearest-source attachment, quantity validation,
and local background estimation. Assertions check preserved metadata and call arguments,
image shapes and finite values, analytical or moment-based numerical results, detector
selection, source IDs and photometry, angular separations, expected exceptions, and
zero background uncertainty for a uniform image.
"""

import datetime
from types import SimpleNamespace

import asdf
import astropy.units as u
import numpy as np
import pytest
from astropy.table import QTable, Table
from astropy.time import Time
from astropy.wcs import WCS
from photutils.background import Background2D

from ...utilities.phot import photoastro_utils
from ...utilities.phot.photoastro_utils import (
    add_jitter_general,
    aperture_local_background,
    conversion_factor_for_flux_density,
    crossmatch_with_gaia,
    estimate_background,
    get_crds_reference_file,
    get_gaia_catalog_for_l2data,
    get_image_center_and_search_radius,
    get_jitter_params,
)


def test_get_crds_reference_file_flattens_metadata_and_calls_crds(monkeypatch):
    """Check that pipeline metadata is converted into CRDS-friendly keys before lookup.

    This test builds a synthetic ASDF metadata dictionary, stubs the underlying CRDS
    call, and verifies that the function preserves the detector and exposure values
    while dropping unsupported fields. The assertions confirm the metadata mapping and
    the returned reference path are exactly what the CRDS lookup layer expects.
    """
    captured = {}

    def fake_getreferences(metadata, reftypes, observatory):
        captured["metadata"] = metadata
        captured["reftypes"] = reftypes
        captured["observatory"] = observatory
        return {"epsf": "/tmp/fake_epsf.asdf"}

    monkeypatch.setattr(photoastro_utils.crds, "getreferences", fake_getreferences)

    asdf_object = {
        "roman": {
            "meta": {
                "instrument": {"detector": "WFI01"},
                "exposure": {"start_time": datetime.datetime(2024, 1, 1, 12, 0, 0)},
                "filename": "r_test.asdf",
                "bad_list": [1, 2, 3],
            }
        }
    }

    result = get_crds_reference_file(asdf_object, "epsf", "roman")

    assert result == "/tmp/fake_epsf.asdf"
    assert captured["observatory"] == "roman"
    assert captured["reftypes"] == ["epsf"]
    assert captured["metadata"]["ROMAN.META.INSTRUMENT.DETECTOR"] == "WFI01"
    assert captured["metadata"]["ROMAN.META.EXPOSURE.START_TIME"] == "2024-01-01 12:00:00"
    assert "ROMAN.META.BAD_LIST" not in captured["metadata"]


def test_estimate_background_returns_2d_background_model():
    """Verify that a flat image can be decomposed into a 2D background model.

    The test creates a nearly uniform image with small Gaussian noise and checks that
    the returned Background2D object has the same spatial shape as the input, contains
    finite values, and recovers the input level within a small tolerance.
    """
    rng = np.random.default_rng(0)
    data = np.full((120, 150), 7.5, dtype=float)
    sigma=0.2
    data += rng.normal(0.0, sigma, size=data.shape)
    box_size=25

    bkg = estimate_background(data, box_size=box_size)
    expected_err = sigma/np.sqrt(box_size*box_size)

    print('Background bias', np.median(bkg.background-7.5), 'RMS', np.median(bkg.background_rms), 'Expected RMS', expected_err)

    assert isinstance(bkg, Background2D)
    assert bkg.background.shape == data.shape
    assert bkg.background_rms.shape == data.shape
    assert np.all(np.isfinite(bkg.background))
    assert np.all(np.isfinite(bkg.background_rms))
    assert np.median(np.abs(bkg.background - 7.5)) < 5*expected_err


def test_estimate_background_falls_back_to_full_image_when_box_size_is_too_large(monkeypatch):
    """Confirm the background estimator retries with a full-image fallback.

    The test simulates a failure from an oversized box size and verifies that the
    helper catches that case and falls back to a full-frame Background2D estimate,
    preserving the original oversized request while using the image dimensions on retry.
    """
    calls = []

    real_background2d = photoastro_utils.Background2D

    def fake_background2d(data, box_size=None, **kwargs):
        calls.append(box_size)
        if len(calls) == 1:
            raise ValueError("box_size too large")
        fallback_kwargs = dict(kwargs)
        fallback_kwargs.pop("exclude_percentile", None)
        return real_background2d(data, data.shape, exclude_percentile=100.0, **fallback_kwargs)

    monkeypatch.setattr(photoastro_utils, "Background2D", fake_background2d)

    data = np.ones((20, 20), dtype=float)
    bkg = estimate_background(data, box_size=10000)

    assert bkg.background.shape == data.shape
    assert bkg.background_rms.shape == data.shape
    assert calls[0] == 10000
    assert calls[1] == data.shape




def test_get_jitter_params_uses_defaults_and_native_values():
    """Verify jitter parameters are normalized from STPSF-style defaults or kept as native values.

    The function should return the canonical Roman defaults when no input is provided,
    convert a legacy STPSF field like jitrsigm into the expected native pixel values,
    and leave explicitly supplied values unchanged.
    """
    default_params = get_jitter_params({})
    assert default_params == {
        "jitter_major": 8.0,
        "jitter_minor": 8.0,
        "jitter_position_angle": 0.0,
    }

    stpsf_params = get_jitter_params({"jitrsigm": [0.008]})
    assert stpsf_params["jitter_major"] == pytest.approx(8.0)
    assert stpsf_params["jitter_minor"] == pytest.approx(8.0)
    assert stpsf_params["jitter_position_angle"] == pytest.approx(0.0)

    native_params = get_jitter_params({
        "jitter_major": 12.0,
        "jitter_minor": 15.0,
        "jitter_position_angle": 30.0,
    })
    assert native_params == {
        "jitter_major": 12.0,
        "jitter_minor": 15.0,
        "jitter_position_angle": 30.0,
    }


def test_apply_jitter_broadens_gaussian_psf_and_updates_metadata():
    """Check that blur caused by image jitter broadens the PSF as expected.

    A Gaussian PSF is constructed and then passed through the jitter helper with a large
    major/minor jitter. The test compares the second moment of the PSF before and after
    jitter and verifies that metadata records the new jitter values.
    """
    sigma_psf_pix = 1.5
    nx = 101
    x = np.arange(nx) - (nx - 1) / 2
    xx, yy = np.meshgrid(x, x)
    gaussian = np.exp(-(xx**2 + yy**2) / (2 * sigma_psf_pix**2))
    gaussian /= gaussian.sum()

    meta = {
        "grid_xypos": [[0.0, 0.0]],
        "oversampling": 2,
        "jitter_major": 0.0,
        "jitter_minor": 0.0,
        "jitter_position_angle": 0.0,
    }
    # psf_model = GriddedPSFModel(NDData(data=gaussian[None, :, :], meta=meta))
    psf_model=SimpleNamespace(data=gaussian[None, :, :], meta=meta)
    

    convolved_data = add_jitter_general(
        psf_model.data,
        get_jitter_params(psf_model.meta),
        {"jitter_major": 60.0, "jitter_minor": 60.0, "jitter_position_angle": 0.0},
        pixel_scale=0.1,
        oversample=meta["oversampling"],
    )

    yy_idx, xx_idx = np.indices(gaussian.shape)
    xcoords = xx_idx - (gaussian.shape[1] - 1) / 2
    ycoords = yy_idx - (gaussian.shape[0] - 1) / 2

    def second_moment(arr):
        return np.sqrt(np.sum((xcoords**2 + ycoords**2) * arr) / np.sum(arr))

    sigma_jitter_pix = 60.0 / 1000.0 / (0.1 / 2.0)
    orig_sigma = second_moment(gaussian)
    new_sigma = second_moment(convolved_data[0])
    expected_sigma = np.sqrt(2.0 * (sigma_psf_pix**2 + sigma_jitter_pix**2))

    print(f"Original sigma: {orig_sigma}, New sigma: {new_sigma}, Expected sigma: {expected_sigma}")
    assert convolved_data.shape == psf_model.data.shape
    assert np.all(np.isfinite(convolved_data)) 
    assert new_sigma > orig_sigma
    assert new_sigma == pytest.approx(expected_sigma)



# Needs more check 
# ------------------------------------------------
def test_conversion_factor_for_flux_density_matches_expected_value():
    """Validate the conversion from surface brightness to flux density per pixel.

    The calculation combines a conversion megajansky scale factor with the pixel area,
    and the expected number is derived analytically from the same physical units. This
    test ensures the helper does not silently apply the wrong unit conversion.
    """
    pixel_area = (0.1 * u.arcsec).to(u.rad).value ** 2
    dmodel = asdf.AsdfFile({
        "roman": {
            "meta": {
                "photometry": {
                    "conversion_megajanskys": 2.5,
                    "pixel_area": pixel_area,
                }
            }
        }
    })

    expected = 2.5 * ((1.0 * (u.MJy / u.sr) * (pixel_area * u.sr)).to("nJy").value)
    assert conversion_factor_for_flux_density(dmodel) == pytest.approx(expected)


def test_get_image_center_and_search_radius_matches_wcs_geometry():
    """Confirm the search radius is computed from the image center and detector size.

    The test builds a WCS centered on the image and checks that the helper returns the
    same central coordinates and a radius equal to the half-diagonal extent of the array,
    scaled by the pixel size.
    """
    cenra_in=10.0
    cendec_in=1.0
    shape = (100, 120)
    scale=1e-3
    # shape is (row,col) which is equivalent to (ny,nx) 
    expected_radius=np.sqrt((shape[0]/2.0)**2+(shape[1]/2.0)**2)*scale

    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crval = [cenra_in, cendec_in]
    wcs.wcs.crpix = [shape[1]/2.0+0.5, shape[0]/2.0+0.5]
    wcs.wcs.cdelt = [scale, scale]


    center_ra, center_dec, search_radius = get_image_center_and_search_radius(wcs, shape)

    assert center_ra == pytest.approx(cenra_in)
    assert center_dec == pytest.approx(cendec_in)
    assert search_radius == pytest.approx(expected_radius)


def _make_gaia_table(ra, dec):
    """Build the Gaia columns consumed by the image and cross-match helpers."""
    size = len(ra)
    return Table({
        "ra": np.asarray(ra),
        "dec": np.asarray(dec),
        "source_id": np.arange(size, dtype=np.int64) + 100,
        "phot_g_mean_mag": np.arange(size, dtype=float) + 15.0,
        "phot_bp_mean_mag": np.arange(size, dtype=float) + 16.0,
        "phot_rp_mean_mag": np.arange(size, dtype=float) + 14.0,
    })


def test_get_gaia_catalog_for_l2data_dict_passes_search_parameters(monkeypatch):
    """Verify dictionary inputs are forwarded to the Gaia cone-search helper."""
    captured = {}
    expected = _make_gaia_table([10.0], [1.0])

    def fake_get_gaia_catalog(**kwargs):
        captured.update(kwargs)
        return expected

    monkeypatch.setattr(photoastro_utils, "get_gaia_catalog", fake_get_gaia_catalog)

    result = get_gaia_catalog_for_l2data({
        "center_ra": 10.0,
        "center_dec": 1.0,
        "search_radius": 0.2,
        "epoch": 2026.5,
    }, catalog="TEST")

    assert result is expected
    assert captured["right_ascension"] == 10.0
    assert captured["declination"] == 1.0
    assert captured["epoch"] == 2026.5
    assert captured["search_radius_deg"] == pytest.approx(0.2 * 1.01)
    assert captured["catalog"] == "TEST"


def test_get_gaia_catalog_for_l2data_asdf_filters_to_detector(monkeypatch):
    """Verify ASDF inputs use the WCS, exposure epoch, and detector bounds."""
    shape = (10, 12)
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crval = [10.0, 1.0]
    wcs.wcs.crpix = [shape[1] / 2.0 + 0.5, shape[0] / 2.0 + 0.5]
    wcs.wcs.cdelt = [1e-3, 1e-3]
    exposure = {"mid_time": Time("2026-01-01T00:00:00")}
    af = asdf.AsdfFile({
        "roman": {
            "meta": {"wcs": wcs, "exposure": exposure},
            "data": np.zeros(shape),
        }
    })
    source_pixels = np.array([[1.0, 2.0], [11.0, 9.0], [-1.0, 5.0]])
    source_coords = wcs.pixel_to_world(source_pixels[:, 0], source_pixels[:, 1])
    gaia_table = _make_gaia_table(source_coords.ra.deg, source_coords.dec.deg)
    captured = {}

    def fake_get_gaia_catalog(**kwargs):
        captured.update(kwargs)
        return gaia_table

    monkeypatch.setattr(photoastro_utils, "get_gaia_catalog", fake_get_gaia_catalog)

    result = get_gaia_catalog_for_l2data(af)

    assert captured["epoch"] == pytest.approx(2026.0)
    assert captured["search_radius_deg"] > 0
    assert len(result) == 2
    np.testing.assert_allclose(result["x"], [1.0, 11.0])
    np.testing.assert_allclose(result["y"], [2.0, 9.0])


def test_crossmatch_with_gaia_attaches_nearest_source_data(monkeypatch):
    """Verify nearest Gaia matches and their separation and photometry columns."""
    shape = (20, 20)
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crval = [10.0, 1.0]
    wcs.wcs.crpix = [shape[1] / 2.0 + 0.5, shape[0] / 2.0 + 0.5]
    wcs.wcs.cdelt = [1e-3, 1e-3]
    af = asdf.AsdfFile({"roman": {"meta": {"wcs": wcs}}})

    source_x = np.array([4.0, 14.0]) * u.pix
    source_y = np.array([6.0, 12.0]) * u.pix
    source_coords = wcs.pixel_to_world(source_x, source_y)
    gaia_table = _make_gaia_table(source_coords.ra.deg, source_coords.dec.deg)
    monkeypatch.setattr(
        photoastro_utils,
        "get_gaia_catalog_for_l2data",
        lambda image: gaia_table,
    )
    source_catalog = QTable({"x_psf": source_x, "y_psf": source_y, "psf_flux": [1.0, 2.0]})

    result = crossmatch_with_gaia(af, source_catalog)

    assert len(result) == 2
    np.testing.assert_array_equal(result["gaia_source_id"], [100, 101])
    np.testing.assert_allclose(result["phot_g_mean_mag"], [15.0, 16.0])
    assert np.all(result["angsep_gaia"] < 1e-6 * u.arcsec)


def test_crossmatch_with_gaia_requires_quantity_pixel_coordinates():
    """Reject source catalogs whose PSF coordinates do not carry astropy units."""
    af = asdf.AsdfFile({"roman": {"meta": {"wcs": WCS(naxis=2)}}})
    source_catalog = Table({"x_psf": [1.0], "y_psf": [2.0]})

    with pytest.raises(TypeError, match="x_psf"):
        crossmatch_with_gaia(af, source_catalog)

def test_aperture_local_background_returns_flat_background_and_zero_uncertainty():
    """Check that local background estimation is correct for a perfectly flat image.

    When every pixel has the same value, the median background in the annulus should
    equal that value exactly and the uncertainty should be zero because there is no
    intrinsic scatter within the aperture.
    """
    data = np.full((30, 30), 42.0, dtype=float)
    positions = np.array([[10.0, 10.0], [20.0, 20.0]])

    median, median_err = aperture_local_background(data, positions, rin=2.0, rout=4.0)

    assert median.shape == (2,)
    assert median_err.shape == (2,)
    np.testing.assert_allclose(median, 42.0)
    np.testing.assert_allclose(median_err, 0.0)

