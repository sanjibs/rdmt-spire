"""Tests for aperture_psf.PhotometryCatalog using a fully simulated L2 image.

Unlike test_aperture_psf_soc_simulation.py (which requires real SOC-simulation
data, a real parquet catalog, and CRDS access), this module builds a synthetic
Roman-L2-like ASDF file and a matching parquet source catalog entirely
in-memory. Point sources are laid out on a grid and injected with a Gaussian
PRF via photutils, and PhotometryCatalog is monkeypatched to fit with that
same analytic Gaussian PSF, so the whole test is self-contained and does not
need CRDS or external data files.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("asdf")
pytest.importorskip("photutils")
pytest.importorskip("astropy")
pytest.importorskip("pyarrow")

import asdf
import astropy.units as u
from astropy.table import QTable, Table
from astropy.wcs import WCS
from photutils.datasets import make_model_image
from photutils.psf import CircularGaussianPRF

from ...monitors.photoastro import aperture_psf as aperture_psf_module
from ...monitors.photoastro.aperture_psf import (
    REQUIRED_OUTPUT_COLUMNS,
    PhotometryCatalogPipeline,
)
from ...utilities import aws_utils
from ...utilities.phot.photoastro_utils import conversion_factor_for_flux_density

IMAGE_SHAPE = (256, 256)
PIXEL_SCALE_ARCSEC = 0.11
PSF_FWHM_PIX = 0.8
BACKGROUND_LEVEL = 0.25 # ct/(pix s)
T_EXP=60.0 # s
NOISE_SIGMA = 10.0
MAX_APERTURE = 30  # in pixels, corresponding to the largest aperture radius used in tests
# curve of growth measurements use 2.8" (~25 px) as the background annulus inner radius.
INIT_SCATTER = PSF_FWHM_PIX*3

def _make_source_table(image_shape: tuple[int, int], rng: np.random.Generator) -> Table:
    """Lay out point sources on a regular grid, offset from the image edges
    by the same spacing used between sources.
    """
    spacing = max(int(PSF_FWHM_PIX * 5), MAX_APERTURE)
    xs = np.arange(spacing, image_shape[1] - spacing + 1, spacing, dtype=float)
    ys = np.arange(spacing, image_shape[0] - spacing + 1, spacing, dtype=float)
    xx, yy = np.meshgrid(xs, ys)

    table = Table()
    table["x_0"] = xx.ravel()
    table["y_0"] = yy.ravel()
    table["flux"] = rng.uniform(4.0e4, 1.2e5, size=table["x_0"].size)/T_EXP
    print('sources:',len(table))
    return table


def _make_wcs() -> WCS:
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crval = [10.0, 20.0]
    wcs.wcs.crpix = [IMAGE_SHAPE[1] / 2.0, IMAGE_SHAPE[0] / 2.0]
    wcs.wcs.cdelt = [-PIXEL_SCALE_ARCSEC / 3600.0, PIXEL_SCALE_ARCSEC / 3600.0]
    return wcs


def _make_gaussian_psf_model() -> CircularGaussianPRF:
    return CircularGaussianPRF(flux=1.0, fwhm=PSF_FWHM_PIX)


def _build_simulated_image(sources: Table, rng: np.random.Generator) -> np.ndarray:
    """Render Gaussian-PRF point sources plus flat background and noise."""
    model = _make_gaussian_psf_model()
    data = make_model_image(IMAGE_SHAPE, model, sources, x_name="x_0", y_name="y_0")
    sigma=(NOISE_SIGMA+np.sqrt(BACKGROUND_LEVEL*T_EXP)+np.sqrt(data*T_EXP))/T_EXP
    data += BACKGROUND_LEVEL
    data += rng.normal(scale=sigma, size=IMAGE_SHAPE)
    return data.astype(np.float32)


@pytest.fixture(scope="module")
def simulated_asdf_and_catalog(tmp_path_factory: pytest.TempPathFactory):
    """Create a synthetic Roman-L2-like ASDF file and matching parquet catalog."""
    rng = np.random.default_rng(seed=42)
    sources = _make_source_table(IMAGE_SHAPE, rng)
    data = _build_simulated_image(sources, rng)
    err = np.full(IMAGE_SHAPE, NOISE_SIGMA, dtype=np.float32)
    dq = np.zeros(IMAGE_SHAPE, dtype=np.uint32)

    tmp_dir = tmp_path_factory.mktemp("simulated_l2")
    l2_dir = tmp_dir / "l2"
    l4_dir = tmp_dir / "l4"
    l2_dir.mkdir()
    l4_dir.mkdir()

    filename = "r_test01_wfi01_f106_cal.asdf"
    l2_path = l2_dir / filename

    pixel_area_sr = (PIXEL_SCALE_ARCSEC * u.arcsec).to(u.rad).value ** 2

    tree = {
        "roman": {
            "data": data,
            "err": err,
            "dq": dq,
            "meta": {
                "filename": filename,
                "instrument": {"detector": "WFI01", "optical_element": "F106"},
                "photometry": {
                    "conversion_megajanskys": 1.0,
                    "pixel_area": pixel_area_sr,
                },
                "wcs": _make_wcs(),
                "guide_star": {},
            },
        }
    }
    af = asdf.AsdfFile(tree)
    af.write_to(l2_path)

    catalog = QTable()
    catalog["label"] = np.arange(1, len(sources) + 1)
    catalog["x_centroid"] = sources["x_0"].copy()*u.pix
    catalog["y_centroid"] = sources["y_0"].copy()*u.pix
    catalog["x_psf"] = sources["x_0"].copy()*u.pix
    catalog["y_psf"] = sources["y_0"].copy()*u.pix
    catalog["is_extended"] = np.zeros(len(sources), dtype=bool)
    cat_path = l4_dir / filename.replace("_cal.asdf", "_cat.parquet")
    catalog.write(cat_path, format="parquet")

    config=aws_utils.get_monitor_config()
    config["RDMT_SPIRE_L4_DIR"] = str(l4_dir)
    return l2_path, config, sources


@pytest.fixture()
def patched_gaussian_psf(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force the current module-level PSF builder to use the same analytic
    Gaussian PRF used to simulate the image, bypassing CRDS/STPSF lookups.
    """

    def _fake_build_psf_model(*args, **kwargs):
        return _make_gaussian_psf_model()

    monkeypatch.setattr(aperture_psf_module, "build_psf_model", _fake_build_psf_model)


def test_run_produces_required_columns(
    simulated_asdf_and_catalog: tuple, patched_gaussian_psf: None
) -> None:
    """Check that the end-to-end photometry run emits the required output schema.

    The synthetic image includes a known set of sources, and the routine should always
    return at least the columns required by the public API. This test ensures the
    workflow produces the expected table structure and preserves the number of sources.
    """
    l2_path, config, sources = simulated_asdf_and_catalog
    print(config)
    with asdf.open(l2_path) as af:
        out = PhotometryCatalogPipeline(af, config).catalog

    for name, unit in REQUIRED_OUTPUT_COLUMNS:
        print(f'Checking column: {name} with expected unit: {unit}, actual unit: {out[name].unit}')
        assert name in out.colnames
        assert out[name].unit == u.Unit(unit)
    assert len(out) == len(sources)


def test_psf_photometry_recovers_positions(
    simulated_asdf_and_catalog: tuple, patched_gaussian_psf: None
) -> None:
    """Verify PSF centroiding can recover the injected source positions.

    The image is generated from a Gaussian PRF at known coordinates, so the test checks
    that the fitted x/y PSF positions are close to the truth to within a small tolerance.
    The RMS-based assertion also guards against broad systematic centroid errors.
    """
    l2_path, config, sources = simulated_asdf_and_catalog
    with asdf.open(l2_path) as af:
        out = PhotometryCatalogPipeline(af, config, init_scatter=INIT_SCATTER).run()

    true_x = np.asarray(sources["x_0"], dtype=float)
    true_y = np.asarray(sources["y_0"], dtype=float)

    xy_out=np.array([out["x_psf"], out["y_psf"]])
    xy_true=np.array([true_x, true_y])
    res = xy_out - xy_true
    print('sources:',len(out['x_psf']))
    print('rmse of positional offsets / PSF_FWHM_PIX:',np.sqrt(np.mean(res**2)*2)/PSF_FWHM_PIX)
    print('medae of positional offsets / PSF_FWHM_PIX:',np.median(np.abs(res))/PSF_FWHM_PIX)
    print('mae of positional offsets / PSF_FWHM_PIX:',np.mean(np.abs(res))/PSF_FWHM_PIX)


    assert np.all(np.isfinite(out["x_psf"]))
    assert np.all(np.isfinite(out["y_psf"]))
    np.testing.assert_allclose(np.asarray(out["x_psf"], dtype=float), true_x, atol=0.5)
    np.testing.assert_allclose(np.asarray(out["y_psf"], dtype=float), true_y, atol=0.5)
    assert (np.sqrt(np.mean(res**2)*2))/PSF_FWHM_PIX < 0.01


def test_psf_photometry_recovers_flux(
    simulated_asdf_and_catalog: tuple, patched_gaussian_psf: None
) -> None:
    """Check that PSF flux measurements are close to the injected source flux.

    The test converts the known source flux into the detector's native flux units and
    compares that to the fitted PSF flux column. It enforces a tight relative-error limit
    to confirm that the photometric calibration and fitting are behaving correctly.
    """
    l2_path, config, sources = simulated_asdf_and_catalog
    with asdf.open(l2_path) as af:
        factor = conversion_factor_for_flux_density(af)
        out = PhotometryCatalogPipeline(af, config, init_scatter=INIT_SCATTER).run()

    true_flux = np.asarray(sources["flux"], dtype=float) * factor
    recovered_flux = np.asarray(out["psf_flux"], dtype=float)

    rel_err = np.abs(recovered_flux - true_flux) / true_flux
    print('medae of relative psf flux error / poisson error:', np.median(np.abs(rel_err)))
    assert np.all(rel_err < 0.1)
    assert np.median(rel_err) < 0.01


def test_largest_aperture_flux_matches_injected_flux(
    simulated_asdf_and_catalog: tuple, patched_gaussian_psf: None
) -> None:
    """Validate that a large aperture captures nearly all of the simulated source flux.

    Because the Gaussian PSF is narrow and the 0.8-arcsec aperture is large enough to
    include almost all of its light, the measured aperture flux should agree with the
    injected source flux to within a small fractional error.
    """
    # The 0.8" aperture radius (~7.3 px) encloses effectively all flux of the
    # simulated 0.8-px-FWHM Gaussian PRF, so it should match the injected flux.
    l2_path, config, sources = simulated_asdf_and_catalog
    with asdf.open(l2_path) as af:
        factor = conversion_factor_for_flux_density(af)
        out = PhotometryCatalogPipeline(af, config, init_scatter=INIT_SCATTER, refine_centroids=True).run()

    true_flux = np.asarray(sources["flux"], dtype=float) * factor
    recovered_flux = np.asarray(out["aper08_flux"], dtype=float)

    rel_err = np.abs(recovered_flux - true_flux) / true_flux
    print('medae of relative aperture flux error / poisson error:', np.median(np.abs(rel_err)))
    assert np.all(rel_err < 0.1)
    assert np.median(rel_err) < 0.01
