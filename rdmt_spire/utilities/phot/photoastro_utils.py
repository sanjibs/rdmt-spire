"""Reusable astrometry/photometry utility functions for Roman WFI L2 images.

Stateless helpers used by ``photometry.Monitor``: CRDS reference lookup,
background estimation, WCS pixel scale, PSF model construction, PSF jitter
correction, and flux unit conversion.
"""

from __future__ import annotations

import datetime
import logging
import warnings
from collections import OrderedDict
from typing import Any, Union

import asdf
import astropy.time
import astropy.units as u
import crds
import numpy as np
import roman_datamodels.datamodels as rdm
from astropy.convolution import Box2DKernel, convolve
from astropy.coordinates import SkyCoord
from astropy.nddata import NDData
from astropy.stats import SigmaClip
from astropy.table import QTable, Table
from astropy.utils.exceptions import AstropyUserWarning
from photutils.aperture import CircularAnnulus
from photutils.background import Background2D, MedianBackground
from photutils.psf import (
    CircularGaussianPRF,
    GriddedPSFModel,
)
from roman_datamodels import datamodels

from .. import aws_utils
from .gaia import get_gaia_catalog
from .psf import _evaluate_gaussian_fft

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


def get_crds_reference_file(asdf_object, reference_file_type, observatory):
    """
    Retrieve a calibration reference file from CRDS for a given observation.

    Queries the Calibration Reference Data System (CRDS) to find the best-matching
    reference file for the specified reference type, using metadata from the input
    ASDF object. Metadata is flattened and converted to the format expected by CRDS.

    Simplified version of the following, and with the ability to work without roman_datamodels:
    https://github.com/spacetelescope/stpipe/blob/main/src/stpipe/crds_client.py

    Parameters
    ----------
    asdf_object : ASDF-like object
        Open Roman L2 ASDF-like object exposing mapping access to model data.
        Metadata is extracted from the observatory-specific branch (e.g., 'roman').
    reference_file_type : str
        Type of reference file to retrieve (e.g., 'epsf', 'flat', 'dark').
    observatory : str
        Telescope/observatory name used with CRDS (e.g., 'jwst', 'roman').

    Returns
    -------
    str or None
        Absolute file path of the reference file in the CRDS cache,
        or None if no matching reference is found.

    Notes
    -----
    This function flattens nested metadata dictionaries and converts non-scalar
    types (e.g., astropy Time objects) to string representations that CRDS can parse.
    """

    def convert_val(val):
        """Convert Time objects to ISO format strings for CRDS compatibility."""
        if isinstance(val, datetime.datetime) or isinstance(val, astropy.time.Time):
            return str(val)
        return val

    def flatten_dict(d, parent_key="", observatory_prefix="ROMAN.META"):
        """Recursively flatten a nested dictionary for CRDS matching.

        Converts nested dict structures into dot-separated uppercase keys.
        Includes only scalar values; skips lists, tuples, and other complex types.
        Converts astropy Time objects to strings for CRDS compatibility.
        """
        for k, v in d.items():
            if parent_key:
                new_key = f"{parent_key}.{k.upper()}"
            else:
                new_key = f"{observatory_prefix}.{k.upper()}" if observatory_prefix else k.upper()

            if isinstance(v, dict):
                flatten_dict(v, new_key, observatory_prefix=None)
            elif not isinstance(v, (list, tuple)):
                flat_metadata[new_key] = convert_val(v)

    # Flatten nested metadata dictionary to the format CRDS expects
    flat_metadata = {}

    # Extract and flatten metadata with observatory-specific prefix
    metadata = asdf_object[observatory.lower()]["meta"]
    flatten_dict(metadata, parent_key="", observatory_prefix=f"{observatory.upper()}.META")

    # Query CRDS for the best-matching reference file
    refs = crds.getreferences(flat_metadata, reftypes=[reference_file_type], observatory=observatory)
    return refs.get(reference_file_type)


def estimate_background(data: np.ndarray, box_size: int = 1000, coverage_mask: np.ndarray | None = None) -> Background2D:
    """Estimate a smooth 2D background model and RMS from image data.

    Computes a spatial 2D background using a mesh of boxes, with each box
    estimated using sigma-clipped (3-sigma, MAD) median filter. Output is
    smoothed with a 3×3 median filter.

    This approach mirrors the Roman pipeline's ``romancal.source_catalog._background.RomanBackground``.
    https://github.com/spacetelescope/romancal/blob/main/romancal/source_catalog/_background.py
    https://github.com/spacetelescope/romancal/blob/main/romancal/source_catalog/source_catalog_step.py

    Parameters
    ----------
    data : np.ndarray
        2D science image array.
    box_size : int, optional
        Edge length in pixels of the square mesh boxes. Default is 1000.
    coverage_mask : np.ndarray | None, optional
        Boolean mask where True indicates pixels with no valid data (e.g., cosmic rays).
        Default is None (all pixels used).

    Returns
    -------
    photutils.background.Background2D
        Object exposing ``background`` and ``background_rms`` 2D images
        for the input data shape.
    """

    kwargs = dict(
        filter_size=(3, 3),
        mask=None,
        coverage_mask=coverage_mask,
        sigma_clip=SigmaClip(sigma=3.0, stdfunc="mad_std"),
        bkg_estimator=MedianBackground(),
    )
    try:
        return Background2D(data, box_size, **kwargs)
    except ValueError:
        # Fallback: single box spanning the entire array when box_size is too large
        return Background2D(data, data.shape, exclude_percentile=100.0, **kwargs)


# Similar to psf._get_jitter_params
# RDMT addition: functions to handle standard and STPSF jitter parameters
def get_jitter_params(meta: dict) -> dict[str, float]:
    """Extract jitter parameters from metadata with sensible defaults.

    Handles both native metadata format (jitter_major/minor/position_angle in mas)
    and STPSF format (jitrsigm in arcsec). Returns standardized jitter parameters
    in milliarcseconds.

    Parameters
    ----------
    meta : dict
        Metadata dictionary. Expected keys (in priority order):
        - ``jitter_major``, ``jitter_minor``, ``jitter_position_angle`` (native format)
        - ``jitrsigm`` (STPSF format: per-axis Gaussian sigma in arcsec)

    Returns
    -------
    dict[str, float]
        Jitter parameters with keys: ``jitter_major``, ``jitter_minor``,
        ``jitter_position_angle`` [all in milliarcseconds].
        Missing values default to 8.0 mas (major/minor) or 0.0 deg (angle).
    """

    # Default jitter sigma (8 mas) or extract from STPSF metadata
    axis_default = 8.0
    if "jitrsigm" in meta:
        axis_default = float(meta["jitrsigm"][0]) * 1000.0  # Convert arcsec to mas

    return {
        "jitter_major": float(meta.get("jitter_major", axis_default)),
        "jitter_minor": float(meta.get("jitter_minor", axis_default)),
        "jitter_position_angle": float(meta.get("jitter_position_angle", 0.0)),
    }


# Similar to psf.add_jitter
# RDMT addition: function to apply jitter to a generic PSF
def add_jitter_general(
    psf_data: np.ndarray,
    psf_jitter_params: dict,
    image_jitter_params: dict,
    pixel_scale: float,
    oversample: float,
) -> np.ndarray:
    """Adjust PSF grid stamps to match exposure jitter via Fourier deconvolution.

    The reference PSF contains jitter from reference conditions. This function
    deconvolves reference jitter and reconvolves with exposure jitter entirely
    in Fourier space, efficiently applying the transformation to all stamps.

    Parameters
    ----------
    psf_data :np.ndarray
        PSF grid stamps with shape (nstamp, ny, nx).
    psf_jitter_params : dict
    image_jitter_params : dict
        Image jitter parameters with keys:
        - "jitter_major" (mas)
        - "jitter_minor" (mas)
        - "jitter_position_angle" (deg)
    pixel_scale : float
        Detector pixel scale in arcsec/pixel.
    oversample : float
        Oversampling factor of the PSF grid stamps.
        Typically obtained from ``meta.oversampling`` or ``meta.oversample``.

    Returns
    -------
    np.ndarray
        PSF grid stamps adjusted for exposure jitter, with the same shape as ``psf_data``.

    """

    # Compute Fourier transform of jitter deconvolution/convolution kernel
    shape = psf_data.shape[-2:]
    stamp_pixel_scale = pixel_scale / oversample
    jitter_fft = _evaluate_gaussian_fft(image_jitter_params, shape, stamp_pixel_scale) / (
        _evaluate_gaussian_fft(psf_jitter_params, shape, stamp_pixel_scale)
    )

    # Apply jitter transformation in Fourier space
    stamp_fft = np.fft.rfft2(psf_data, axes=(-2, -1))
    convolved = np.fft.irfft2(stamp_fft * jitter_fft[None, ...], axes=(-2, -1), s=shape)
    return convolved


def conversion_factor_for_flux_density(dmodel: Any) -> float:
    """Compute conversion factor from DN/s to flux density (nJy/pixel).

    Multiplies photometry calibration (DN/s -> MJy/sr) by pixel solid angle
    to get flux density per pixel.

    Parameters
    ----------
    dmodel : ASDF-like object or roman_datamodels.DataModel
        Open Roman L2 data model containing photometry metadata.

    Returns
    -------
    float
        Conversion factor: (DN/s) × factor = nJy/pixel
    """
    # Extract photometry calibration constants depending on model type
    if isinstance(dmodel, asdf.AsdfFile):
        l2_to_sb = dmodel["roman"]["meta"]["photometry"]["conversion_megajanskys"]
        pixel_area = dmodel["roman"]["meta"]["photometry"]["pixel_area"] * u.sr
        sb_to_flux = (1.0 * (u.MJy / u.sr) * pixel_area).to("nJy")
    elif isinstance(dmodel, rdm.DataModel):
        l2_to_sb = dmodel.meta.photometry.conversion_megajanskys
        pixel_area = dmodel.meta.photometry.pixel_area * u.sr
        sb_to_flux = (1.0 * (u.MJy / u.sr) * pixel_area).to("nJy")

    return float(l2_to_sb * sb_to_flux.value)


def get_image_center_and_search_radius(wcs, shape):
    """Return image center and cone-search radius covering the detector footprint.

    Parameters
    ----------
    wcs : astropy.wcs.WCS or gwcs.wcs.WCS
        World coordinate system for the image.
    shape : tuple[int, int]
        2D image shape as ``(ny, nx)``.
    buffer_factor : float, optional
        Multiplicative safety factor applied to the detector-footprint radius.

    Returns
    -------
    tuple[float, float, float]
        ``(center_ra_deg, center_dec_deg, search_radius_deg)``.
    """
    # shape is (row,col) which corresponds to (y,x) in image coordinates
    ny, nx = shape

    # Compute field center in pixel and world coordinates
    xcen = (nx - 1) / 2.0
    ycen = (ny - 1) / 2.0
    center_coord = wcs.pixel_to_world(xcen, ycen)
    center_ra = center_coord.ra.deg
    center_dec = center_coord.dec.deg

    # Compute search radius enclosing all 4 detector corners
    corners_x = np.array([-0.5, nx - 1 + 0.5, nx - 1 + 0.5, -0.5])
    corners_y = np.array([-0.5, -0.5, ny - 1 + 0.5, ny - 1 + 0.5])
    corner_coords = wcs.pixel_to_world(corners_x, corners_y)
    max_sep_deg = float(np.max(center_coord.separation(corner_coords).to(u.deg).value))
    search_radius_deg = max_sep_deg

    return center_ra, center_dec, search_radius_deg


def get_gaia_catalog_for_l2data(af: Union[asdf.AsdfFile, dict], catalog="GAIADR3_S3"):
    """
    Query Gaia DR3 catalog via cone search covering the region of a Roman ASDF image.

    Parameters
    ----------
    af : asdf.AsdfFile or dict
        Open Roman ASDF object containing 'roman'/'meta' and 'roman'/'data'.
        or dict with center_ra, center_dec, search_radius and optionally epoch.
    buffer_factor : float, optional
        Multiplicative safety factor for the search radius (default is 1.0).
    filter_to_detector : bool, optional
        If True, adds pixel coordinates (x, y) and filters the table to only
        sources falling within the [0, nx-1] x [0, ny-1] detector footprint.

    Returns
    -------
    astropy.table.Table
        Astropy Table containing the Gaia DR3 sources within the field of view.
    """
    epoch = None
    filter_to_detector = False
    buffer_factor = 1.01

    if isinstance(af, asdf.AsdfFile):
        filter_to_detector = True
        # Extract relevant metadata and WCS from the ASDF file
        wcs = af["roman"]["meta"]["wcs"]
        data = af["roman"]["data"]
        center_ra, center_dec, search_radius_deg = get_image_center_and_search_radius(wcs, data.shape)

        # Extract observation epoch (decimal year) for proper motion correction
        if "mid_time" in af["roman"]["meta"]["exposure"]:
            epoch = af["roman"]["meta"]["exposure"]["mid_time"].decimalyear
        elif "start_time" in af["roman"]["meta"]["exposure"]:
            epoch = af["roman"]["meta"]["exposure"]["start_time"].decimalyear

    else:
        center_ra = af["center_ra"]
        center_dec = af["center_dec"]
        search_radius_deg = af["search_radius"]
        if "epoch" in af:
            epoch = af["epoch"]

    gaia_table = get_gaia_catalog(
        right_ascension=center_ra,
        declination=center_dec,
        epoch=epoch,
        search_radius_deg=search_radius_deg * buffer_factor,
        catalog=catalog,
    )

    # Filter strictly to the detector pixel bounds and attach (x, y)
    if filter_to_detector and len(gaia_table) > 0:
        x_pix, y_pix = wcs.world_to_pixel_values(gaia_table["ra"], gaia_table["dec"])
        # shape is (row,col) which corresponds to (y,x) in image coordinates
        in_bounds = (x_pix >= 0) & (x_pix <= data.shape[1] - 1) & (y_pix >= 0) & (y_pix <= data.shape[0] - 1)
        gaia_table = gaia_table[in_bounds]
        gaia_table["x"] = x_pix[in_bounds]
        gaia_table["y"] = y_pix[in_bounds]

    return gaia_table


def crossmatch_with_gaia(af: asdf.AsdfFile, cat: QTable):
    """Cross-match to Gaia a given source catalog for a given image.

    Args:
    - af: ASDF file object containing the WCS of the image.
    - cat: Astropy Table for sources identified in the image, with source positions (x_psf, y_psf).

    Returns:
    - matched_cat: Astropy Table containing the sources from the input catalog that have been cross-matched with Gaia.

    """
    wcs = af["roman"]["meta"]["wcs"]
    if type(cat["x_psf"]) is not u.Quantity:
        raise TypeError("cat['x_psf'] must be an astropy Quantity")
    if type(cat["y_psf"]) is not u.Quantity:
        raise TypeError("cat['y_psf'] must be an astropy Quantity")

    cat_coords = wcs.pixel_to_world(cat["x_psf"], cat["y_psf"])

    gaia_cat = get_gaia_catalog_for_l2data(af)
    gaia_coords = SkyCoord(ra=gaia_cat["ra"] * u.deg, dec=gaia_cat["dec"] * u.deg)

    # Cross-match: find the nearest source in cat for each Gaia star
    idx, sep, _ = gaia_coords.match_to_catalog_sky(cat_coords)
    # For future: to screen out points having multiple close matches
    # idx2, sep2, _ = match_coordinates_sky(gaia_coords, cat_coords, nthneighbor=2)
    # cond = (sep < max_sep) & (sep2 > 2 * max_sep)

    # Filter matches within a matching threshold, currently we select all
    max_sep = np.inf * u.arcsec
    matched_mask = sep < max_sep

    # print(f"Number of matched sources: {np.sum(matched_mask)}")

    # Build the matched catalog with Gaia information attached
    matched_cat = cat[idx[matched_mask]].copy()
    matched_cat["angsep_gaia"] = sep[matched_mask].to(u.arcsec)
    for key in ["source_id", "ra", "dec"]:
        matched_cat[f"gaia_{key}"] = gaia_cat[key][matched_mask].copy()
    for key in ["phot_g_mean_mag", "phot_bp_mean_mag", "phot_rp_mean_mag"]:
        matched_cat[key] = gaia_cat[key][matched_mask].copy()

    stats = gaia_astrometry_stats(matched_cat)
    print(stats)

    return matched_cat


def gaia_astrometry_stats(matched_cat: Table, max_sep=0.5 * u.arcsec):
    """Compute astrometry statistics for a catalog matched to Gaia.

    Args:
    - matched_cat: Astropy Table for sources matched to Gaia, with 'angsep_gaia' column.
    - max_sep: Maximum separation for considering a match (default 0.5 arcsec).

    Returns:
    - stats: Dictionary with astrometry statistics (rmse, medae, mae, n_sources, f_missed).
    """
    cond = matched_cat["angsep_gaia"] < max_sep.to(u.arcsec)
    temp = matched_cat["angsep_gaia"][cond]

    stats = {}
    prefix = "angsep_gaia_"
    stats[f"{prefix}n_sources"] = temp.size
    stats[f"{prefix}f_missed"] = 1.0 - (temp.size / len(matched_cat)) if len(matched_cat) > 0 else np.nan
    if temp.size > 0:
        stats[f"{prefix}rmse"] = np.sqrt(np.mean(temp**2))
        stats[f"{prefix}medae"] = np.median(temp)
        stats[f"{prefix}mae"] = np.mean(temp)
    else:
        stats[f"{prefix}rmse"] = np.nan
        stats[f"{prefix}medae"] = np.nan
        stats[f"{prefix}mae"] = np.nan

    return stats


def build_psf_model(
    asdf_file: asdf.AsdfFile,
    config: dict,
    psf_option: str,
    apply_jitter: bool = True,
    psf_fwhm_pix: float = 0.1,
) -> GriddedPSFModel | CircularGaussianPRF:
    """Construct the PSF model passed to ``PSFPhotometry``.

    Parameters
    ----------
    asdf_file : asdf.AsdfFile
        The ASDF file containing the Roman WFI exposure metadata and data.
    config : dict
        Configuration dictionary containing directory paths/buckets.
        Required keys: ['RDMT_SPIRE_RDATA_DIR']
    psf_option : str
        PSF model option, either "crds" or "stpsf" or 'gaussian'.
    apply_jitter : bool, optional
        Whether to apply jitter to the PSF model (default is True).
    psf_fwhm_pix : float, optional
        Full-width at half-maximum (FWHM) of the Gaussian PSF in pixels (default is 0.1).


    Returns
    -------
    photutils.psf.GriddedPSFModel | photutils.psf.CircularGaussianPRF
        Detector/filter-matched gridded ePSF model loaded from the Roman
        STPSF ASDF file when available. When ``apply_jitter`` is enabled the
        stamps are re-convolved from the reference jitter to the jitter
        reported for this exposure.
    """
    try:
        # Load PSF model based on configuration
        if psf_option == "crds":
            # Fetch ePSF reference file from CRDS
            psf_path = get_crds_reference_file(asdf_file, "epsf", "roman")
            psf_model = datamodels.open(psf_path)
            psf_model.psf = add_jitter_general(
                psf_model.psf,
                psf_jitter_params=get_jitter_params(psf_model.meta),
                image_jitter_params=get_jitter_params(asdf_file["roman"]["meta"].get("guide_star", {})),
                pixel_scale=0.11,  # Roman WFI detector pixel scale in arcsec
                oversample=psf_model.meta["oversample"],
            )
            psf_model = get_gridded_psf_model_crds(psf_model)

        elif psf_option == "stpsf":
            # Load from local STPSF grid file
            detector = str(asdf_file["roman"]["meta"]["instrument"]["detector"])
            filt = str(asdf_file["roman"]["meta"]["instrument"]["optical_element"])
            psf_key = f"{detector}_{filt}"
            #file_object = Path(config["RDMT_SPIRE_RDATA_DIR"]) / "psfs/wfi_stpsf_grid_n9o4.asdf"
            file_object=aws_utils.load_file_object(config["RDMT_SPIRE_RDATA_DIR"], "psfs/wfi_stpsf_grid_n9o4.asdf")
            with asdf.open(file_object) as af:
                psf_data = np.asarray(af[psf_key]["data"], dtype=float)
                psf_meta = dict(af[psf_key]["meta"])

            # Adjust PSF jitter to match this exposure (if enabled)
            if apply_jitter:
                psf_data = add_jitter_general(
                    psf_data,
                    psf_jitter_params=get_jitter_params(psf_meta),
                    image_jitter_params=get_jitter_params(asdf_file["roman"]["meta"].get("guide_star", {})),
                    pixel_scale=0.11,  # Roman WFI detector pixel scale in arcsec
                    oversample=psf_meta["oversampling"],
                )
            psf_model = GriddedPSFModel(NDData(data=psf_data, meta=psf_meta))
        else:
            psf_model = CircularGaussianPRF(flux=1.0, fwhm=psf_fwhm_pix)
            # raise ValueError(f"Unsupported PSF option: {psf_option}")

        return psf_model
    except Exception:
        # Raise error if PSF loading fails (fallback analytical PSF could be added here)
        raise RuntimeError("Failed to load gridded PSF model")


# Same as romancal:psf.get_gridded_psf_mode except for the ability to work with asdf files
# At this stage the use of romand_datamodels.datamodels.EpsfRefModel vs ASDF-like object
# is undecided so kept here.
def get_gridded_psf_model_crds(psf_ref_model, focus=0, spectral_type=1):
    """Generate a gridded PSF model from a Roman CRDS ePSF reference file.

    Creates a GriddedPSFModel from an ePSF reference file, selecting a specific
    focus position and spectral type. Reference files contain multiple focus
    positions (in-focus and defocused) and spectral types (A0V, G2V, M6V).

    Parameters
    ----------
    psf_ref_model : roman_datamodels.datamodels.EpsfRefModel or ASDF-like object
        ePSF reference data model containing PSF stamp grids, positions, and metadata.
    focus : int, optional
        Index of the focus position to use (0=in-focus, default=0).
    spectral_type : int, optional
        Index of the spectral type to use (0=A0V, 1=G2V, 2=M6V; default=1).

    Returns
    -------
    photutils.psf.GriddedPSFModel
        Gridded PSF model for the specified focus and spectral type,
        with oversampled PSF stamps at detector positions.
    """
    # Extract PSF data and metadata depending on model type
    meta = OrderedDict()
    if isinstance(psf_ref_model, rdm.EpsfRefModel):
        psf_images = psf_ref_model.psf[focus, spectral_type, :, :, :].copy()
        # get the central position of the cutouts in a list
        psf_positions_x = psf_ref_model.meta.pixel_x
        psf_positions_y = psf_ref_model.meta.pixel_y
        oversample = psf_ref_model.meta.oversample
        meta["jitter_major"] = psf_ref_model.meta.jitter_major
        meta["jitter_minor"] = psf_ref_model.meta.jitter_minor
        meta["jitter_position_angle"] = psf_ref_model.meta.jitter_position_angle
    else:
        psf_images = psf_ref_model["roman"]["psf"][focus, spectral_type, :, :, :].copy()
        # get the central position of the cutouts in a list
        psf_positions_x = psf_ref_model["roman"]["meta"]["pixel_x"]
        psf_positions_y = psf_ref_model["roman"]["meta"]["pixel_y"]
        oversample = psf_ref_model["roman"]["meta"]["oversample"]
        meta["jitter_major"] = psf_ref_model["roman"]["meta"]["jitter_major"]
        meta["jitter_minor"] = psf_ref_model["roman"]["meta"]["jitter_minor"]
        meta["jitter_position_angle"] = psf_ref_model["roman"]["meta"]["jitter_position_angle"]

    position_list = []
    for index in range(len(psf_positions_x)):
        position_list.append([psf_positions_x[index], psf_positions_y[index]])

    # Check PSF normalization to detect format version
    # Old format: PSF stamps sum to 1 (not integrated over pixel)
    # New format: PSF stamps sum to oversample^2 (integrated over pixel)
    is_old_format = np.median(np.sum(psf_images, axis=(-1, -2))) < oversample**2 / 2
    if is_old_format:
        log.info("Integrating old-format PSF stamps over the native pixel scale.")
        # Convolve with pixel response kernel to account for pixel integration
        pixel_response_kernel = Box2DKernel(width=oversample)
        for i in range(psf_images.shape[0]):
            psf = psf_images[i, :, :]
            im = convolve(psf, pixel_response_kernel) * oversample**2
            psf_images[i, :, :] = im

    meta["grid_xypos"] = position_list
    meta["oversampling"] = oversample
    nd = NDData(psf_images, meta=meta)
    model = GriddedPSFModel(nd)

    return model


# Only for standalone (independent of romancal) local background estimation
def aperture_local_background(data, positions, rin, rout):
    """Estimate local background and uncertainty using annular apertures.

    Measures the background level and uncertainty around source positions using
    sigma-clipped (3-sigma) statistics within circular annuli.

    Copied from `romancal` (with minor modifications)
    https://github.com/spacetelescope/romancal/blob/main/romancal/source_catalog/_aperture.py : _aperture_background

    Parameters
    ----------
    data : np.ndarray
        2D science image array.
    positions : np.ndarray
        Nx2 array of (x, y) source positions in pixel coordinates.
    rin : float
        Inner radius of the circular annulus in pixels.
    rout : float
        Outer radius of the circular annulus in pixels.

    Returns
    -------
    bkg_median : np.ndarray
        Sigma-clipped median background value at each source position.
    bkg_median_err : np.ndarray
        Standard error of the background median for each position, computed as
        ``sqrt(pi / (2 * N)) * sigma``, where N is the number of pixels in
        the annulus and sigma is the standard deviation.

    Notes
    -----
    This implementation provides the same functionality as photutils'
    ``LocalBackground`` but also retains the annulus pixel counts needed
    to estimate background uncertainties.
    """

    # Create circular annulus apertures at source positions
    bkg_aper = CircularAnnulus(positions, r_in=rin, r_out=rout)
    bkg_aper_masks = bkg_aper.to_mask(method="center")
    sigclip = SigmaClip(sigma=3.0)  # 3-sigma clipping for outlier rejection

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        warnings.simplefilter("ignore", category=AstropyUserWarning)

        nvalues = []
        bkg_median = []
        bkg_std = []
        for mask in bkg_aper_masks:
            bkg_data = mask.get_values(data)
            values = sigclip(bkg_data, masked=False)
            nvalues.append(values.size)
            med = np.median(values)
            std = np.std(values)
            bkg_median.append(med)
            bkg_std.append(std)

        nvalues = np.array(nvalues)
        bkg_std = u.Quantity(bkg_std)
        bkg_median = u.Quantity(bkg_median)
        bkg_median_err = np.sqrt(np.pi / (2.0 * nvalues)) * bkg_std

    return bkg_median, bkg_median_err
