"""Roman WFI aperture and PSF photometry pipeline.

This module provides :class:`PhotometryCatalog`, which performs the full
source-photometry workflow for a Roman L2 ASDF image:

1. loads the matching source catalog,
2. selects point-like sources,
3. estimates and subtracts background,
4. measures multi-radius aperture fluxes,
5. fits PSF astrometry and photometry, and
6. returns a single Astropy table containing the required output columns.

Reusable utilities for CRDS lookups, PSF construction, jitter correction, and
flux conversion live in the ``utilities`` package.
"""

from __future__ import annotations

import logging
from abc import abstractmethod
from types import SimpleNamespace
from typing import Any

import asdf
import astropy.units as u
import numpy as np
from astropy.table import QTable, Table
from photutils.aperture import CircularAperture, aperture_photometry
from photutils.centroids import centroid_2dg, centroid_sources
from photutils.profiles import CurveOfGrowth
from photutils.psf import (
    PSFPhotometry,
    SourceGrouper,
)
from roman_datamodels.dqflags import pixel

from ...utilities import aws_utils, wcs_utils
from ...utilities.phot.aperture import ApertureCatalog
from ...utilities.phot.photoastro_utils import (
    aperture_local_background,
    build_psf_model,
    conversion_factor_for_flux_density,
    estimate_background,
)
from ...utilities.phot.psf import _PSFCatalog
from ...utilities.techinfo import RomanWFIPhotometricParameters

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

APER_OUTPUT_COLUMNS = [
    ("aper01_flux", "nJy"),
    ("aper01_flux_err", "nJy"),
    ("aper02_flux", "nJy"),
    ("aper02_flux_err", "nJy"),
    ("aper04_flux", "nJy"),
    ("aper04_flux_err", "nJy"),
    ("aper08_flux", "nJy"),
    ("aper08_flux_err", "nJy"),
    ("aper_bkg_flux", "nJy/arcsec2"),
    ("aper_bkg_flux_err", "nJy/arcsec2"),
]
EE_OUTPUT_COLUMNS = [
    ("ee25_radius", "arcsec"),
    ("ee50_radius", "arcsec"),
    ("ee75_radius", "arcsec"),
]
PSF_OUTPUT_COLUMNS = [
    ("x_psf", "pix"),
    ("x_psf_err", "pix"),
    ("y_psf", "pix"),
    ("y_psf_err", "pix"),
    ("psf_flux", "nJy"),
    ("psf_flux_err", "nJy"),
]


REQUIRED_OUTPUT_COLUMNS = APER_OUTPUT_COLUMNS + EE_OUTPUT_COLUMNS + PSF_OUTPUT_COLUMNS


class PhotometryCatalogPipeline:
    """Compute aperture and PSF photometry for point sources in Roman L2 images.

    This class orchestrates the complete photometry workflow: loading data,
    estimating background, computing multi-radius aperture fluxes, and running
    PSF fitting astrometry/photometry.
    """

    def __init__(
        self,
        asdf_file: asdf.AsdfFile,
        config: dict[str, str],
        init_scatter: float | None = None,
        refine_centroids: bool = True,
        pixel_area_is_constant: bool = False,
        psf_option: str = "crds",
    ) -> None:
        """Initialize a photometry pipeline for a single Roman L2 image.

        Parameters
        ----------
        asdf_file : asdf.AsdfFile
            Open Roman L2 data model exposing mapping access to the science image,
            error array, metadata, and quality flags.
        config : dict[str, str]
            Configuration dictionary with required paths such as
            Required keys: ['RDMT_SPIRE_L4_DIR', 'RDMT_SPIRE_RDATA_DIR'].
        init_scatter : float or None, optional
            Optional random centroid perturbation size in pixels used to test
            initialization robustness.
        refine_centroids : bool, optional
            If ``True``, refine detected centroid positions with local Gaussian
            centroiding before fitting.
        pixel_area_is_constant : bool, optional
            If ``True``, assume constant pixel area and use the nominal pixel
            scale instead of the WCS-derived per-pixel area map.
        psf_option : {'crds', 'stpsf'}, optional
            Source of the PSF model. ``'crds'`` loads the calibration PSF from
            CRDS; ``'stpsf'`` loads a local STPSF grid.

        Notes
        -----
        The instance immediately runs the full photometry workflow in
        :meth:`run` and stores the output catalog in ``self.catalog``.
        """

        self.asdf_file = asdf_file
        self.config = config
        # Configuration options
        self.psf_option = psf_option  # Use CRDS PSF reference files
        self.apply_jitter = True  # Re-jitter PSF to exposure conditions
        self.init_scatter = init_scatter
        self.pixel_area_is_constant = pixel_area_is_constant
        self.refine_centroids = refine_centroids
        self.scheme = "romancal"
        self.catalog = self.run()

    def run(self) -> Table:
        """Run the complete monitor workflow and return final source results.

        Workflow
        --------
        1. Derive and load the matching source catalog.
        2. Resolve coordinate and classification column names.
        3. Select point sources (``is_extended == False``).
        4. Extract image/error arrays and pixel scale from the L2 model.
        5. Estimate and subtract the image-wide 2D background.
        6. Compute aperture photometry features.
        7. Compute PSF-fit astrometry and photometry features.
        8. Merge identity columns + required output columns into one table.

        Returns
        -------
        astropy.table.Table
            Table with selected-source identity fields and all required
            monitor columns defined in ``REQUIRED_OUTPUT_COLUMNS``.
        """

        # Load the source catalog matching this image: 
        image_filename = str(self.asdf_file["roman"]["meta"]["filename"])
        # QTable not required, using Table instead
        file_object = aws_utils.load_file_object(self.config["RDMT_SPIRE_L4_DIR"], image_filename.replace('_cal.asdf', '_cat.parquet'))
        catalog = Table.read(file_object, format="parquet")



        # Extract image data, uncertainties, and quality mask
        data, err, mask = self._extract_image_error_mask()

        # Estimate and subtract 2D background if enabled
        background = estimate_background(data, box_size=1000, coverage_mask=mask).background
        data -= background

        # Compute the mean pixel scale from the pixel area map.
        self.pixel_area_map = wcs_utils.pixel_area_map(self.asdf_file["roman"]["meta"]["wcs"], data.shape)

        self.pixel_scale_mean = np.sqrt(np.mean(self.pixel_area_map.to(u.arcsec**2)))
        if self.pixel_area_is_constant:
            # self.pixel_scale = pixel_scale_old(self.asdf_file).value
            self.pixel_area_map = (np.full(data.shape, self.pixel_scale_mean.value**2, dtype=float) * u.arcsec**2).to(u.sr)

        # Select valid point sources (on-detector, finite coords, not extended)
        cond = (catalog["x_psf"] < data.shape[1]) & (catalog["y_psf"] < data.shape[0])
        cond = cond & (np.isfinite(catalog["x_psf"])) & (np.isfinite(catalog["y_psf"]))
        cond = cond & (~catalog["is_extended"])
        selected = catalog[cond]

        x_init = selected["x_centroid"].value.copy() * u.pix
        y_init = selected["y_centroid"].value.copy() * u.pix

        # Randomly perturb initial positions if init_scatter is specified
        if self.init_scatter is not None:
            noise_x = np.random.uniform(size=len(selected))
            temp = x_init.copy()
            x_init = x_init + (noise_x - 0.5) * 2.0 * self.init_scatter * u.pix
            cond = (x_init.to_value(u.pix) > data.shape[1]) | (x_init.to_value(u.pix) < 0)
            x_init[cond] = temp[cond]

            noise_y = np.random.uniform(size=len(selected))
            temp = y_init.copy()
            y_init = y_init + (noise_y - 0.5) * 2.0 * self.init_scatter * u.pix
            cond = (y_init.to_value(u.pix) > data.shape[0]) | (y_init.to_value(u.pix) < 0)
            y_init[cond] = temp[cond]

        # Convert from L2 units (DN/s) to flux density (nJy/pixel)
        factor = conversion_factor_for_flux_density(self.asdf_file) * u.nJy
        data = data * factor
        err = err * factor

        # Initialize output table with identity columns
        output = QTable()
        output["label"] = selected["label"].copy()

        # Return early if no sources were selected.
        if len(selected) == 0:
            return output

        wfi_properties = RomanWFIPhotometricParameters(self.config)
        optical_filter = str(self.asdf_file["roman"]["meta"]["instrument"]["optical_element"]).lower()
        psf_fwhm = wfi_properties.get_psf_fwhm(optical_filter)
        psf_fwhm_pix = psf_fwhm.to_value(u.arcsec) / self.pixel_scale_mean.to_value(u.arcsec)

        # Centroiding is necessary if initial positions are shifted or scatter is applied.
        if self.refine_centroids:
            obj = CentroidCatalog()
            obj.analyze(data, err, mask, x_init, y_init, self.pixel_area_map, psf_fwhm)
            obj.write(output, [("x_init", "pix"), ("y_init", "pix")])
        else:
            output["x_init"] = x_init.to(u.pix)
            output["y_init"] = y_init.to(u.pix)

        if self.scheme == "romancal":
            obj = ApertureCatalogRomancal()
        else:
            obj = ApertureCatalogRDMT()
        obj.analyze(data, err, mask, output["x_init"], output["y_init"], self.pixel_area_map)
        obj.write(output, APER_OUTPUT_COLUMNS)

        if self.scheme == "romancal":
            obj = PSFCatalogRomancal()
        else:
            obj = PSFCatalogRDMT()

        psf_model = build_psf_model(
            self.asdf_file,
            self.config,
            self.psf_option,
            apply_jitter=self.apply_jitter,
            psf_fwhm_pix=psf_fwhm_pix,
        )
        # psf_model = self._build_psf_model()
        obj.analyze(data, err, mask, output["x_init"], output["y_init"], self.pixel_area_map, psf_model)
        obj.write(output, PSF_OUTPUT_COLUMNS)

        obj = EECatalog()
        obj.analyze(data, err, mask, output["x_init"], output["y_init"], self.pixel_area_map, psf_fwhm)
        obj.write(output, EE_OUTPUT_COLUMNS)

        return output

    def _extract_image_error_mask(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract science image and uncertainty planes from the ASDF object.

        Workflow role
        -------------
        Extracts the science image, error array, and constructs a quality mask.
        Supplies photometry routines with same-shape ``data``, ``err``, and ``mask`` arrays.
        If no explicit error array exists, a fallback uncertainty is estimated
        from absolute image values.

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray, np.ndarray]
            ``(data, err, mask)`` float arrays with identical shapes.

        Raises
        ------
        KeyError
            If no image data array is found.
        ValueError
            If extracted data and error arrays have different shapes.
        """

        # Extract science image data
        try:
            data = self.asdf_file["roman"]["data"]
        except Exception as exc:
            raise KeyError("Could not locate image data at af['roman']['data'].") from exc

        # Extract error array
        try:
            err = self.asdf_file["roman"]["err"]
        except Exception as exc:
            raise KeyError("Could not locate image error at af['roman']['err'].") from exc

        # Convert to float arrays
        data_arr = np.asarray(data, dtype=float)
        err_arr = np.asarray(err, dtype=float)

        # Validate array shapes match
        if data_arr.shape != err_arr.shape:
            raise ValueError(f"Data and error shape mismatch: data={data_arr.shape}, err={err_arr.shape}")

        # Create quality mask for invalid/non-finite pixels
        mask_arr = ~np.isfinite(data_arr) | ~np.isfinite(err_arr) | (err_arr <= 0)  # Negative or zero errors are invalid

        # Add data quality flags to mask
        dq = self.asdf_file["roman"]["dq"]
        if dq.shape != data_arr.shape:
            msg = f"dq shape {dq.shape} does not match " f"data shape {data_arr.shape}; expected a 2D DQ array."
            raise ValueError(msg)
        dq_mask = (dq & pixel.DO_NOT_USE) != 0
        mask_arr |= dq_mask

        return data_arr, err_arr, mask_arr


class CatalogCreator:
    def __init__(self) -> None:
        pass

    @abstractmethod
    def analyze(
        self,
        data: u.Quantity,
        err: u.Quantity,
        mask: np.ndarray,
        x: u.Quantity,
        y: u.Quantity,
        pixel_area_map: u.Quantity,
        meta: dict[str, Any] | None = None,
        frac: float = 1.0,
    ) -> None:
        """Computes aperture or psf photometry for sources.

        Parameters
        ----------
        data : astropy.units.Quantity[numpy.ndarray]
            2D science image.
        err : astropy.units.Quantity[numpy.ndarray]
            2D uncertainty image.
        mask : numpy.ndarray
            Boolean mask indicating valid pixels (True for valid, False for invalid).
        x : astropy.units.Quantity[numpy.ndarray]
            Source x coordinates in u.pix
        y : astropy.units.Quantity[numpy.ndarray]
            Source y coordinates in u.pix.
        pixel_area_map : astropy.units.Quantity[numpy.ndarray]
            2D map of pixel areas in u.sr, same shape as `data`.
        psf_fwhm : astropy.units.Quantity[numpy.ndarray]
            Full width at half maximum of the PSF in u.arcsec.
        meta : dict[str, Any] | None, optional
            Metadata associated with the observation, by default None.
        frac : float, optional
            Fraction of sources to analyze, by default 1.0.
        """
        pass

    def write(self, output_table: QTable, requested_properties: list[str]):
        """Writes the analysis results to the output table.

        Parameters
        ----------
        output_table : astropy.table.QTable
            Table to which the results will be written.
        requested_properties : list[tuple[str,str]]
            List of tuples (property_name, unit_name) specifying which properties to write to the table.
        """
        x_empty = np.asarray([], dtype=float)
        for key, kunit in requested_properties:
            output_table[key] = self.results.get(key, x_empty * u.Unit(kunit))
            if output_table[key].unit is None:
                output_table[key] = output_table[key] * u.Unit(kunit)


class ApertureCatalogRomancal(CatalogCreator):
    # Photometry aperture radii (arcsec)
    _RADII_ARCSEC = (0.1, 0.2, 0.4, 0.8)

    def analyze(self, data, err, mask, x, y, pixel_area_map, meta=None) -> None:
        self.results = {}
        if len(x) == 0:
            return

        xypos = np.column_stack([x.to_value(u.pix), y.to_value(u.pix)])

        # When no sources are provided return empty arrays with appropriate units
        if len(x) > 0:
            model = SimpleNamespace(data=data, err=err)
            catalog = ApertureCatalog(model=model, xypos_finite=xypos, pixel_area_map=pixel_area_map)

            # copy the aperture photometry results from the catalog to the results dictionary
            for radius_arcsec in self._RADII_ARCSEC:
                name = f"aper{int(radius_arcsec * 10):02d}"
                self.results[f"{name}_flux"] = getattr(catalog, f"{name}_flux")
                self.results[f"{name}_flux_err"] = getattr(catalog, f"{name}_flux_err")
            for key in ["aper_bkg_flux", "aper_bkg_flux_err"]:
                self.results[key] = getattr(catalog, key)


class ApertureCatalogRDMT(CatalogCreator):
    # Photometry aperture radii (arcsec)
    _RADII_ARCSEC = (0.1, 0.2, 0.4, 0.8)
    # Annulus radii for local background estimation (arcsec)
    _ANNULUS_ARCSEC = (2.4, 2.8)

    def analyze(self, data, err, mask, x, y, pixel_area_map, meta=None) -> None:
        """Compute multi-radius aperture fluxes and uncertainties for sources.

        See base class `CatalogCreator` for more details.
        """

        self.results = {}
        if len(x) == 0:
            return

        xq = u.Quantity(x, copy=False)
        yq = u.Quantity(y, copy=False)
        pixel_scale_mean = np.sqrt(np.mean(pixel_area_map.to(u.arcsec**2)))

        # Convert source coordinates to arrays
        xypos = np.column_stack([xq.to_value(u.pix), yq.to_value(u.pix)])

        # Create apertures at specified radii
        apertures = [CircularAperture(xypos, radius_arcsec / pixel_scale_mean.to_value(u.arcsec)) for radius_arcsec in self._RADII_ARCSEC]

        # Run aperture photometry for all radii
        cat = aperture_photometry(data, apertures, error=err, mask=mask)

        # Organize results
        self.results: dict[str, np.ndarray] = {}

        # Extract fluxes from catalog and store in results dictionary
        for i, aperture in enumerate(apertures):
            radius_arcsec = self._RADII_ARCSEC[i]
            cat_key = f"aperture_sum_{i}"
            res_key = f"aper{int(radius_arcsec * 10):02d}_flux"
            self.results[res_key] = cat[cat_key]
            cat_key = f"aperture_sum_err_{i}"
            res_key = f"aper{int(radius_arcsec * 10):02d}_flux_err"
            self.results[res_key] = cat[cat_key]

        # Run aperture local background
        rin = self._ANNULUS_ARCSEC[0] / pixel_scale_mean.to_value(u.arcsec)
        rout = self._ANNULUS_ARCSEC[1] / pixel_scale_mean.to_value(u.arcsec)
        bkg_per_pix, bkg_per_pix_err = aperture_local_background(data, xypos, rin, rout)

        # Store background results
        pixel_area = wcs_utils.pixel_area_at(pixel_area_map, xypos[:, 0], xypos[:, 1]).to(u.arcsec**2)
        self.results["aper_bkg_flux"] = bkg_per_pix / pixel_area
        self.results["aper_bkg_flux_err"] = bkg_per_pix_err / pixel_area


class PSFCatalogRomancal(CatalogCreator):

    def analyze(self, data, err, mask, x, y, pixel_area_map, psf_model, meta=None) -> None:
        """Run PSF fitting with the romncal based implementation.

        See base class `CatalogCreator` for more details.
        """
        self.results = {}

        xypos = np.column_stack((x.to_value(u.pix), y.to_value(u.pix)))
        requested_properties = ["x_psf", "y_psf", "psf_flux", "x_psf_err", "y_psf_err", "psf_flux_err"]

        if len(x) == 0:
            return

        model = SimpleNamespace(data=data, err=err, meta=meta)
        catalog = _PSFCatalog(
            model=model,
            psf_ref_model=psf_model,
            xypos=xypos,
            mask=mask,
            requested_properties=requested_properties,
        )

        for name in requested_properties:
            if hasattr(catalog, name):
                self.results[name] = getattr(catalog, name)


class PSFCatalogRDMT(CatalogCreator):

    def analyze(self, data, err, mask, x, y, pixel_area_map, psf_model, meta=None) -> None:
        """Run PSF fitting with the current photutils API and a PSF model.

        See base class `CatalogCreator` for more details.
        """
        self.results = {}
        if len(x) == 0:
            return

        # Initialize PSF fitting table with source positions
        init = QTable()
        init["x_0"] = x.to_value(u.pix)
        init["y_0"] = y.to_value(u.pix)

        # The PSF model is provided by the caller and should be used directly.

        # Set up PSF photometry with grouping and fitting configuration
        grouper = SourceGrouper(min_separation=5)  # pixels; fit nearby sources together
        fit_shape = (5, 5)  # Fitting region size
        psf_photometry = PSFPhotometry(
            psf_model=psf_model,
            fit_shape=fit_shape,
            aperture_radius=fit_shape[0],
            grouper=grouper,
        )

        # Run PSF fitting on all sources
        phot_table = psf_photometry(data=data, error=err, init_params=init, mask=mask)

        # Map PSFPhotometry output columns to required monitor columns
        requested_properties = [
            ("x_psf", "x_fit"),
            ("y_psf", "y_fit"),
            ("psf_flux", "flux_fit"),
            ("x_psf_err", "x_err"),
            ("y_psf_err", "y_err"),
            ("psf_flux_err", "flux_err"),
        ]

        # Extract results, applying units where necessary
        for key_res, key_cat in requested_properties:
            if key_cat in phot_table.colnames:
                self.results[key_res] = phot_table[key_cat]


class EECatalog(CatalogCreator):
    # Encircled-energy fractions to solve for via curve of growth
    _EE_FRACTIONS = (0.25, 0.5, 0.75)

    # Annulus radii for local background estimation (arcsec)
    _ANNULUS_ARCSEC = (2.4, 2.8)

    def analyze(self, data, err, mask, x, y, pixel_area_map, psf_fwhm, meta=None, frac=1.0) -> None:
        """Estimate per-source encircled-energy radii from a curve of growth.

        See base class `CatalogCreator` for more details.
        """
        self.results = {}
        if len(x) == 0:
            return

        xv = x.to_value(u.pix)
        yv = y.to_value(u.pix)
        pixel_scale_fixed_arcsec = 0.11

        # Sample the curve of growth at radii where
        # energy rises in steps, capped at the annulus outer radius
        # ee radius for F129 filter in arcsec, computed from crds reference file as follows
        # ee_targets = [0.1, 0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.85, 0.95]
        # filter='F129'
        # apcorr_ref = photastro_utils.get_crds_reference_file(self.asdf_file, "apcorr", "roman")
        # appcorr_file = asdf.open(apcorr_ref)
        # ee_radii = appcorr_file['roman']['data'][filter]['ee_radii']
        # ee_fractions = appcorr_file['roman']['data'][filter]['ee_fractions']
        # ee_spline_inv = PchipInterpolator(ee_fractions, ee_radii)
        # ee_radii_arcsec_F129 = ee_spline_inv(ee_targets)
        ee_radii_arcsec_f129 = np.array(
            [
                0.02745977,
                0.03936521,
                0.05186293,
                0.06383583,
                0.07126782,
                0.08037714,
                0.09550239,
                0.15168068,
                0.1739305,
                0.19182791,
                0.21277176,
                0.49865768,
                1.68357505,
            ]
        )
        psf_fwhm_f129 = 0.106 * u.arcsec

        radii = ee_radii_arcsec_f129 * psf_fwhm.to_value(u.arcsec) / psf_fwhm_f129.to_value(u.arcsec)
        radii = np.append(radii, np.inf)
        radii = np.unique(np.clip(radii, a_min=None, a_max=self._ANNULUS_ARCSEC[1]))
        radii_px = radii / pixel_scale_fixed_arcsec

        pixel_scale_arcsec = np.sqrt(wcs_utils.pixel_area_at(pixel_area_map, xv, yv).to(u.arcsec**2)).value
        for frac in self._EE_FRACTIONS:
            key = f"ee{int(frac * 100):02d}_radius"
            self.results[key] = np.full(len(xv), np.nan, dtype=float) * u.arcsec

        ind = np.arange(len(xv))
        if frac < 1.0:
            np.random.shuffle(ind)
            ind = ind[: int(len(ind) * frac)]

        # for i, (xi, yi) in enumerate(zip(xv, yv)):
        for i in ind:
            xi, yi = xv[i], yv[i]
            if not (np.isfinite(xi) and np.isfinite(yi)):
                continue
            try:
                cog = CurveOfGrowth(data, (xi, yi), radii_px, error=err, mask=mask)
                cog.normalize()
                radii_at_ee = cog.calc_radius_at_ee(self._EE_FRACTIONS)
            except ValueError:
                continue

            for frac, radius_px in zip(self._EE_FRACTIONS, radii_at_ee):
                key = f"ee{int(frac * 100):02d}_radius"
                self.results[key].value[i] = radius_px * pixel_scale_arcsec[i]


class CentroidCatalog(CatalogCreator):

    def analyze(self, data, err, mask, x, y, pixel_area_map, psf_fwhm, meta=None) -> None:
        """Refine catalog positions with local two-dimensional Gaussian fits.

        See base class `CatalogCreator` for more details.
        """
        self.results = {}
        if len(x) == 0:
            return

        x_init = x.to_value(u.pix)
        y_init = y.to_value(u.pix)

        pixel_scale_mean = np.sqrt(np.mean(pixel_area_map.to(u.arcsec**2)))
        psf_fwhm_px = psf_fwhm.to_value("arcsec") / pixel_scale_mean.to_value("arcsec")
        position_scatter_px = 5 * psf_fwhm_px

        half_size = max(3, int(np.ceil(position_scatter_px)))
        box_size = 2 * half_size + 1
        x_centroid, y_centroid = centroid_sources(
            data,
            x_init,
            y_init,
            box_size=box_size,
            mask=mask,
            centroid_func=centroid_2dg,
            error=err,
        )

        shift = np.hypot(x_centroid - x_init, y_centroid - y_init)
        valid = (
            np.isfinite(x_centroid)
            & np.isfinite(y_centroid)
            & (shift <= 2 * position_scatter_px)
            & (x_centroid >= 0)
            & (x_centroid < data.shape[1])
            & (y_centroid >= 0)
            & (y_centroid < data.shape[0])
        )
        self.results = {
            "x_init": np.where(valid, x_centroid, x_init) * x.unit,
            "y_init": np.where(valid, y_centroid, y_init) * y.unit,
        }
