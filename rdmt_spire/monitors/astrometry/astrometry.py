import logging
import os
from typing import Any, Dict, Tuple

import asdf
import astropy.modeling
import astropy.units as u
import gwcs
import healpy
import numpy as np
import numpy.typing as npt
import stpsf
import yaml
from astropy.coordinates import SkyCoord
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.nddata import NDData
from astropy.stats import gaussian_sigma_to_fwhm
from astropy.table import Table, vstack
from photutils.background import LocalBackground, MADStdBackgroundRMS
from photutils.detection import DAOStarFinder
from photutils.psf import GriddedPSFModel, PSFPhotometry, SourceGrouper

from ...constants.astrometry_thresholds import (
    ASTROMETRIC_OFFSET_THRESHOLD,
    NUM_ASTROMETRIC_SOURCES,
)
from ...utilities import aws_utils
from ..monitor_base import BaseMonitor

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class AstrometryMonitor(BaseMonitor):
    """
    Astrometry monitor derived from class BaseMonitor
    Tested with asdf files from https://github.com/spacetelescope/roman_notebooks/blob/main/notebooks/aperture_photometry/aperture_photometry.ipynb
    Files: ['r0003201001001001004_0001_wfi01_f106_cal.asdf', 'full_catalog.ecsv']
    From: 's3://stpubdata/roman/nexus/soc_simulations/tutorial_data/'
    """

    def __init__(self, asdf_file: asdf.AsdfFile, datadir: str):
        super().__init__(asdf_file)
        self.monitor_name = "astrometry"
        self.log.append(f"{self.monitor_name}: initialized")
        self.datadir = datadir
        self._set_sigma_psf()
        self.params={}
        self.params["epsf_image_name"] = "astrometry/wfi_stpsf_grid_n9o4.asdf"
        self.params["catalog_name"] = "astrometry/input_catalogs"

        # photometric band used to select stars from input catalog
        self.params['photometric_band'] = 'F106'
        # Magnitude upper limit, below which to select stars for estimating astrometric error
        self.params['magnitude_threshold'] = 17.0

        # psf files were generated from stpsf by
        # self._write_stpsf_tofile(os.path.join(self.datadir,"astrometry/wfi_stpsf_grid_n9o4.asdf"))

    def _set_sigma_psf(self):
        """
        width of psf in pixel
        """
        self.sigma_psf = {}
        pixel_size_arcsec = 0.11  # arcsec
        self.sigma_psf["F062"] = 0.058 / pixel_size_arcsec
        self.sigma_psf["F087"] = 0.073 / pixel_size_arcsec
        self.sigma_psf["F106"] = 0.087 / pixel_size_arcsec
        self.sigma_psf["F129"] = 0.106 / pixel_size_arcsec
        self.sigma_psf["F158"] = 0.128 / pixel_size_arcsec
        self.sigma_psf["F184"] = 0.146 / pixel_size_arcsec
        self.sigma_psf["F213"] = 0.169 / pixel_size_arcsec
        self.sigma_psf["F146"] = 0.105 / pixel_size_arcsec

    def _get_sigma_psf(self, filter):
        """
        width of psf in pixel units
        """
        return self.sigma_psf[filter]

    def calculate_metrics(self):
        """
        Calculates astrometry metrics. 
        Following two metrics are added to the DataCard
        astrometric_offset: RMS separation [arcsec] between estimated and true values based on input catalog 
        num_astrometric_source: Number of sources used to compute the astrometric offset
        """
        offset, n_sources = self._astrometry_computation()
        offset = offset.to(u.arcsec)
        self.log.append(f"RMS offset is {offset:.4f} computed over {n_sources} sources")

        # ---------------------------------------------------------
        # Store results in a specific format (name:str, quantity:Any, unit:str)
        self.append_data("astrometric_offset", offset.value, str(offset.unit))
        self.append_data("num_astrometric_sources", n_sources, "")

    def _write_stpsf_tofile(self, filename: str):
        """
        Write psf computed by stpsf to local file. psf is computed for each detector and filter
        We use a 3x3 gridded psf. For further details see
        https://stpsf.readthedocs.io/en/latest/psf_grids.html
        https://photutils.readthedocs.io/en/stable/api/photutils.psf.GriddedPSFModel.html
        """
        wfi = stpsf.roman.WFI()
        tree = {}
        for filter in ["F062", "F087", "F106", "F129", "F146", "F158", "F184", "F213"]:
            for i in range(1, 19):
                wfi.detector = f"WFI{i:02d}"
                wfi.filter = filter
                psf_grid = wfi.psf_grid(
                    num_psfs=9, all_detectors=False, verbose=True, oversample=4
                )
                tree[f"{wfi.detector}_{wfi.filter}"] = {
                    "data": psf_grid.data,
                    "meta": psf_grid.meta,
                }
        af = asdf.AsdfFile(tree)
        af.write_to(filename)
        self.log.append(f"Wrote psf file {filename}")

    def evaluate_metrics(self):
        """
        Evaluate astrometry metrics against pre-defined thresholds.
        """
        astro_offset = self.get_data("astrometric_offset")
        offset_eval = astro_offset < ASTROMETRIC_OFFSET_THRESHOLD

        n_sources = self.get_data("num_astrometric_sources")
        n_source_eval = n_sources > NUM_ASTROMETRIC_SOURCES

        # Store results in the correct location
        self.add_evaluation("astrometric_offset", offset_eval)
        self.add_evaluation("num_astrometric_sources", n_source_eval)

    def read_healpix_catalog(self, bucket_name: str, key_name: str):
        """
        Read input catalog split across multiple parquet files based on healpix indices
        The healpix related parameters are read from meta.yaml file.
        The mapping of program number to input catalog is hard coded for the time being.

        Parameters
        ----------
        bucket_name : str
            path to the S3 bucket (or local directory) that is storing the file
        key_name : str
            subfolder pointing to the base directory of the input ctalogs
        """        
        filename = self.asdf_file["roman"]["meta"]["filename"]
        target_ra = self.asdf_file["roman"]["meta"]["wcsinfo"]["ra_ref"]
        target_dec = self.asdf_file["roman"]["meta"]["wcsinfo"]["dec_ref"]


        # Multiple program_no could be using same catalog
        # so we use a dict to map the program_no to the input_catalog directory
        program_no = int(filename[1:6])
        program_key_names = {32: "p00032", 555: "p00555"}
        program_key_name = os.path.join(key_name, program_key_names[program_no])

        # Read input_catalog formating details
        params = yaml.safe_load(
            aws_utils.load_file_object(bucket_name, os.path.join(program_key_name, "meta.yaml"))
        )
        nside = params["nside"]
        coord = SkyCoord(
            ra=target_ra * u.degree, dec=target_dec * u.degree, frame="icrs"
        )
        if params["frame"] == "galactic":
            coord = coord.galactic
            vec = healpy.ang2vec(coord.l.value, coord.b.value, lonlat=True)
        elif params['frame'] == 'equatorial':
            vec=healpy.ang2vec(coord.ra.value, coord.dec.value, lonlat=True)
        else:
            raise RuntimeError('Unrecognized Coordinate frame')

        # A factor of 1.01 is sufficient to enclose all WFI pixels
        radius = 1.1 * np.radians(0.125/np.sqrt(2))
        if nside > 0:
            ind = healpy.query_disc(nside, vec, radius, nest=params["nest"])
        else:
            ind = [0]

        # load and concatenate the tables
        table = []
        for i in ind:
            if aws_utils.file_exists(
                bucket_name, os.path.join(program_key_name, f"cat-{i:d}.parquet")
            ):
                file_object = aws_utils.load_file_object(
                    bucket_name, os.path.join(program_key_name, f"cat-{i:d}.parquet"), mode="rb"
                )
                table_cur = Table.read(file_object, format="parquet")
                distance = astropy.coordinates.angular_separation(
                    np.radians(table_cur["ra"]),
                    np.radians(table_cur["dec"]),
                    np.radians(target_ra),
                    np.radians(target_dec),
                )
                table_cur = table_cur[
                    (distance < radius) & (table_cur["type"] == "PSF")
                ]
                table.append(table_cur)

        if len(table) == 0:
            raise RuntimeError("Input catalog or stars in it not found")

        return vstack(table)

    def _astrometry_computation(self) -> Tuple[float, int]:
        """
        Astrometry monitor computation
        """
        # Should probably be changed in future to throw an error
        # and implement a test with real simulated data
        if self.asdf_file.uri is None:
            self.log.append("No Asdf file provided, returning dummy values")
            return 0.02 * u.arcsec, 20

        # load image
        l2_image = self.asdf_file["roman"]["data"]
        l2_image_error = self.asdf_file["roman"]["err"]
        l2_image_mask = np.bool(self.asdf_file["roman"]["dq"])
        wcs = self.asdf_file["roman"]["meta"]["wcs"]
        sca = self.asdf_file["roman"]["meta"]["instrument"]["detector"]
        filter = self.asdf_file["roman"]["meta"]["instrument"]["optical_element"]
        self.log.append(f"Roman WFI L2 image (SCA {sca}_{filter}) loaded.")

        # load ePSF
        # In future this will need to be updated to use the ePSF reference file or
        # another method to get filter/detector dependent PSFs using the file metadata.
        # Currently using file created by self.write_stpsf_tofile(self.datadir+"wfi_stpsf_grid_n9o4.asdf")
        # Also, needs to be read from an s3 bucket rather than a local file, for example with
        # psf_image_content = load_s3_object(bucket_name = epsf_image_bucket_name, key_name = epsf_image_key_name)
        psf_model_content = aws_utils.load_file_object(self.datadir, self.params["epsf_image_name"])

        af = asdf.open(psf_model_content)
        nd = NDData(af[f"{sca}_{filter}"]["data"], meta=af[f"{sca}_{filter}"]["meta"])
        psf_model = GriddedPSFModel(nd)

        # load input_catalog and initialize guesses
        cat = self.read_healpix_catalog(self.datadir, self.params["catalog_name"])

        self.log.append("Source catalog loaded.")

        init_params = self._initialize_guesses(cat, wcs)
        self.log.append("Using bright stars to initialize photometry.")

        # astrometry step
        # Rather than running source detection form scratch we run with initial guesses set
        # by setting keyword init_params. This is because we are only in astrometric
        # precision of stars for which we know the astrometry.
        photutils_pipeline = self._get_photutils_pipeline(
            l2_image, psf_model, sigma_psf=self._get_sigma_psf(filter)
        )
        photometric_table = photutils_pipeline(
            l2_image, error=l2_image_error, mask=l2_image_mask, init_params=init_params
        )

        # compute offsets
        offset, n_sources = self._solve_offsets(photometric_table, wcs)

        return (offset, n_sources)

    def _solve_offsets(
        self,
        phot: Table,
        wcs: gwcs.wcs.WCS,
        max_sep=1.0 * u.arcsec,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Returns the RMS offsets and number of offsets.
        The wcs is used to convert between detector and sky coordinates.
        """

        cat_coords = SkyCoord(*wcs(phot["x_init"], phot["y_init"]), unit="deg")
        fit_coords = SkyCoord(*wcs(phot["x_fit"], phot["y_fit"]), unit="deg")
        separations = fit_coords.separation(cat_coords)
        # We may need to  adopt a more robust statistics like MAD
        # or remove outliers in future with
        # separation=separation[separation<max_sep]

        return ((separations**2).mean() ** 0.5), len(separations)

    def _initialize_guesses(self, cat: Table, wcs: gwcs.wcs.WCS) -> Table:
        """
        Use source catalog for initial guesses on photometry step.
        We convert catalog equatorial coordinates to pixel coordinates (x,y) in the L2 image frame.
        Catalog flux are in maggie. For magnitude m, maggie=10.0^{-0.4*m}.

        """

        if self.params['magnitude_threshold'] is not None:
            #  Catalog flux are in maggie, so we convert to magnitude, maggie=10.0^{-0.4*m}.
            cond = (-2.5 * np.log10(cat[self.params['photometric_band']])) < self.params['magnitude_threshold']
            cat = cat[cond].copy()

        x_cat, y_cat = wcs.invert(cat["ra"], cat["dec"])
        on_detector_mask = (x_cat > 0) & (x_cat < 4088) & (y_cat > 0) & (y_cat < 4088)

        stars = cat[on_detector_mask & (cat["type"] == "PSF")]

        self.log.append(
            f"Using {len(stars)} stars for PSF fitting with mag < {self.params['magnitude_threshold']}"
        )

        x0, y0 = wcs.invert(stars["ra"], stars["dec"])

        init_guesses = Table([x0, y0], names=["x_0", "y_0"])

        return init_guesses

    def _get_photutils_pipeline(
        self,
        l2_image: npt.ArrayLike,
        psf_model: astropy.modeling.Model,
        sigma_psf: float = 1.0,
    ):
        """
        Photutils pipeline, based on the guide at
        https://photutils.readthedocs.io/en/stable/user_guide/psf.html
        and implementation of psf photometry in the romancal pipeline
        https://github.com/spacetelescope/romancal/blob/main/romancal/source_catalog/psf.py
        """

        background_rms = MADStdBackgroundRMS()
        std = background_rms(l2_image)

        # romancal uses a threshold of 0,
        # we set it higher to filter out low significance peaks.
        finder = DAOStarFinder(
            fwhm=sigma_psf * gaussian_sigma_to_fwhm,
            threshold=50 * std,
            sharplo=0.2,
            sharphi=1.0,
            roundlo=-1.0,
            roundhi=1.0,
            peakmax=None,
        )

        # separation of 5 pix based on romancal but we scale for sigma_psf
        grouper = SourceGrouper(5.0 * sigma_psf)

        # For the time being based on romancal,
        # but, in future, worth investigation the optimum choice.
        local_bkg = LocalBackground(10, 30)

        fitter = LevMarLSQFitter()

        # Based on romancal, probably can be set smaller or even scale with sigma_psf
        fit_shape = (15, 15)

        # Alternative is to use IterativePSFPhotometry with maxiters=1.
        # But given the we know the stars for which we want the answers, we do not
        # use the iterative option.
        photometry = PSFPhotometry(
            finder=finder,
            grouper=grouper,
            psf_model=psf_model,
            fitter=fitter,
            localbkg_estimator=local_bkg,
            aperture_radius=fit_shape[0],
            fit_shape=fit_shape,
        )
        return photometry
