"""Thresholds for source catalog monitor metrics.

These thresholds are intended for metric names ending in ``_median`` and ``_std``
for both bright and faint source bins. Values are placeholders and should be
refined with calibration data.
"""
import numpy as np

PROPERTIES = [
    ("sharpness", "", 'dist1d',[np.nan,np.nan]),
    ("roundness1", "", 'dist1d',[np.nan,np.nan]),
    ("ellipticity", "", 'dist1d',[np.nan,np.nan]),
    ("fluxfrac_radius_50", "arcsec", 'dist1d',[np.nan,np.nan]),
    ("flux_ratio_aper02_aper01", "", 'dist1d',[np.nan,np.nan]),
    ("flux_ratio_aper04_aper02", "", 'dist1d',[np.nan,np.nan]),
    ("flux_ratio_aper08_aper04", "", 'dist1d',[np.nan,np.nan]),
    ("flux_err_ratio_psf_theory", "", 'dist1d',[np.nan,np.nan]),
    ("angsep_gaia", "arcsec", 'genchi2',[0,0.5]),
]
SUFFIX1 = ["_bright", "_faint"]


