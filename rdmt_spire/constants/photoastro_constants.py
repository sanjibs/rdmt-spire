"""Thresholds for source catalog monitor metrics.

These thresholds are intended for metric names ending in ``_median`` and ``_std``
for both bright and faint source bins. Values are placeholders and should be
refined with calibration data.
"""
import numpy as np

PROPERTIES = [
    ("ee25_radius", "arcsec", 'dist1d',[np.nan,np.nan]),
    ("ee50_radius", "arcsec", 'dist1d',[np.nan,np.nan]),
    ("ee75_radius", "arcsec", 'dist1d',[np.nan,np.nan]),
    # ("flux_ratio_aper02_aper01_custom", "", 'dist1d',[np.nan,np.nan]),
    # ("flux_ratio_aper04_aper02_custom", "", 'dist1d',[np.nan,np.nan]),
    # ("flux_ratio_aper08_aper04_custom", "", 'dist1d',[np.nan,np.nan]),
    # ("flux_err_ratio_psf_theory_custom", "", 'dist1d',[np.nan,np.nan]),
    ("angsep_gaia_custom", "arcsec", 'genchi2',[0,0.5]),
]
SUFFIX1 = ["_bright", "_faint"]

