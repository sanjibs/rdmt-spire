# PhotoAstro Monitor

The PhotoAstro monitor remeasures point sources in a Roman WFI Level 2 (L2) calibrated image using aperture photometry and PSF fitting. It uses the matching Level 4 (L4) source catalog to identify point-like sources and provide starting positions, then summarizes the remeasured encircled-energy radii and image-to-Gaia angular separations. Metrics are computed separately for bright and faint sources to make changes in image quality and photometric behavior easier to distinguish across the usable dynamic range.

Unlike the Source Catalog monitor, which characterizes properties already in the pipeline catalog, this monitor independently performs measurements on the L2 pixel data. It does not currently report direct differences between the remeasured fluxes and the catalog fluxes; flux measurements support source binning and derived metrics, while the active monitored properties are listed below.

## Inputs and configuration

`PhotoAstroMonitor` accepts an open Roman ASDF L2 file and a configuration dictionary. It uses the following paths and input data:

| Input | Use |
|---|---|
| `RDMT_SPIRE_L4_DIR` | Directory or S3 location containing the L4 source catalog. The catalog filename is derived from `roman.meta.filename` by replacing `_cal.asdf` with `_cat.parquet`. |
| `RDMT_SPIRE_LDATA_DIR` | Photometric-parameter ECSV tables and `metric_thresholds/photometric_metric_thresholds.ecsv`. |
| `RDMT_SPIRE_RDATA_DIR` | Needed by the optional STPSF grid choice in the lower-level pipeline; the monitor currently uses its default CRDS PSF choice. |
| L2 ASDF fields | `roman.data`, `roman.err`, `roman.dq`, `roman.meta.wcs`, `roman.meta.photometry.conversion_megajanskys`, `roman.meta.photometry.pixel_area`, filter metadata, exposure time, and the source filename. |

The source catalog must provide `label`, `x_psf`, `y_psf`, `x_centroid`, `y_centroid`, and `is_extended`. The catalog loader and image measurement pipeline are implemented in `aperture_psf.py`; photometric calibration and magnitude limits are supplied by `utilities/techinfo.py`.

## Measurement workflow

1. **Load and prepare the image.** The pipeline reads the matching Parquet catalog and extracts science, error, and DQ arrays from the ASDF file. Pixels with non-finite science/error values, non-positive errors, or the `DO_NOT_USE` DQ bit are masked. A smooth two-dimensional background is estimated using sigma-clipped medians on a 1000-pixel mesh and a 3-by-3 median filter, then subtracted from the science image. If the mesh size is not usable for the image, the estimator falls back to a single box spanning the image.
2. **Select point sources and starting coordinates.** Rows are retained when `is_extended` is false and `x_psf`, `y_psf` are finite and less than the image width and height. The initial measurement positions are `x_centroid`, `y_centroid`. The selection currently has no explicit lower-bound test on `x_psf` or `y_psf`.
3. **Calibrate and refine positions.** Science and error values are converted from DN/s to nJy per pixel using the L2 photometric conversion and pixel area. By default, positions are refined with a local two-dimensional Gaussian centroid fit; non-finite, implausibly shifted, or out-of-image centroid results fall back to their catalog centroid.
4. **Measure aperture and PSF photometry.** Circular-aperture fluxes and errors are measured at radii 0.1, 0.2, 0.4, and 0.8 arcsec. A local 2.4–2.8 arcsec annulus background estimate is also produced. The default PSF is the CRDS ePSF for the Roman observation; reference jitter is adjusted to the exposure's guide-star jitter, and PSF fitting uses 5-by-5-pixel fitting regions with sources grouped when separated by less than 5 pixels. The fitted positions and PSF flux/error are returned with the aperture results.
5. **Measure encircled-energy radii.** A curve of growth on the background-subtracted image estimates the radii containing 25%, 50%, and 75% of the measured source flux. The sampling radii are based on hard-coded F129 encircled-energy radii, scaled by the current filter's PSF FWHM relative to 0.106 arcsec, and capped at 2.8 arcsec. The radii are converted using a fixed 0.11 arcsec/pixel sampling scale and the local WCS pixel area. Thus, these are image measurements, but their sampling grid is not currently a filter-specific encircled-energy calibration.
6. **Assign magnitude bins.** AB magnitudes are calculated from the remeasured PSF-fit fluxes, not the catalog fluxes. Sources outside the modeled unsaturated-to-detectable range are excluded.
7. **Cross-match to Gaia.** For each magnitude bin, fitted positions are transformed through the L2 WCS and matched to Gaia sources covering the detector field. Angular separations are summarized as an astrometric diagnostic.

## Bright and faint source limits

The magnitude limits depend on the optical filter and exposure time. Filter zero points, effective PSF pixel counts, peak-pixel fractions, thermal background, and zodiacal background are read from the Roman technical-information tables under `RDMT_SPIRE_LDATA_DIR`.

The saturation limit uses a full-well charge of $C_{\rm sat}=120{,}000\,e^-$ and the filter's peak-flux fraction $f_{\rm peak}$:

$$
\alpha_{\rm sat}=\frac{C_{\rm sat}}{f_{\rm peak}t_{\rm exp}},\qquad
m_{\rm sat}=Z_R-2.5\log_{10}(\alpha_{\rm sat}),
$$

where $Z_R$ is the filter zero point and $t_{\rm exp}$ is the exposure time. The faint limit is based on a target signal-to-noise ratio of 50 and the modeled background rate

$$
f_{\rm bkg}=2f_{\rm min\text{-}zodi}+f_{\rm thermal}.
$$

The faint source rate is obtained by solving

$$
{\rm SNR}=\frac{f_{\rm source}t_{\rm exp}}
{\sqrt{t_{\rm exp}\left(n_{\rm eff}f_{\rm bkg}+f_{\rm source}\right)}}
$$

for $f_{\rm source}$ at SNR = 50, then converting it to magnitude with the filter zero point. This is a model from the reference parameter tables, not a noise estimate measured from the current image. Flux is converted to AB magnitude as $m_{\rm AB}=-2.5\log_{10}(f_{\rm Jy})+8.9$.

Let $m_{\rm mid}=(m_{\rm sat}+m_{\rm faint})/2$. The bins are defined as

$$
\begin{aligned}
\mathrm{bright:} &\quad m_{\rm sat}<m_{\rm AB}\leq m_{\rm mid},\\
\mathrm{faint:} &\quad m_{\rm mid}<m_{\rm AB}<m_{\rm faint}.
\end{aligned}
$$

Because lower magnitudes are brighter, sources at or brighter than saturation, at or beyond the faint limit, and any sources outside the two intervals are omitted.

## Monitored properties and statistics

The active properties are configured in `constants/photoastro_constants.py`. Each is calculated independently in the bright and faint bins.

| Property | Unit | Interpretation | Statistics emitted per bin |
|---|---|---|---|
| `ee25_radius` | arcsec | Radius enclosing 25% of measured source flux | `n_sources`, `f_outliers`, `median`, `nmad`, `mean`, `std` |
| `ee50_radius` | arcsec | Radius enclosing 50% of measured source flux | `n_sources`, `f_outliers`, `median`, `nmad`, `mean`, `std` |
| `ee75_radius` | arcsec | Radius enclosing 75% of measured source flux | `n_sources`, `f_outliers`, `median`, `nmad`, `mean`, `std` |
| `angsep_gaia_custom` | arcsec | Separation between a fitted image position and its matched Gaia position | `n_sources`, `f_outliers`, `medae`, `rmse`, `mae` |

For the three radius distributions, the monitor drops non-finite values and identifies outliers using five one-sided percentile deviations from the median, based on the 15.86th and 84.14th percentiles. The outlier fraction is reported, and the remaining values are used for distribution statistics. The normalized median absolute deviation is

$$
{\rm NMAD}=1.4826\,\operatorname{median}\left(|x-\operatorname{median}(x)|\right).
$$

For Gaia separations, values outside 0–0.5 arcsec are counted as outliers and removed before the absolute-separation summaries (`medae`, `rmse`, and `mae`) are calculated. `n_sources` counts finite values before outlier removal for both property types.

The code also calculates three aperture-flux ratios (`aper02/aper01`, `aper04/aper02`, and `aper08/aper04`) and the ratio of measured to theoretical PSF flux error. These are currently inactive properties: they are not enabled in `photoastro_constants.PROPERTIES` and therefore do not produce monitor metric cards or evaluations.

## Evaluation and interpretation

Thresholds are loaded from `metric_thresholds/photometric_metric_thresholds.ecsv`. For each property, a bin with fewer than 10 finite values fails its `n_sources` evaluation. With at least 10 values, the monitor evaluates only the radius median and NMAD, or the Gaia median absolute separation (`medae`), against the applicable threshold range expanded by three estimated standard errors:

$$
z_{\min}-3\epsilon_z < z < z_{\max}+3\epsilon_z.
$$

For a radius distribution, the standard errors use its NMAD $s$ and count $N$: $\epsilon_{\rm median}=s/\sqrt{N}$ and $\epsilon_{\rm NMAD}=s/\sqrt{2(N-1)}$. Gaia `medae` uses the generalized-chi-squared property's dispersion estimate and its configured two degrees of freedom. The thresholds currently include general ranges of 0–0.5 arcsec for radius medians, 0–1.0 arcsec for radius NMADs, and 0–0.5 arcsec for Gaia `medae`; these values are explicitly placeholders for refinement with calibration data.

Only the named statistics are threshold-tested. Other cards retain their default evaluation state; in particular, a `True` flag on an untested statistic is not evidence that it passed a calibrated threshold. Non-finite statistics are also skipped by the threshold comparison. Interpret the evaluation flags together with the source count and statistic value.

## Implementation notes

- Gaia matching queries the Gaia DR3 S3 catalog over the detector footprint (with a 1% radius buffer), applies proper motion correction to present epoch, then finds the nearest fitted image-catalog position for each Gaia source. The match cutoff is currently unlimited, so large separations may be included; the 0.5 arcsec outlier cut is applied later when computing Gaia statistics. This is not a constrained one-to-one match in which every image source is independently matched.
- Aperture and curve-of-growth measurements are made after global background subtraction. The local annular background is reported, but is not subtracted again from the aperture fluxes. The PSF fitter receives the invalid-pixel/DQ mask; the aperture measurement helper does not receive that mask.
- The pipeline's default PSF path requires CRDS access and a matching ePSF reference. Although the lower-level pipeline supports alternate PSF models, `PhotoAstroMonitor` currently instantiates it with default settings.
