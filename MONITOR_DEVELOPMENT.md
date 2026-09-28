# Monitor Development Tracking

This file tracks the development status and prioritization of monitors across the project.

## Table of Contents
- [**Implemented**](#implemented) : Monitors that have been fully implemented and merged into the codebase.
- [**Prioritized**](#prioritized) : Monitors that have been prioritized for development based on their importance and impact.
- [**Not Prioritized**](#not-prioritized) : Monitors that have been suggested but not yet prioritized for development.
- [**Deferred**](#deferred) : Monitors that were previously considered but have been deferred for future consideration.

## Executive Summary

| Monitor | Implementation Stage |
| --- | --- |
| [Astrometry Monitor](#monitor-astrometry) | Implemented |
| [1/f Noise Monitor](#monitor-noise-1f) | Implemented |
| [Guide Window Monitor](#monitor-guide-window) | Implemented |
| [Source Catalog Monitor](#monitor-source-catalog) | Implemented |
| [Photometry-Astrometry Monitor](#monitor-photometry) | Implemented |
| [Pixel Monitor](#monitor-pixel-monitor) | Implemented |
| [Flat Field Flux Monitor](#monitor-flat-field-flux) | Prioritized (P2) |
| [Jump Monitor](#monitor-jump) | Prioritized (P2) |
| [Persistence Monitor](#monitor-persistence) | Prioritized (P3) |
| [Standard Star Monitor](#monitor-standard-star) | Not Prioritized |
| [Astrometry+ Monitor](#monitor-astrometry-plus) | Not Prioritized |
| [First-read Anomaly Monitor](#monitor-first-read-anomaly) | Not Prioritized |
| [Background Matching Monitor](#monitor-background-matching) | Deferred |

---

<details>
<summary><strong>Instructions to Update This File</strong></summary>

### Updating This File:
1. When implementing a new monitor, move it from "Prioritized" to "Implemented"
2. Link to the corresponding submodule in `rdmt_spire/monitors/`
3. Reference all related GitHub issues
4. Include brief notes on status and any relevant technical details
5. Keep the Executive Summary table synchronized (monitor name, stage, and link)
6. If monitor headings are renamed, update the corresponding anchor IDs and summary links

### Linking Conventions:
- **Submodule links:** Use relative paths to `rdmt_spire/monitors/[monitor_name]/`
- **Issue links:** Use format `[#ISSUE_ID](https://github.com/[org]/[repo]/issues/[ID])`
- **Documentation:** Link to README files within each monitor submodule

</details>

---


### Implemented

<a id="monitor-astrometry"></a>
#### Astrometry Monitor
- **Submodule:** [astrometry](rdmt_spire/monitors/astrometry/)
- **Related Issues/PRs:** 
  - [PR 12](https://github.com/spacetelescope/rdmt-spire-archival/pull/12) - Initial implementation
  - [PR 22](https://github.com/spacetelescope/rdmt-spire-archival/pull/22) - Catalog matching improvement
- **Description:** Checks the astrometric solution computed by the Exposure Pipeline by comparing cataloged star locations to the observed pixel positions. See [astrometry README](rdmt_spire/monitors/astrometry/README.md) for details.

<a id="monitor-noise-1f"></a>
#### 1/f Noise Monitor
- **Submodule:** [noise_1f](rdmt_spire/monitors/noise_1f/)
- **Related Issues/PRs:**
  - [PR 13](https://github.com/spacetelescope/rdmt-spire-archival/pull/13) - Initial implementation
- **Description:** Checks to ensure the 1/f noise has been sufficiently removed from the Level 2 files. See [noise_1f README](rdmt_spire/monitors/noise_1f/README.md) for details.

<a id="monitor-guide-window"></a>
#### Guide Window Monitor
- **Submodule:** [guide_window](rdmt_spire/monitors/guide_window/)
- **Related Issues/PRs:**
  - [ISSUE 31](https://github.com/spacetelescope/rdmt-spire-archival/issues/31) - Requirements and design
  - [PR 35](https://github.com/spacetelescope/rdmt-spire-archival/pull/35) - Initial implementation
- **Description:** The three main goals for this monitor are to 1) track the jitter in each detector for each exposure,  2) identify failed guide star acquisitions (especially those that did not cause the entire exposure to fail) so they can be further reviewed, 3) compare the observed guide star counts to the predicted counts to help tighten the dim and bright bounds of future observations. See [guide_window README](rdmt_spire/monitors/guide_window/README.md) for details.

<a id="monitor-pixel-monitor"></a>
#### Pixel Monitor
- **Submodule:** [pixel](rdmt_spire/monitors/pixel/)
- **Related Issues/PRs:**
  - [ISSUE 8](https://github.com/spacetelescope/rdmt-spire/issues/8) - Requirements and design
  - [PR 16](https://github.com/spacetelescope/rdmt-spire/pull/16) - Initial implementation
- **Description:** This is a large catch-all monitor for any types of pixel that would be helpful to trend. This currently includes, but is not limited to the number of saturated pixels, min/mean/max/std/percentiles of the pixel values, pixel scale at the corners and center of the detector, and includes these values over the entire detector as well as different regions (e.g. 4 corners and a central region).

<a id="monitor-source-catalog"></a>
#### Source Catalog Monitor
- **Submodule:** [source_catalog](rdmt_spire/monitors/source_catalog/)
- **Related Issues/PRs:**
  - [ISSUE 4](https://github.com/spacetelescope/rdmt-spire/issues/4) -  Requirements and design
  - [PR 13](https://github.com/spacetelescope/rdmt-spire/pull/13) - Initial implementation  
- **Description:** Using the L4 detector catalog files, this monitor bins point sources by brightness and then calculates statistics like mean, median, std, and NMAD for quantities like sharpness, roundness, ellipticity, and encircled energy. See [source_catalog README](rdmt_spire/monitors/source_catalog/README.md) for details.

<a id="monitor-photoastro"></a>
-#### Photo-Astrometry Monitor
- **Submodule:** [photoastro](rdmt_spire/monitors/photoastro/)
- **Related Issues/PRs:**
  - [ISSUE 5](https://github.com/spacetelescope/rdmt-spire/issues/5) -  Requirements and design
  - [PR 15](https://github.com/spacetelescope/rdmt-spire/pull/15) - Initial implementation  
- **Description:** 
This monitor conducts PSF and aperture photometry on an L2 image, starting with candidate sources presented in the L4 source catalog.
The source positions are perturbed slightly and re-measured to account for uncertainties and potential systematic errors in the initial catalog positions.
While this monitor can in principle reproduce some of the quantities calculated in the source catalog monitor, it will primarily focus on quantities not captured in the source catalog monitor (in the same magnitude bins as the source catalog monitor and using similar statistical measures). It is designed to  serve as a basis for expanding upon more advanced photometric analyses in the future. Presently it estimates the radiii at typical encircled energy fractions (2-axis FWHM to be done in future work). See [photoastro README](rdmt_spire/monitors/photoastro/README.md) for details.

---

### Prioritized

<a id="monitor-photometry"></a>
#### Photometry Monitor
- **Related Issues:**
  - [ISSUE 5](https://github.com/spacetelescope/rdmt-spire/issues/5)
- **Priority Level:** P1 - High priority
- **Estimated Work:** M (involves expanding on the astrometry monitor to include PSF quantification)
- **Description:** This monitor will be combined with the astrometry monitor to use the best-fit PSFs to estimate the typical encircled energy at several pixel radii, 2-axis FWHM, and so on for sources in the same bins as the source catalog monitor. The monitor will focus on quantities not captured in the source catalog monitor.

<a id="monitor-flat-field-flux"></a>
#### Flat Field Flux Monitor
- **Related Issues:**
  - [ISSUE 6](https://github.com/spacetelescope/rdmt-spire/issues/6)
- **Priority Level:** P2 - Medium priority
- **Estimated Work:** S 
- **Description:** WFI Flat L2s will have the RCS illumination level in the metadata. This monitor will compare the illumination values to the brightness seen in the images.

<a id="monitor-jump"></a>
#### Jump Monitor
- **Related Issues:**
  - [ISSUE 7](https://github.com/spacetelescope/rdmt-spire/issues/7)
- **Priority Level:** P2 - Medium priority
- **Estimated Work:** M (adding clustering analysis to identify snowballs and other extended sources adds work)
- **Description:** Using the DQ array (and the resultant DQ array if available) from the L2 file to count the number of jumps. Ideally this would include a clustering analysis to distinguish CR events from snowballs and other extended sources (e.g. moving objects).

<a id="monitor-persistence"></a>
#### Persistence Monitor
- **Related Issues:**
  - [ISSUE 9](https://github.com/spacetelescope/rdmt-spire/issues/9)
- **Priority Level:** P3 - Low priority
- **Estimated Work:** XL (scope still unclear and involves new infrastructure)
- **Description:** We want to track persistence in the L2 files, but the exact implementation of this is still unclear. We will need to wait for commissioning to clarify which model we should use and how helpful the different possible products will be to users.

---

### Not Prioritized

<a id="monitor-standard-star"></a>
#### Standard Star Monitor
- **Estimated Work:** L
- **Description:** Monitor the performance of standard stars across the sky. 100s-1000s of "standard" stars could be identified over the entire sky. If those stars land on one of the detectors for a Roman observation, thorough analysis of those stars could be executed (e.g. estimating the enclosed energy fraction or checking the photometry estimate / zero-points).This would require storing a database of these standard stars with expected values for the different filters.

<a id="monitor-astrometry-plus"></a>
#### Astrometry+ Monitor
- **Estimated Work:** L
- **Description:** It would be important to include measurements of the plate scale (average by WFI detector and also the average over the full 18 detectors) after velocity aberration is accounted for, as well as the 2 skew terms of the linear transformations to Gaia sources, on top of statistics of the offsets along 2 directions which are already in the implementation plan.

<a id="monitor-first-read-anomaly"></a>
#### First-read Anomaly Monitor
- **Estimated Work:** S
- **Description:** WFI18 has a significant first-read anomaly. There are high residuals between the first and subsequent read (up to -100 DN) in the bottom rows of detector WFI18. The parameters for the current models vary between exposures and are possibly temperature dependent and thus would benefit from trending analysis.

---

### Deferred

<a id="monitor-background-matching"></a>
#### Background Matching Monitor
- **Reason:** It is too difficult to accommodate as it is scene dependent and would be hard to do without AI or ML which is currently outside the scope of this project.
- **Description:** Compute mean background & peak-to-valley variation between L2 images and compare with known models.

---
