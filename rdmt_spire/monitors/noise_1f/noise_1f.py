import logging

import asdf
import numpy as np
import scipy.fft as spfft
from astropy.stats import sigma_clip

from ...constants.noise_1f_constants import (
    AMP_WIDTH,
    GW_PAD,
    N_AMPS,
    PIXEL_READ_FREQ,
    POWER_RATIO_THRESHOLD,
)
from ..monitor_base import BaseMonitor

logger = logging.getLogger(__name__)

class Noise1fMonitor(BaseMonitor):
    """
    Monitor derived from class BaseMonitor for assessing residual 1/f noise in a Roman WFI L2 image via an
    amplifier-wise power-spectrum metric.

    The monitor computes a power ratio that compares low-frequency power
    (below cutoff_freq) to high-frequency "white noise" power (above
    cutoff_freq). A lower ratio indicates more successful 1/f noise removal.

    Notes
    -----
    * If amp="all", the metric is computed per amplifier and averaged.
    * Time-series construction assumes bottom-to-top row order and pads each
      row by GW_PAD pixels to account for end-of-row scan delay.
    """
    def __init__(self,
                 asdf_file: asdf.AsdfFile,
                 amp="all",
                 cutoff_freq=1100,
                 verbose=False):
        """
        Initialize the Noise1fMonitor monitor.

        Parameters
        ----------
        asdf_file : asdf.AsdfFile
            Open ASDF file containing a Roman WFI L2 image in
            asdf_file.tree["roman"]["data"]
        amp : "all" or int, optional
            Amplifier selection. If "all", compute the metric for every
            amplifier (1 to N_AMPS) and average the results. If an integer,
            compute the metric for only that amplifier. Default is "all".
        cutoff_freq : float, optional
            Frequency in Hz defining the boundary between low-frequency
            (1/f-dominated) and high-frequency (white-noise-dominated) regions
            of the power spectrum. Default is 1100 Hz.
        verbose : bool, optional
            If True, print logging during processing.
        """

        super().__init__(asdf_file)

        self.monitor_name = 'noise_1f'
        self.log.append(f"{self.monitor_name}: initialized")

        self.verbose = verbose
        self.amp = amp
        self.cutoff_freq = cutoff_freq


    def _extract_amp(self, image, amp):
        """
        Extract a single amplifier slice from a full-frame L2 image.

        Parameters
        ----------
        image : numpy.ndarray
            Two-dimensional full-frame image array.
        amp : int
            One-indexed amplifier number.

        Returns
        -------
        numpy.ndarray
            Two-dimensional array corresponding to the selected amplifier with
            shape (rows, AMP_WIDTH).
        """
        min_col = (amp - 1) * AMP_WIDTH
        max_col = amp * AMP_WIDTH

        return image[:, min_col:max_col]


    def _put_in_time_series(self, data2d):
        """
        Convert a 2D amplifier image into a one-dimensional,
        time-ordered pixel readout stream.

        Pixels are ordered to approximate the detector read sequence by
        reversing the row order from bottom to top and flattening the result.
        Each row is padded by GW_PAD pixels to account for end-of-row guide-window scan
        delay.

        Parameters
        ----------
        data2d : numpy.ndarray
            Two-dimensional amplifier slice with shape (rows, AMP_WIDTH).

        Returns
        -------
        numpy.ndarray
            One-dimensional array representing the time-ordered pixel stream.
        """
        flipped = data2d[::-1]
        padded = np.pad(flipped, [(0, 0), (0, GW_PAD)])

        return padded.ravel()


    def _time_series_clean(self, data2d, sigma=5.0):
        """
        Clean an amplifier slice and convert it into a time-ordered pixel
        stream.

        Outlier pixels are identified using sigma-clipping and replaced with
        the mean of the clipped distribution. The cleaned data are then
        converted to a one-dimensional time series.

        Parameters
        ----------
        data2d : numpy.ndarray
            Two-dimensional amplifier slice.
        sigma : float, optional
            Sigma threshold used for outlier rejection.

        Returns
        -------
        numpy.ndarray
            Cleaned, one-dimensional time-ordered pixel stream.

        TODO Note: in Sarah's original code, she did:

            cutoff=10
            outliers = np.abs(data) > cutoff
            data[outliers] = np.mean(data[~outliers])

        Is sigma-clipping still an okay result? The cutoff might vary (and also I'm not sure where that number came from too)
        """
        data = data2d.astype(float)
        
        # Sigma-clip to remove outliers
        finite = np.isfinite(data) # remove nans and infs
        clipped = sigma_clip(data[finite], sigma=sigma, masked=True)
        mean_val = clipped.mean()

        bad = ~finite
        bad[finite] = clipped.mask
        data[bad] = mean_val

        # Detrending the data
        time_series = self._put_in_time_series(data)
        time_series = time_series - np.mean(time_series)

        return time_series


    def _perform_fft(self, ts):
        """
        Compute the one-sided power spectrum of a time-ordered pixel stream.

        Parameters
        ----------
        ts : numpy.ndarray
            One-dimensional time-ordered pixel data.

        Returns
        -------
        freq : numpy.ndarray
            One-sided frequency array in Hz.
        power : numpy.ndarray
            One-sided power spectrum corresponding to the input time series.

        Notes
        -----
        The power normalization is intended for relative comparisons between
        frequency ranges rather than for absolute power spectral density
        calibration.
        """
        n = ts.size

        fft_vals = spfft.rfft(ts)
        power = np.abs(fft_vals)**2 / n

        dt = 1.0 / PIXEL_READ_FREQ
        freq = np.fft.rfftfreq(n, d=dt)

        return freq, power
    

    def _compute_amp_metric(self, amp_slice, cutoff_freq):
        """
        Compute the low-frequency to high-frequency power ratio for a single
        amplifier.

        The metric is defined as the mean power below the cutoff frequency
        divided by the mean power above the cutoff frequency.

        Parameters
        ----------
        amp_slice : numpy.ndarray
            Two-dimensional amplifier slice.
        cutoff_freq : float
            Frequency in Hz separating low-frequency and high-frequency
            regions of the power spectrum.

        Returns
        -------
        power_ratio : float or None
            Ratio of low-frequency power to high-frequency power, or None if
            the computation fails.
        """
        try:
            ts = self._time_series_clean(amp_slice)
            freq, power = self._perform_fft(ts)

            idx = np.searchsorted(freq, cutoff_freq)

            if idx == 0:
                logger.warning('noise_1f: idx was 0: setting to 1 to avoid division by 0. Check cutoff_freq.')
                idx = 1
            elif idx == power.size:
                logger.warning('noise_1f: idx was equal to power.size: setting to power.size - 1 to avoid division by 0. Check cutoff_freq.')
                idx = power.size - 1

            white_noise = sum(power[idx:])

            if not np.isfinite(white_noise):
                raise ValueError("White noise estimate is invalid.")

            power_1f = sum(power[:idx])

            n_lowband = idx
            n_highband = power.size - n_lowband

            power_ratio = (power_1f / n_lowband) / (white_noise / n_highband)

            if not np.isfinite(power_ratio):
                raise ValueError("Computed power_ratio is NaN or inf.")

            return power_ratio

        except Exception as e:
            logger.error(f"{self.monitor_name}: Amp {self.amp} metric failed, {e}")
            return None


    def calculate_metrics(self):
        """
        Compute and store the 1/f power-ratio metric.

        Image data are read from the ASDF file and the power ratio is computed
        either for all amplifiers or for a single specified amplifier. The
        resulting metric is stored in the monitor datacard under the key
        "noise_1f_power_ratio".

        Notes
        -----
        When amp is set to "all", non-finite per-amplifier ratios are excluded
        from the average. If all amplifiers fail, the stored metric is NaN.
        The metric is dimensionless.
        """
        im = self.asdf_file.tree["roman"]["data"]

        if self.verbose:
            logger.info(f"{self.monitor_name}: running on {self.filename}")

        # Compute power ratio by averaging all amps FFT
        if self.amp == "all":

            ratios = []

            for amp in range(1, N_AMPS + 1):
                try:
                    amp_slice = self._extract_amp(im, amp)
                    pratio = self._compute_amp_metric(amp_slice=amp_slice,
                                                      cutoff_freq=self.cutoff_freq)

                    if np.isfinite(pratio):
                        ratios.append(pratio)
                    
                    else:
                        logger.warning(f"{self.monitor_name}: Amp {amp} produced invalid ratio.")
                
                except Exception as e:
                    logger.error(f"{self.monitor_name}: Failed to process amp {amp}, {e}")

            if len(ratios) == 0:
                logger.error(f"{self.monitor_name}: All amps failed! Setting ratio = NaN.")
                power_ratio = np.nan

            else:
                power_ratio = float(np.nanmean(ratios))

        # Compute power ratio for a single given amp
        else:
            amp_slice = self._extract_amp(im, int(self.amp))
            power_ratio = self._compute_amp_metric(amp_slice,
                                                   cutoff_freq=self.cutoff_freq)

        # Store power ratio result in datacard
        self.append_data("noise_1f_power_ratio", power_ratio, "")


    def evaluate_metrics(self):
        """
        Evaluate the stored 1/f power-ratio metric against POWER_RATIO_THRESHOLD.
        The resulting boolean evaluation is added to the monitor datacard.

        Notes
        -----
        If the stored metric is non-finite, the evaluation is set to False and
        an error is logged.
        """
        power_ratio = self.get_data("noise_1f_power_ratio")

        if not np.isfinite(power_ratio):
            logger.error(f"{self.monitor_name}: Power ratio is invalid, marking metric as failed.")
            evaluation = False

        else:
            evaluation = power_ratio < POWER_RATIO_THRESHOLD

        self.add_evaluation("noise_1f_power_ratio", evaluation)
