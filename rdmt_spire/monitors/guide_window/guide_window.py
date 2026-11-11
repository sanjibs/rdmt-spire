import logging

import asdf
import numpy as np

from ...constants.dq import (
    NOT_SATURATED_STR,
    SATURATED_STR,
    SOMETIMES_SATURATED_STR,
)
from ...constants.guide_window_constants import (
    BACKGROUND_STD_THRESHOLD,
    COUNT_RATE_MATCHING_FACTOR,
    FGS_LIMIT_MATCHING_FACTOR,
    MEAN_NOISE_THRESHOLD,
    MEDIAN_BACKGROUND_THRESHOLD,
    NOISE_STD_THRESHOLD,
    NUM_BACKGROUND_OUTLIERS_THRESHOLD,
    NUM_COUNT_RATE_OUTLIERS_THRESHOLD,
    NUM_NOISE_OUTLIERS_THRESHOLD,
    RMS_CENTROID_OFFSET_THRESHOLD,
    RMS_X_JITTER_THRESHOLDS,
    RMS_Y_JITTER_THRESHOLDS,
    WSM_EDGE_LOCATIONS,
    WSM_GW_TOP,
)
from ..monitor_base import BaseMonitor

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class GuideWindowMonitor(BaseMonitor):
    """
    Monitor for evaluating guide-window performance metrics from Roman FGS data.

    Parameters
    ----------
    asdf_file : asdf.AsdfFile
        Open ASDF file containing Roman guide-window products and metadata.
    verbose : bool, optional
        If ``True``, enable additional runtime verbosity.
    """
    def __init__(self,
                 asdf_file: asdf.AsdfFile,
                 verbose=False):
        """
        Initialize the GuideWindowMonitor monitor.

        Parameters
        ----------
        asdf_file : asdf.AsdfFile
            Open ASDF file containing a Roman WFI L2 image in
            asdf_file.tree["roman"]["data"]
        verbose : bool, optional
            If True, print logging during processing.
        """

        super().__init__(asdf_file)

        self.monitor_name = 'guide_window'
        self.log.append(f"{self.monitor_name}: initialized")

        self.verbose = verbose

    def _process_centroids(self):
        """
        Compute centroid statistics from tracking centroids.

        Returns
        -------
        mean_centroid_positions : numpy.ndarray
            Mean centroid position ``[x, y]`` across all tracking frames.
        rms_x_centroid_error : float
            Root-mean-square radial centroid deviation from the mean position in the X direction, in pixels.
        rms_y_centroid_error : float
            Root-mean-square radial centroid deviation from the mean position in the Y direction, in pixels.
        """
        # TODO: make sure we are only looking at relevant tracking centroids
        mean_centroid_positions = np.mean(self.asdf_file.tree["roman"]["centroid"]["track_centroids"], axis=0)
        centroid_deviations = self.asdf_file.tree["roman"]["centroid"]["track_centroids"] - mean_centroid_positions
        rms_x_centroid_error = np.sqrt(np.mean(np.sum(centroid_deviations[0,:]**2)))
        rms_y_centroid_error = np.sqrt(np.mean(np.sum(centroid_deviations[1,:]**2)))

        return mean_centroid_positions, rms_x_centroid_error, rms_y_centroid_error
    

    def _get_wim_annulus_mask(self, arr: np.ndarray, edge_buffer: int, annulus_width: int) -> np.ndarray:
        """
        Build a square annulus mask used to estimate WIM background/noise.

        Parameters
        ----------
        arr : numpy.ndarray
            Input array whose last two dimensions represent image axes.
        edge_buffer : int
            Number of edge pixels excluded before starting the annulus.
        annulus_width : int
            Width of the annulus in pixels.

        Returns
        -------
        numpy.ndarray
            Boolean mask over the image plane where ``True`` selects annulus
            pixels.
        """
        inner_index = edge_buffer + annulus_width
        outer_index = edge_buffer
        mask = np.zeros(arr.shape[-2:], dtype=bool)
        # Create a ring by enabling a larger square then removing the center.
        mask[outer_index:-outer_index, outer_index:-outer_index] = True
        mask[inner_index:-inner_index, inner_index:-inner_index] = False
        return mask

    @property
    def wsm_edge_location(self) -> str:
        """
        WSM edge designation, either ``TOP`` or ``BOTTOM``.
        """
        if hasattr(self, "_cached_wsm_edge_location"):
            return self._cached_wsm_edge_location
        else:  
            element = self.asdf_file.tree['roman']['meta']['instrument']['optical_element']
            edge = self.asdf_file.tree['roman']['meta']['wsm_edge_used']
            self._cached_wsm_edge_location = WSM_EDGE_LOCATIONS.get((element.upper(), edge.upper()))
            if self._cached_wsm_edge_location is None:
                raise ValueError(f"Unexpected optical element ({element}) or WSM edge ({edge}) in metadata.")
            return self._cached_wsm_edge_location

    def _get_wsm_annulus_mask(self, arr: np.ndarray,
                              edge_buffer: int, 
                              annulus_width: int) -> np.ndarray:
        """
        Build a U-shaped annulus mask for WSM background/noise estimation.

        Parameters
        ----------
        arr : numpy.ndarray
            Input array whose last two dimensions represent image axes.
        edge_buffer : int
            Number of edge pixels excluded before starting the annulus.
        annulus_width : int
            Width of the annulus in pixels.

        Returns
        -------
        numpy.ndarray
            Boolean mask over the image plane where ``True`` selects background pixels while excluding the dispersed spectrum side.
        """
        # start with a rectangular WIM mask
        mask = self._get_wim_annulus_mask(arr, edge_buffer, annulus_width)

        # depending on the element and edge, remove the top or bottom of the mask to avoid the spectrum
        inner_index = edge_buffer + annulus_width
        if self.wsm_edge_location == WSM_GW_TOP:
            mask[-inner_index:, inner_index:-inner_index] = False
        else: # if self.wsm_edge_location == WSM_GW_BOTTOM:
            mask[:inner_index, inner_index:-inner_index] = False
        return mask


    def _estimate_background_and_noise(self, 
                                       annulus_width=2, 
                                       edge_buffer=2):
        """
        Estimate per-frame background and noise from an annulus around the star.

        Parameters
        ----------
        annulus_width : int, optional
            Width of the annulus in pixels.
        edge_buffer : int, optional
            Number of outer-edge pixels to skip before the annulus starts.

        Returns
        -------
        median_backgrounds : numpy.ndarray
            Per-frame median background level in DN.
        noise_backgrounds : numpy.ndarray
            Per-frame background standard deviation in DN.
        """
        if "WSM" in self.asdf_file.tree['roman']['meta']['fgs_modes_used'][0]:
            background_mask = self._get_wsm_annulus_mask(
            self.asdf_file.tree['roman']['track_data']['signal_resultants'],
            edge_buffer=edge_buffer,
            annulus_width=annulus_width)
        else: # good for all other modes (WIM, Standby, or defocused)
            background_mask = self._get_wim_annulus_mask(
            self.asdf_file.tree['roman']['track_data']['signal_resultants'],
            edge_buffer=edge_buffer,
            annulus_width=annulus_width)

        # TODO: make sure we are only looking at relevant tracking frames
        background_pixels = self.asdf_file.tree['roman']['track_data']['signal_resultants'][:, background_mask].astype(float) - self.asdf_file.tree['roman']['track_data']['pedestal_resultants'][:, background_mask].astype(float)

        median_backgrounds = np.median(background_pixels, axis=1)
        noise_backgrounds = np.std(background_pixels, axis=1)

        return median_backgrounds, noise_backgrounds


    def check_saturation(self, saturation_threshold=65535):
        """
        Classify saturation behavior across tracking frames.

        Parameters
        ----------
        saturation_threshold : int, optional
            Pixel value above which a sample is considered saturated.

        Returns
        -------
        str
            Saturation class:

            - ``constants.dq.SATURATED_STR`` when at least one pixel is saturated in >90% frames.
            - ``constants.dq.SOMETIMES_SATURATED_STR`` when saturation occurs intermittently.
            - ``constants.dq.NOT_SATURATED_STR`` when no saturation is detected.
        """
        # TODO: make sure we are only looking at relevant tracking frames
        saturated_pixels = self.asdf_file.tree['roman']['track_data']['signal_resultants'] > saturation_threshold
        
        if (np.sum(saturated_pixels, axis=0) > 0.9 * self.asdf_file.tree['roman']['track_data']['signal_resultants'].shape[0]).any(): # if more than 90% of the frames are saturated for a given pixel, we will consider that pixel to be saturated.
            return SATURATED_STR
        elif (np.sum(saturated_pixels, axis=0) > 0).any(): # if some but not most of the frames are saturated for a given pixel, we will consider that pixel to be sometimes saturated.
            return SOMETIMES_SATURATED_STR
        else:
            return NOT_SATURATED_STR

    def check_acquisition_status(self):
        """
        Determine whether guide-star acquisition appears successful.

        Returns
        -------
        success : bool
            ``True`` when centroid information indicates successful acquisition.
        acquisition_status : str
            Status code summarizing acquisition quality/failure mode.
        """
        # start by assuming the status was successful
        success = True
        acquisition_status = "SUCCESS"

        if (len(self.asdf_file.tree['roman']['centroid'].keys()) == 0) or (self.asdf_file.tree['roman']['centroid']['track_centroids'] is None):
            success = False
            acquisition_status = "NO_TRACK_CENTROIDS"
        else:
            unique_cent_qualities, quality_counts = np.unique(self.asdf_file.tree['roman']['centroid']['track_centroid_quality'], return_counts=True)
            if unique_cent_qualities[np.argmax(quality_counts)] not in ['CQ_GOOD', "CQ_TOO_FEW_MATCHES", "CQ_DETECTOR_NOT_SELECTED"]:
                success = False
                acquisition_status = unique_cent_qualities[np.argmax(quality_counts)]
            elif self.asdf_file.tree['roman']['centroid']['track_centroids'].shape[0] < 300: # if there are fewer than 300 centroids, the acquisition likely failed. 300 is less than the number of centroids expected from a 60 second exposure (the shortest MA table).
                success = False
                acquisition_status = "TOO_FEW_CENTROIDS"

        return success, acquisition_status


    def _create_wim_brightest_pixel_masks(self, images, box_size=3):
        """ 
        Create masks for the brightest pixels in each WIM guide window image.

        Parameters
        ----------
        images : np.ndarray
            Array of shape (num_images, height, width) containing image data.
        box_size : int, optional
            Size of the box in pixels around the brightest pixel to include in the mask.

        Returns
        -------
        masks : np.ndarray
            Boolean array of the same shape as `images` where True indicates the brightest pixels.
        """
        num_images, height, width = images.shape
        masks = np.zeros((num_images, height, width), dtype=bool)
        
        max_indices = np.argmax(images.reshape(num_images, -1), axis=1)
        rows, cols = np.unravel_index(max_indices, (height, width))
        
        row_grid, col_grid = np.ogrid[:height, :width]
        
        for i, (row, col) in enumerate(zip(rows, cols)):
            masks[i] = (np.abs(row_grid - row) <= box_size // 2) & (np.abs(col_grid - col) <= box_size // 2)
        
        return masks
    
    def _create_wsm_brightest_pixel_masks(self, images, box_width=3, box_length=8):
        """ 
        Create masks centered on the brightest part of the WSM spectrum and that extend 8 pixels down and ``box_width`` pixels wide in each WSM guide window image.

        Parameters
        ----------
        images : np.ndarray
            Array of shape (num_images, height, width) containing image data.
        box_width : int, optional
            Width of the box in pixels around the bright edge of the spectrum.
        box_length : int, optional
            Length of the box in pixels along the edge of the spectrum (should be 8 for WSM guide windows).

        Returns
        -------
        masks : np.ndarray
            Boolean array of the same shape as `images` where True indicates the brightest pixels.
        """
        num_images, height, width = images.shape
        masks = np.zeros((num_images, height, width), dtype=bool)

        # Find the brightest column in the top or bottom 8 rows depending on the edge location
        if self.wsm_edge_location == WSM_GW_TOP:
            collapsed_images = np.sum(images[:, -box_length:, :], axis=1)
        else: # if self.wsm_edge_location == WSM_GW_BOTTOM:
            collapsed_images = np.sum(images[:, :box_length, :], axis=1)
        
        max_col_indices = np.argmax(collapsed_images, axis=1)
        
        for i, col in enumerate(max_col_indices):
            if self.wsm_edge_location == WSM_GW_TOP:
                masks[i, -box_length:, max(0, col - box_width // 2):min(width, col + box_width // 2 + 1)] = True
            else: # if self.wsm_edge_location == WSM_GW_BOTTOM:
                masks[i, :box_length, max(0, col - box_width // 2):min(width, col + box_width // 2 + 1)] = True

        return masks

    def calculate_count_rates(self, background_counts=0, box_size=3):
        """
        Compute observed guide-star count-rate statistics.

        Parameters
        ----------
        background_counts : float, optional
            Background level to subtract from each pixel before summation.
        box_size : int, optional
            Size of the box in pixels around the brightest pixel to sum for count rate calculation, in pixels.

        Returns
        -------
        mean_count_rate : float
            Mean count rate over all frames in DN/s.
        std_count_rate : float
            Standard deviation of frame count rates in DN/s.
        num_count_rate_outliers : int
            Number of frames with count rate above the +3 sigma boundary.
        """
        frame_differences = self.asdf_file.tree['roman']['track_data']['signal_resultants'].astype(float) - self.asdf_file.tree['roman']['track_data']['pedestal_resultants'].astype(float)
        exposure_time = self.asdf_file.tree['roman']['meta']['track_signal_resultant_exp_time'] - self.asdf_file.tree['roman']['meta']['track_pedestal_resultant_exp_time']

        # remove a typical background count level
        frame_differences -= background_counts

        element = self.asdf_file.tree['roman']['meta']['instrument']['optical_element']
        # Define a compact aperture where the guide-star signal is expected.
        # For WIM the mask should be centered on the brightest pixel
        if "f" in element.lower():
            masks = self._create_wim_brightest_pixel_masks(frame_differences, box_size=box_size)
        # For WSM the mask should be on the edge of the array where the spectrum is
        else:
            masks = self._create_wsm_brightest_pixel_masks(frame_differences, box_width=box_size)

        count_rates = np.sum(frame_differences * masks, axis=(1,2)) / exposure_time

        # calculate the mean and std of the count rates across all frames
        mean_count_rate = np.mean(count_rates)
        std_count_rate = np.std(count_rates)

        # check for outliers in the count rates by checking upper 3 std boundary
        num_count_rate_outliers = np.sum(count_rates - mean_count_rate > 3 * std_count_rate)

        return mean_count_rate, std_count_rate, num_count_rate_outliers

    def calculate_metrics(self):
        """
        Compute and store guide-window monitoring metrics.
        """
        # check if guide star acquisition was successful 
        success, acquisition_status = self.check_acquisition_status()
        self.append_data("acquisition_status", acquisition_status, "") # to evaluate if the guide star acquisition was successful

        # check if the guide window is saturated
        saturation_status = self.check_saturation()
        self.append_data("saturation_status", saturation_status, "") # to evaluate if the guide window is saturated and therefore unusable for guiding.

        # check the background and noise in the guide window
        median_backgrounds, noise_estimates = self._estimate_background_and_noise()
        self.append_data("median_background", np.median(median_backgrounds), "DN") # to evaluate the background level in the guide window
        self.append_data("mean_noise", np.mean(noise_estimates), "DN")

        # check for outliers in the background and noise estimates by checking upper 3 std boundary
        median_background_std = np.std(median_backgrounds)
        noise_std = np.std(noise_estimates)
        self.append_data("median_background_std", median_background_std, "DN") # to evaluate the stability of the background level in the guide window
        self.append_data("noise_std", noise_std, "DN") # to evaluate the stability of the noise in the guide window

        num_background_outliers = np.sum(median_backgrounds - np.mean(median_backgrounds) > 3 * median_background_std)
        num_noise_outliers = np.sum(noise_estimates - np.mean(noise_estimates) > 3 * noise_std)
        self.append_data("num_background_outliers", num_background_outliers, "") # to evaluate the stability of the background level in the guide window
        self.append_data("num_noise_outliers", num_noise_outliers, "") # to evaluate the stability of the noise in the guide window

        # calculate the average and STD of the observed count rates from the differences in signal and pedestal guide window frames.
        mean_count_rate, std_count_rate, num_count_rate_outliers = self.calculate_count_rates()
        self.append_data("mean_count_rate", mean_count_rate, "DN/s") # to evaluate the average count rate in the guide window
        self.append_data("std_count_rate", std_count_rate, "DN/s") # to evaluate the variability of the count rate in the guide window
        self.append_data("num_count_rate_outliers", num_count_rate_outliers, "") # to evaluate the stability of the count rate in the guide window

        # if the acquisition was successful, calculate metrics that rely on the centroid data. 
        if success:
            mean_centroid_positions, rms_x_centroid_error, rms_y_centroid_error = self._process_centroids()

            # Store centroid metrics in datacards
            self.append_data("rms_x_centroid_error", rms_x_centroid_error, "pixels") # to evaluate jitter / stability 
            self.append_data("rms_y_centroid_error", rms_y_centroid_error, "pixels") 

            # check for systematic centroid position offsets
            expected_centroid_position = np.array([
                self.asdf_file.tree['roman']['meta']['guide_star']['predicted_x'], 
                self.asdf_file.tree['roman']['meta']['guide_star']['predicted_y']
            ])
            # Radial offset between measured and predicted centroid positions.
            rms_centroid_offset = np.sqrt(np.mean((mean_centroid_positions - expected_centroid_position)**2))
            self.append_data("rms_centroid_offset", rms_centroid_offset, "pixels") # to evaluate if there is a systematic offset in the centroid positions compared to the expected position

        else: 
            self.append_data("rms_x_centroid_error", None, "pixels") 
            self.append_data("rms_y_centroid_error", None, "pixels") 
            self.append_data("rms_centroid_offset", None, "pixels")


    def evaluate_metrics(self):
        """
        Evaluate computed metrics against monitor acceptance thresholds.
        """

        acquisition_status = self.get_data("acquisition_status") 
        if acquisition_status.upper() != "SUCCESS":
            logger.error(f"{self.monitor_name}: Guide star acquisition was not successful ({acquisition_status})")
            self.add_evaluation("acquisition_status", False)
        else:
            self.add_evaluation("acquisition_status", True)
        
        saturation_status = self.get_data("saturation_status") 
        if saturation_status.upper() != NOT_SATURATED_STR:
            logger.error(f"{self.monitor_name}: Guide window is saturated ({saturation_status})")
            self.add_evaluation("saturation_status", False)
        else:
            self.add_evaluation("saturation_status", True)
        
        median_background = self.get_data("median_background")
        if self.is_valid_metric("median_background", median_background):
            if median_background > MEDIAN_BACKGROUND_THRESHOLD:
                logger.warning(f"{self.monitor_name}: Median background is above the threshold ({median_background} DN > {MEDIAN_BACKGROUND_THRESHOLD} DN).")
                self.add_evaluation("median_background", False)
            else:
                self.add_evaluation("median_background", True)

        mean_noise = self.get_data("mean_noise")
        if self.is_valid_metric("mean_noise", mean_noise):
            if mean_noise > MEAN_NOISE_THRESHOLD:
                logger.warning(f"{self.monitor_name}: Mean noise is above the threshold ({mean_noise} DN > {MEAN_NOISE_THRESHOLD} DN).")
                self.add_evaluation("mean_noise", False)
            else:
                self.add_evaluation("mean_noise", True)

        background_std = self.get_data("median_background_std")
        if self.is_valid_metric("median_background_std", background_std):
            if background_std > BACKGROUND_STD_THRESHOLD:
                logger.warning(f"{self.monitor_name}: Median background std is above the threshold ({background_std} DN > {BACKGROUND_STD_THRESHOLD} DN).")
                self.add_evaluation("median_background_std", False)
            else:
                self.add_evaluation("median_background_std", True)

        noise_std = self.get_data("noise_std")
        if self.is_valid_metric("noise_std", noise_std):
            if noise_std > NOISE_STD_THRESHOLD:
                logger.warning(f"{self.monitor_name}: Noise std is above the threshold ({noise_std} DN > {NOISE_STD_THRESHOLD} DN).")
                self.add_evaluation("noise_std", False)
            else:
                self.add_evaluation("noise_std", True)

        num_background_outliers = self.get_data("num_background_outliers")
        if self.is_valid_metric("num_background_outliers", num_background_outliers):
            if num_background_outliers > NUM_BACKGROUND_OUTLIERS_THRESHOLD:
                logger.warning(f"{self.monitor_name}: There are {num_background_outliers} background outliers.")
                self.add_evaluation("num_background_outliers", False)
            else:
                self.add_evaluation("num_background_outliers", True)

        num_noise_outliers = self.get_data("num_noise_outliers")
        if self.is_valid_metric("num_noise_outliers", num_noise_outliers):
            if num_noise_outliers > NUM_NOISE_OUTLIERS_THRESHOLD:
                logger.warning(f"{self.monitor_name}: There are {num_noise_outliers} noise outliers.")
                self.add_evaluation("num_noise_outliers", False)
            else:
                self.add_evaluation("num_noise_outliers", True)
        
        mean_count_rate = self.get_data("mean_count_rate")
        std_count_rate = self.get_data("std_count_rate")

        predicted_count_rate = self.asdf_file.tree['roman']['meta']['guide_star']['predicted_count_rate']
        predicted_fgs_mag = self.asdf_file.tree['roman']['meta']['guide_star']['predicted_fgs_mag']
        predicted_fgs_faint_mag = self.asdf_file.tree['roman']['meta']['guide_star']['predicted_fgs_faint_mag']
        predicted_fgs_bright_mag = self.asdf_file.tree['roman']['meta']['guide_star']['predicted_fgs_bright_mag']

        # Convert predicted magnitudes to corresponding bright/faint count-rate bounds.
        zero_point = predicted_fgs_mag + 2.5 * np.log10(predicted_count_rate)
        predicted_bright_count_rate = 10**((zero_point - predicted_fgs_bright_mag) / 2.5)
        predicted_faint_count_rate = 10**((zero_point - predicted_fgs_faint_mag) / 2.5)

        valid_std_count_rate = self.is_valid_metric("std_count_rate", std_count_rate)
        if valid_std_count_rate:
            if FGS_LIMIT_MATCHING_FACTOR*std_count_rate > max(predicted_bright_count_rate - predicted_count_rate, predicted_count_rate - predicted_faint_count_rate):
                logger.warning(f"{self.monitor_name}: Std of count rate is high compared to the predicted mag limits ({std_count_rate} DN vs {predicted_faint_count_rate} - {predicted_bright_count_rate} DN).")
                self.add_evaluation("std_count_rate", False)
            else:
                self.add_evaluation("std_count_rate", True)
        
        if self.is_valid_metric("mean_count_rate", mean_count_rate):
            if not valid_std_count_rate:
                logger.warning(f"{self.monitor_name}: Std of count rate is not valid, only performing limit check.")
                if (mean_count_rate < predicted_faint_count_rate) or (mean_count_rate > predicted_bright_count_rate):
                    self.add_evaluation("mean_count_rate", False)
                else:
                    self.add_evaluation("mean_count_rate", True)
            elif abs(mean_count_rate - predicted_count_rate) > COUNT_RATE_MATCHING_FACTOR * std_count_rate:
                logger.error(f"{self.monitor_name}: Mean count rate is outside the expected range ({mean_count_rate}+-{COUNT_RATE_MATCHING_FACTOR*std_count_rate} DN ({COUNT_RATE_MATCHING_FACTOR} sigma error) vs {predicted_count_rate} DN).")
                self.add_evaluation("mean_count_rate", False)
            else:
                self.add_evaluation("mean_count_rate", True)

        num_count_rate_outliers = self.get_data("num_count_rate_outliers")
        if self.is_valid_metric("num_count_rate_outliers", num_count_rate_outliers):
            if num_count_rate_outliers > NUM_COUNT_RATE_OUTLIERS_THRESHOLD:
                logger.warning(f"{self.monitor_name}: There are {num_count_rate_outliers} count rate outliers.")
                self.add_evaluation("num_count_rate_outliers", False)
            else:
                self.add_evaluation("num_count_rate_outliers", True)

        # evaluate the centroid metrics if they exist (e.g. acquisition was successful)
        if acquisition_status == "SUCCESS":
            rms_x_centroid_error = self.get_data("rms_x_centroid_error")
            rms_y_centroid_error = self.get_data("rms_y_centroid_error")

            optical_element = self.asdf_file.tree['roman']['meta']['instrument']['optical_element']
            fgs_mode = self.asdf_file.tree['roman']['meta']['fgs_modes_used'][0]

            if self.is_valid_metric("rms_x_centroid_error", rms_x_centroid_error):
                if "WIM" in fgs_mode:
                    threshold = RMS_X_JITTER_THRESHOLDS.get("WIM")
                elif "DEF" in optical_element.upper():
                    threshold = RMS_X_JITTER_THRESHOLDS.get("DEFOCUS")
                else:
                    threshold = RMS_X_JITTER_THRESHOLDS.get(optical_element.upper())
                if threshold is None:
                    logger.warning(f"{self.monitor_name}: No X RMS jitter threshold defined for optical element {optical_element} and FGS mode {fgs_mode}. Skipping evaluation of RMS centroid error.")
                    raise ValueError(f"No X RMS jitter threshold defined for optical element {optical_element} and FGS mode {fgs_mode}.")

                if rms_x_centroid_error > threshold:
                    logger.warning(f"{self.monitor_name}: X RMS centroid error is above the threshold ({rms_x_centroid_error} pixels > {threshold} pixels).")
                    self.add_evaluation("rms_x_centroid_error", False)
                else:
                    self.add_evaluation("rms_x_centroid_error", True)

            if self.is_valid_metric("rms_y_centroid_error", rms_y_centroid_error):
                if "WIM" in fgs_mode:
                    threshold = RMS_Y_JITTER_THRESHOLDS.get("WIM")
                elif "DEF" in optical_element.upper():
                    threshold = RMS_Y_JITTER_THRESHOLDS.get("DEFOCUS")
                else:
                    threshold = RMS_Y_JITTER_THRESHOLDS.get(optical_element.upper())
                if threshold is None:
                    logger.warning(f"{self.monitor_name}: No Y RMS jitter threshold defined for optical element {optical_element} and FGS mode {fgs_mode}. Skipping evaluation of RMS centroid error.")
                    raise ValueError(f"No Y RMS jitter threshold defined for optical element {optical_element} and FGS mode {fgs_mode}.")

                if rms_y_centroid_error > threshold:
                    logger.warning(f"{self.monitor_name}: Y RMS centroid error is above the threshold ({rms_y_centroid_error} pixels > {threshold} pixels).")
                    self.add_evaluation("rms_y_centroid_error", False)
                else:
                    self.add_evaluation("rms_y_centroid_error", True)

            rms_centroid_offset = self.get_data("rms_centroid_offset")
            if self.is_valid_metric("rms_centroid_offset", rms_centroid_offset):
                if rms_centroid_offset > RMS_CENTROID_OFFSET_THRESHOLD:
                    logger.warning(f"{self.monitor_name}: RMS centroid offset is above the threshold ({rms_centroid_offset} pixels > {RMS_CENTROID_OFFSET_THRESHOLD} pixels).")
                    self.add_evaluation("rms_centroid_offset", False)
                else:
                    self.add_evaluation("rms_centroid_offset", True)
