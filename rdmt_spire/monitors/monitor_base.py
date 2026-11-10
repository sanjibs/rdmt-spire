import logging
import os
from abc import ABC, abstractmethod
from dataclasses import asdict
from typing import Any, Dict, List, Sequence

import asdf
import numpy as np
from pydantic.dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class DataCard:
    """
    A DataCard formats and standardizes the output of the monitors. 
    
    It is a container object to ensure a standardized set of information is saved for every metric
    from every monitor. The DataCards are used to send output information into the archiving
    functionality.
    """

    filename: str
    monitor_name: str
    data_name: str
    data_value: int| float| str| Sequence[int]| Sequence[float]| Sequence[str]
    data_unit: str
    evaluation_value: bool| None


class BaseMonitor(ABC):
    """
    The base class of monitors. Each monitor should be subclassed from this.
    """

    def __init__(self, asdf_file: asdf.AsdfFile):
        self.asdf_file = asdf_file
        self.monitor_name = "base_monitor"
        self.data: Dict[DataCard] = dict()
        self.log: list[str] = []
        if self.asdf_file.uri is None:
            self.filename = ''
        else:
            self.filename = os.path.basename(self.asdf_file.uri)

    def check_input(self):
        """Confirms the input file is valid."""
        # ensure the asdf_file is valid
        if not isinstance(self.asdf_file, asdf.AsdfFile):
            raise RuntimeError(f"{self.__class__.__name__}: failure, invalid input")

    def _validate_data_name(
        self,
        data_name: str,
        *,
        allow_all: bool = False,
        require_existing: bool = False,
        require_new: bool = False,
    ):
        """Validate data_name input used by public data access/modification methods."""
        if not isinstance(data_name, str) or not data_name:
            raise ValueError(f"{self.__class__.__name__}: failure, data_name must be a non-empty string")

        if not data_name.islower():
            raise ValueError(f"{self.__class__.__name__}: failure, data_name must be lowercase")

        if allow_all and data_name == 'all':
            return

        if require_existing and data_name not in self.data:
            raise ValueError(f"No metric has been added with data_name={data_name}. Please check implementation.")

        if require_new and data_name in self.data:
            raise ValueError(f"A metric with data_name={data_name} already exists. Please check implementation.")

    def append_data(self, data_name: str, data_value: Any, data_unit: str, evaluation_value: bool = None):
        """
        Format and then append the data to be exported out of the monitor
        """
        self._validate_data_name(data_name, require_new=True)
        if data_name in self.data:
            raise ValueError(f"{self.__class__.__name__}: failure, data_name {data_name} already exists in data dictionary")
        else:
            data_card = DataCard(
                self.filename,
                self.monitor_name,
                data_name,
                data_value,
                data_unit,
                evaluation_value
            )
            self.data[data_name] = data_card

    def add_evaluation(self, data_name: str, eval_value: bool):
        """
        Add or modify the evaluation value of a given data card.
        """
        self._validate_data_name(data_name, require_existing=True)
        self.data[data_name].evaluation_value = eval_value

    @abstractmethod
    def calculate_metrics(self):
        """
        Function in which metric data should be calculated and stored in the data attribute. 
        
        This method should be overridden in derived classes.
        """
        pass

    @abstractmethod
    def evaluate_metrics(self):
        """
        Function in which metrics should be evaluated against thresholds. 
        
        The boolean results should be stored in the metrics' data attributes. 
        This method should be overridden in derived classes.
        """
        pass

    def run(self):
        """
        This is the main function where metrics are computed and compared against thresholds 
        to evaluate them.
        """
        logger.info(f"{self.monitor_name}: Starting run")
        self.calculate_metrics()
        self.evaluate_metrics()
        logger.info(f"{self.monitor_name}: Finished run, computed {len(self.data)} data cards")

    def get_data_card(self, data_name: str, serialized: bool = False) -> List[DataCard] | List[Dict[str, Any]]:
        """
        Export the results of the monitor in a standardized format. 
        
        Can be used to export individual data cards or all data cards in a list by using 
        'all' as the input name.
        """
        self._validate_data_name(data_name, allow_all=True, require_existing=True)

        if data_name == 'all':
            if serialized:
                return [asdict(item) for _, item in self.data.items()]
            else:
                return list(self.data.values())
        else:
            data_card = self.data[data_name]
            if serialized:
                return asdict(data_card)
            else:
                return data_card

    def get_data(self, data_name: str):
        """
        Helper function to retrieve the value of a metric saved in a DataCard.
        """
        self._validate_data_name(data_name, require_existing=True, allow_all=True)
        return self.get_data_card(data_name).data_value

    def is_valid_metric(self, metric_name: str, value: Any) -> bool:
        """
        Check if a metric value is valid (not None and finite).

        If invalid, logs an error and marks the metric evaluation as failed.

        Parameters
        ----------
        metric_name : str
            Name of the metric being validated.
        value : Any
            The metric value to validate.

        Returns
        -------
        bool
            True if the value is valid, False otherwise.
        """
        is_valid = False
        if value is not None:
            try:
                is_valid = bool(np.all(np.isfinite(value)))
            except (TypeError, ValueError):
                is_valid = False

        if not is_valid:
            logger.error(f"{self.monitor_name}: {metric_name} is invalid, marking metric as failed.")
            self.add_evaluation(metric_name, False)
            return False
        return True