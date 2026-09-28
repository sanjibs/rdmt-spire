from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Sequence

import astropy.units as u
import numpy as np
from astropy.table import QTable, Table
from astropy.units import Quantity

__all__ = [
    "Property",
    "ScalarProperty",
    "Dist1DProperty",
    "GenChi2Property",
    "MetricCard",
]


@dataclass
class MetricCard:
    """Standardized output for one monitored metric."""

    data_name: str
    data_value: int | float
    data_unit: str
    evaluation_value: bool = True

    def evaluate(
        self,
        metric_thresholds: Table | QTable,
        error: Any,
        nsigma: float,
        suffix1: str = "",
        suffix2: str = "",
    ) -> None:
        vmin, vmax = self.get_metric_threshold(metric_thresholds, self.data_name, suffix1, suffix2)
        x = self.data_value * u.Unit(self.data_unit)
        print(self.data_name, "x:", x, "vmin:", vmin, "vmax:", vmax, "error:", error, "nsigma:", nsigma, "data_unit:", self.data_unit)

        if np.isfinite(self.data_value):
            if (x < vmin - nsigma * error) or (x > vmax + nsigma * error):
                self.evaluation_value = False

    @staticmethod
    def get_metric_threshold(
        metric_thresholds: Table | QTable,
        metric_name: str,
        suffix1: str = "",
        suffix2: str = "",
    ) -> list[float | Quantity]:
        if suffix1:
            metric_name = metric_name.removesuffix(suffix1)

        candidates = (
            f"{metric_name}{suffix1}{suffix2}",
            f"{metric_name}{suffix1}",
            metric_name,
        )
        for candidate in candidates:
            condition = metric_thresholds["property_name"] == candidate
            if np.any(condition):
                vmin= np.asarray(metric_thresholds["min"][condition])[0]
                vmax = np.asarray(metric_thresholds["max"][condition])[0]
                vunit = str(np.asarray(metric_thresholds["unit"][condition])[0])
                if vunit in [None,'None','none']:
                    vunit = u.Unit("")
                else:
                    vunit = u.Unit(vunit)
                return [float(vmin)*vunit, float(vmax)*vunit]

        raise RuntimeError(
            f"SourceCatalogMonitor: filter '{metric_name}{suffix1}{suffix2}' "
            "not found in evaluation thresholds table"
        )




class Property(ABC):
    """Base class for a monitored property.

    Constructing ``Property`` with a ``stat_style`` returns the matching
    derived class, preserving the constructor used by the original API.
    """

    _styles: dict[str, type[Property]] = {}

    def __new__(cls, prop_name: str, stat_style: str, *args: Any, **kwargs: Any):
        if cls is Property:
            try:
                cls = Property._styles[stat_style]
            except KeyError as exc:
                raise RuntimeError(
                    f"SourceCatalogMonitor: unknown stat_style '{stat_style}'"
                ) from exc
        return super().__new__(cls)

    def __init__(
        self,
        prop_name: str,
        stat_style: str,
        outlier_thresholds: Sequence[float] | None = None,
        suffix1: str = "",
        dof: int = 2,
    ) -> None:
        self.prop_name = prop_name
        self.stat_style = stat_style
        self.outlier_thresholds = outlier_thresholds
        self.suffix1 = suffix1
        # Remove the suffix from the property name if it matches the provided suffix1
        # if len(suffix1) > 0:
        #     if self.prop_name.endswith(suffix1):
        #         self.prop_name = self.prop_name.removesuffix(suffix1)
        self.dof = dof
        self.cards: dict[str, MetricCard] = {}

    def process(self, x: Any, unit: str = "") -> tuple[Any, str]:
        if hasattr(x, "unit"):
            unit = str(x.unit)
            x = x.value
        if unit in [None,'None','none']:
            unit = ""
        return x, unit


    @abstractmethod
    def compute(self, x: Any, unit: str = "") -> Property:
        raise NotImplementedError()

    @abstractmethod
    def evaluate(self, metric_thresholds: Table | QTable, suffix2: str = "") -> None:
        raise NotImplementedError()

    def default_status(self, z) -> bool:
        # return bool(np.isfinite(z))
        return True

    def f_outliers(self, x, unit):
        thresholds = self.outlier_thresholds  # Assuming a method to get thresholds based on the unit
        if np.size(x) > 0:
            if np.isnan(self.outlier_thresholds[0]):
                percentiles = np.percentile(x, [15.86, 50.0, 84.14])
                lower = percentiles[1] - 5 * (percentiles[1] - percentiles[0])
                upper = percentiles[1] + 5 * (percentiles[2] - percentiles[1])
                temp=float(np.sum((x > upper) | (x < lower)) / np.size(x))
            else:
                temp=float(np.sum((x < thresholds[0]) | (x > thresholds[1])) / np.size(x))
        else:
            temp=0.0
        name=f"{self.prop_name}_{'f_outliers'}{self.suffix1}"
        return MetricCard(name, temp, "", self.default_status(temp))

    def remove_outliers(self, x):
        thresholds = self.outlier_thresholds
        if np.size(x) > 0:
            if np.isnan(thresholds[0]):
                percentiles = np.percentile(x, [15.86, 50.0, 84.14])
                lower = percentiles[1] - 5 * (percentiles[1] - percentiles[0])
                upper = percentiles[1] + 5 * (percentiles[2] - percentiles[1])
                x = x[(x >= lower) & (x <= upper)]
            else:
                x = x[(x >= thresholds[0]) & (x <= thresholds[1])]
        return x

    def n_sources(self, x, unit):
        temp=np.size(x) 
        name=f"{self.prop_name}_{'n_sources'}{self.suffix1}"
        return MetricCard(name, temp, "", self.default_status(temp))

    def blank(self, x, unit):
        temp=x
        if isinstance(x, (float, int)):
            name=f"{self.prop_name}{self.suffix1}"
            return MetricCard(name, temp, unit, self.default_status(temp))
        raise ValueError("Input x should be a scalar value of type float or int.")

    def median(self, x, unit):
        temp=float(np.median(x)) if np.size(x) > 0 else np.nan
        name=f"{self.prop_name}_{'median'}{self.suffix1}"
        return MetricCard(name, temp, unit, self.default_status(temp))
    
    def mean(self, x, unit):
        temp=float(np.mean(x)) if np.size(x) > 0 else np.nan
        name=f"{self.prop_name}_{'mean'}{self.suffix1}"
        return MetricCard(name, temp, unit, self.default_status(temp))

    def std(self, x, unit):
        temp=float(np.std(x)) if np.size(x) > 0 else np.nan
        name=f"{self.prop_name}_{'std'}{self.suffix1}"
        return MetricCard(name, temp, unit, self.default_status(temp))

    def nmad(self, x, unit):
        temp=float(1.4826 * np.median(np.abs(x - np.median(x)))) if np.size(x) > 0 else np.nan
        name=f"{self.prop_name}_{'nmad'}{self.suffix1}"
        return MetricCard(name, temp, unit, self.default_status(temp))

    def mad(self, x, unit):
        temp=float(np.median(np.abs(x - np.median(x)))) if np.size(x) > 0 else np.nan
        name=f"{self.prop_name}_{'mad'}{self.suffix1}"
        return MetricCard(name, temp, unit, self.default_status(temp))

    def stdp68(self, x, unit):
        temp=float(np.diff(np.percentile(x, [84.14, 15.86])) * 0.5) if np.size(x) > 0 else np.nan
        name=f"{self.prop_name}_{'stdp68'}{self.suffix1}"
        return MetricCard(name, temp, unit, self.default_status(temp))

    def stdp95(self, x, unit):
        temp=float(np.diff(np.percentile(x, [97.725, 2.275])) * 0.25) if np.size(x) > 0 else np.nan
        name=f"{self.prop_name}_{'stdp95'}{self.suffix1}"
        return MetricCard(name, temp, unit, self.default_status(temp))

    def rmse(self, x, unit):
        temp=float(np.sqrt(np.mean(x * x))) if np.size(x) > 0 else np.nan
        name=f"{self.prop_name}_{'rmse'}{self.suffix1}"
        return MetricCard(name, temp, unit, self.default_status(temp))

    def mae(self, x, unit):
        temp=float(np.mean(np.abs(x))) if np.size(x) > 0 else np.nan
        name=f"{self.prop_name}_{'mae'}{self.suffix1}"
        return MetricCard(name, temp, unit, self.default_status(temp))

    def medae(self, x, unit):
        temp=float(np.median(np.abs(x))) if np.size(x) > 0 else np.nan
        name=f"{self.prop_name}_{'medae'}{self.suffix1}"
        return MetricCard(name, temp, unit, self.default_status(temp))



class ScalarProperty(Property):
    """Property family that evaluates the raw scalar value."""

    def compute(self, x: np.ndarray|Quantity, unit: str = "") -> "ScalarProperty":
        x, unit = self.process(x, unit)
        self.cards={}
        self.cards["blank"]=self.blank(x, unit)
        return self

    def evaluate(self, metric_thresholds: Table | QTable, suffix2: str = "") -> None:
        self.cards["blank"].evaluate(metric_thresholds, 0.0, 3, self.suffix1, suffix2)


class Dist1DProperty(Property):
    """Property family that evaluates 1D distributions."""

    def compute(self, x: np.ndarray|Quantity, unit: str = "") -> "Dist1DProperty":
        x, unit = self.process(x, unit)
        x=x[np.isfinite(x)]
        self.cards={}
        self.cards["n_sources"]=self.n_sources(x, "")
        if self.outlier_thresholds is not None:
            self.cards["f_outliers"]=self.f_outliers(x, "")
            x=self.remove_outliers(x)

        self.cards["median"]=self.median(x, unit)
        self.cards["nmad"]=self.nmad(x, unit)
        self.cards["mean"]=self.mean(x, unit)
        self.cards["std"]=self.std(x, unit)
        # print('Computed Dist1DProperty cards:', self.cards.keys())
        return self

    def evaluate(self, metric_thresholds: Table | QTable, suffix2: str = "") -> None:
        n_sources = self.cards["n_sources"].data_value
        if n_sources < 10:
            self.cards["n_sources"].evaluation_value = False
        else:
            vunit = u.Unit(self.cards["nmad"].data_unit)
            dispersion = self.cards["nmad"].data_value *vunit
            error = dispersion / np.sqrt(n_sources)
            self.cards["median"].evaluate(metric_thresholds, error, 3, self.suffix1, suffix2)
            error = dispersion / np.sqrt(2 * (n_sources - 1))
            self.cards["nmad"].evaluate(metric_thresholds, error, 3, self.suffix1, suffix2)



class GenChi2Property(Property):
    """Property family that evaluates generalized chi-squared statistics."""

 
    def compute(self, x: np.ndarray|Quantity, unit: str = "") -> "GenChi2Property":
        x, unit = self.process(x, unit)
        x=x[np.isfinite(x)]
        self.cards={}
        self.cards["n_sources"]=self.n_sources(x, "")
        if self.outlier_thresholds is not None:
            self.cards["f_outliers"]=self.f_outliers(x, "")
            x=self.remove_outliers(x)
        self.cards["medae"]=self.medae(x, unit)
        self.cards["rmse"]=self.rmse(x, unit)
        self.cards["mae"]=self.mae(x, unit)
        return self

    def evaluate(self, metric_thresholds: Table | QTable, suffix2: str = "") -> None:
        n_sources = self.cards["n_sources"].data_value
        if n_sources < 10:
            self.cards["n_sources"].evaluation_value = False
        else:
            vunit = u.Unit(self.cards["medae"].data_unit)
            dispersion = self.cards["medae"].data_value * (3 * np.sqrt(self.dof)) * vunit/ (3 * self.dof - 1)
            error = dispersion / np.sqrt(2 * n_sources)
            self.cards["medae"].evaluate(metric_thresholds, error, 3, self.suffix1, suffix2)


Property._styles = {
    "scalar": ScalarProperty,
    "dist1d": Dist1DProperty,
    "genchi2": GenChi2Property,
}


