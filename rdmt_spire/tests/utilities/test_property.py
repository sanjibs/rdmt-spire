
"""Unit tests for ``utilities.property.Property`` value summarization and threshold evaluation.

The module exercises ``Property.compute()`` and ``Property.evaluate()`` with small,
fully synthetic inputs built inline: scalar values, short numeric distributions,
quantity-valued arrays, empty and all-NaN samples, and simple threshold tables created
with ``astropy.table.Table``. The strategy is direct and local; there are no fixtures,
monkeypatches, or external files, only minimal arrays chosen to make the expected
statistics and pass/fail states easy to verify analytically.

The tests cover scalar cards, ``dist1d`` summaries, optional outlier fractions, and
``genchi2`` summary metrics, including unit propagation and tolerance checks against
known median, mean, NMAD, standard deviation, median absolute error, and RMSE values.
They also check edge cases for single-element and empty inputs, confirm that evaluation
flags remain stable when thresholds are absent or unit-mismatched, and verify the name
resolution order for suffixed threshold keys through the ``suffix1`` and ``suffix2``
fallback chain.
"""

from __future__ import annotations

import astropy.units as u
import numpy as np
import pytest
from astropy.table import Table

from ...utilities.property import Property


def test_property_scalar_compute() -> None:
    """Check scalar property handling for valid, missing, and out-of-range values.

    This exercises the core logic of the Property class: a scalar value should be stored,
    flagged as valid when finite, and evaluated against a threshold table. The test also
    covers NaN inputs and units to confirm that type conversion and validity checks behave
    as intended.
    """
    thresholds = Table({
        "property_name": ["flux", "flux2"],
        "min": [0.0, 0.0],
        "max": [10.0, 10.0],
        "unit": ["m", "m"],
    })

    prop = Property("flux", "scalar")
    value = 1*u.m
    prop.compute(value)
    assert prop.cards["blank"].data_name == "flux"
    assert prop.cards["blank"].data_value == 1
    assert prop.cards["blank"].evaluation_value is True
    prop.evaluate(thresholds)
    assert prop.cards["blank"].evaluation_value is True

    prop = Property("flux", "scalar")
    value = np.nan
    prop.compute(value)
    assert prop.cards["blank"].data_name == "flux"
    assert np.isnan(prop.cards["blank"].data_value)
    assert prop.cards["blank"].evaluation_value is True
    prop.evaluate(thresholds)
    assert prop.cards["blank"].evaluation_value is True

    prop = Property("flux", "scalar")
    value = 1.0*u.m
    prop.compute(value)
    assert prop.cards["blank"].data_name == "flux"
    assert prop.cards["blank"].data_value == pytest.approx(1.0)
    assert prop.cards["blank"].data_unit == "m"
    assert prop.cards["blank"].evaluation_value is True
    prop.evaluate(thresholds)
    assert prop.cards["blank"].evaluation_value is True


    prop = Property("flux", "scalar")
    value = 11.0*u.m
    prop.compute(value)
    prop.evaluate(thresholds)
    assert prop.cards["blank"].evaluation_value is False

    prop = Property("flux", "scalar")
    value = -1.0*u.m
    prop.compute(value)
    prop.evaluate(thresholds)
    assert prop.cards["blank"].evaluation_value is False



def test_property_dist1d_summary_and_thresholds() -> None:
    """Validate summary statistics and threshold evaluation for a 1D distribution.

    The distribution is a simple sequence with a known median, mean, MAD, and standard
    deviation, so the test checks that those summary statistics are computed correctly and
    that threshold passes/fails reflect the expected values.
    """
    values = np.arange(1.0, 11.0)
    prop = Property("flux", "dist1d", outlier_thresholds=[np.nan, np.nan])
    prop.compute(values)

    assert prop.cards["n_sources"].data_value == 10
    assert prop.cards["median"].data_value == pytest.approx(5.5)
    assert prop.cards["mean"].data_value == pytest.approx(5.5)
    assert prop.cards["nmad"].data_value == pytest.approx(3.7065, rel=1e-3)
    assert prop.cards["std"].data_value == pytest.approx(2.8722, rel=1e-3)

    assert prop.cards["n_sources"].evaluation_value is True
    assert prop.cards["median"].evaluation_value is True
    assert prop.cards["nmad"].evaluation_value is True
    assert prop.cards["mean"].evaluation_value is True
    assert prop.cards["std"].evaluation_value is True
    assert prop.cards["f_outliers"].evaluation_value is True

    thresholds = Table({
        "property_name": ["flux_median", "flux_nmad"],
        "min": [0.0, 0.0],
        "max": [10.0, 10.0],
        "unit": ["", ""],
    })
    prop.evaluate(thresholds)

    assert prop.cards["n_sources"].evaluation_value is True
    assert prop.cards["median"].evaluation_value is True
    assert prop.cards["nmad"].evaluation_value is True
    assert prop.cards["mean"].evaluation_value is True
    assert prop.cards["std"].evaluation_value is True
    assert prop.cards["f_outliers"].evaluation_value is True

    thresholds = Table({
        "property_name": ["flux_median", "flux_nmad"],
        "min": [-50.0, 100.0],
        "max": [-20.0, 200.0],
        "unit": ["", ""],
    })
    prop.evaluate(thresholds)

    assert prop.cards["n_sources"].evaluation_value is True
    assert prop.cards["median"].evaluation_value is False
    assert prop.cards["nmad"].evaluation_value is False 
    assert prop.cards["mean"].evaluation_value is True
    assert prop.cards["std"].evaluation_value is True
    assert prop.cards["f_outliers"].evaluation_value is True

def test_property_dist1d_single_element() -> None:
    """Confirm summary values remain valid for a single-element, non-empty sample.

    With only a few values, the code still must compute the count, central moments, and
    threshold evaluation consistently. This test checks the edge case where the input is
    single element array, and ensures the evaluator does not mis-classify it.
    """
    values = np.array([1.0])
    prop = Property("flux", "dist1d", outlier_thresholds=[np.nan, np.nan])
    prop.compute(values)
    thresholds = Table({
        "property_name": ["flux_median", "flux_nmad"],
        "min": [-50.0, 100.0],
        "max": [-20.0, 200.0],
    })
    assert prop.cards["n_sources"].evaluation_value is True
    prop.evaluate(thresholds)
    assert prop.cards["n_sources"].evaluation_value is False
    assert prop.cards["f_outliers"].evaluation_value is True
    assert prop.cards["median"].evaluation_value is True
    assert prop.cards["nmad"].evaluation_value is True
    assert prop.cards["mean"].evaluation_value is True
    assert prop.cards["std"].evaluation_value is True


def test_property_dist1d_small_distribution() -> None:
    """Confirm summary values remain valid for a short, non-empty sample.

    With only a few values, the code still must compute the count, central moments, and
    threshold evaluation consistently. This test checks the edge case where the input is
    small but not empty, and ensures the evaluator does not mis-classify it.
    """
    values = np.arange(1.0, 5.0)
    prop = Property("flux", "dist1d", outlier_thresholds=[np.nan, np.nan])
    prop.compute(values)
    thresholds = Table({
        "property_name": ["flux_median", "flux_nmad"],
        "min": [-50.0, 100.0],
        "max": [-20.0, 200.0],
    })
    assert prop.cards["n_sources"].evaluation_value is True
    prop.evaluate(thresholds)
    assert prop.cards["n_sources"].evaluation_value is False
    assert prop.cards["f_outliers"].evaluation_value is True
    assert prop.cards["median"].evaluation_value is True
    assert prop.cards["nmad"].evaluation_value is True
    assert prop.cards["mean"].evaluation_value is True
    assert prop.cards["std"].evaluation_value is True

def test_property_dist1d_units() -> None:
    """Confirm summary values remain valid for a short, non-empty sample.

    With only a few values, the code still must compute the count, central moments, and
    threshold evaluation consistently. This test checks the edge case where the input is
    small but not empty, and ensures the evaluator does not mis-classify it.
    """
    values = np.arange(1.0, 5.0)*u.nJy

    prop = Property("flux", "dist1d", outlier_thresholds=[np.nan, np.nan])
    prop.compute(values.value, unit="nJy")
    thresholds = Table({
        "property_name": ["flux_median", "flux_nmad"],
        "min": [-50.0, 100.0],
        "max": [-20.0, 200.0],
        "unit": ["nJy", "nJy"],
    })
    prop.evaluate(thresholds)
    assert prop.cards["median"].evaluation_value is True
    assert prop.cards["nmad"].evaluation_value is True

    prop = Property("flux", "dist1d", outlier_thresholds=[np.nan, np.nan])
    prop.compute(values)
    prop.evaluate(thresholds)
    assert prop.cards["median"].evaluation_value is True
    assert prop.cards["nmad"].evaluation_value is True

    thresholds = Table({
        "property_name": ["flux_median", "flux_nmad"],
        "min": [-50.0, 100.0],
        "max": [-20.0, 200.0],
        "unit": ["", None],
    })
    prop = Property("flux", "dist1d", outlier_thresholds=[np.nan, np.nan])
    prop.compute(values.value)
    prop.evaluate(thresholds)
    assert prop.cards["median"].evaluation_value is True
    assert prop.cards["nmad"].evaluation_value is True

    thresholds = Table({
        "property_name": ["flux_median", "flux_nmad"],
        "min": [-50.0, 100.0],
        "max": [-20.0, 200.0],
        "unit": ["", ""],
    })
    prop = Property("flux", "dist1d", outlier_thresholds=[np.nan, np.nan])
    prop.compute(values)
    try:
        prop.evaluate(thresholds)
    except Exception as e:
        pytest.fail(f"Evaluation raised an exception: {e}")

    thresholds = Table({
        "property_name": ["flux_median", "flux_nmad"],
        "min": [-50.0, 100.0],
        "max": [-20.0, 200.0],
        "unit": ["m", "m"],
    })
    try:
        prop.evaluate(thresholds)
    except Exception as e:
        pytest.fail(f"Evaluation raised an exception: {e}")

def test_property_dist1d_empty_distribution_sets_nan_and_zero_sources() -> None:
    """Check the empty-distribution edge case returns safe NaN statistics.

    For empty or all-NaN inputs, the Property object should record zero source counts and
    NaN summary metrics without crashing. The test verifies both the raw computed values
    and the boolean evaluation flags in this degenerate case.
    """
    thresholds = Table({
        "property_name": ["flux_median", "flux_nmad"],
        "min": [0.0, 0.0],
        "max": [10.0, 10.0],
    })
    x1=np.array([])
    x2=[np.nan]
    x3=[np.nan, np.nan]
    for x in [x1, x2, x3]:
        prop = Property("flux", "dist1d", outlier_thresholds=[0.0, 10.0])
        prop.compute(np.array(x))

        assert prop.cards["n_sources"].data_value == 0
        assert prop.cards["f_outliers"].data_value == pytest.approx(0.0)
        assert np.isnan(prop.cards["median"].data_value)
        assert np.isnan(prop.cards["nmad"].data_value)
        assert np.isnan(prop.cards["mean"].data_value)
        assert np.isnan(prop.cards["std"].data_value)

        assert prop.cards["n_sources"].evaluation_value is True
        assert prop.cards["f_outliers"].evaluation_value is True
        assert prop.cards["median"].evaluation_value is True
        assert prop.cards["nmad"].evaluation_value is True
        assert prop.cards["mean"].evaluation_value is True
        assert prop.cards["std"].evaluation_value is True

        prop.evaluate(thresholds)
        assert prop.cards["n_sources"].evaluation_value is False
        assert prop.cards["f_outliers"].evaluation_value is True
        assert prop.cards["median"].evaluation_value is True
        assert prop.cards["nmad"].evaluation_value is True
        assert prop.cards["mean"].evaluation_value is True
        assert prop.cards["std"].evaluation_value is True


def test_property_dist1d_outlier_fraction_matches_thresholds() -> None:
    """Validate the outlier fraction logic for a distribution with an extreme value.

    One value is intentionally far outside the rest of the sample, so the expected
    outlier fraction is 25%. The test checks that the computed fraction and the
    threshold-evaluation boolean both reflect that behavior.
    """
    thresholds = Table({
        "property_name": ["flux_median", "flux_nmad"],
        "min": [0.0, 0.0],
        "max": [10.0, 10.0],
        "unit": ["nJy", "nJy"]
    })
    values = np.array([1.0, 2.0, 3.0, 100.0])*u.nJy
    prop = Property("flux", "dist1d", outlier_thresholds=[0.0, 10.0])
    prop.compute(values)
    prop.evaluate(thresholds)
    assert prop.cards["f_outliers"].data_value == pytest.approx(0.25)
    assert prop.cards["f_outliers"].evaluation_value is True

    prop = Property("flux", "dist1d")
    prop.compute(values)
    prop.evaluate(thresholds)
    assert "f_outliers" not in prop.cards


def test_property_genchi2_summary_and_evaluation() -> None:
    """Confirm the generalized chi-squared summary statistics and threshold logic.

    The test uses a simple increasing sequence with known median absolute error and RMS,
    so the expected summary statistics are easy to compute analytically. It then verifies
    that threshold checks on the median absolute error behave as expected.
    """
    values = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])*u.nJy
    prop = Property("residual", "genchi2")

    prop.compute(values)
    assert prop.cards["n_sources"].data_value == 10
    assert prop.cards["medae"].data_value == pytest.approx(0.55)
    assert prop.cards["mae"].data_value == pytest.approx(0.55)
    assert prop.cards["rmse"].data_value == pytest.approx(0.6205, abs=1e-3)

    assert prop.cards["n_sources"].evaluation_value is True
    assert prop.cards["medae"].evaluation_value is True
    assert prop.cards["mae"].evaluation_value is True
    assert prop.cards["rmse"].evaluation_value is True

    thresholds = Table({
        "property_name": ["residual_medae"],
        "min": [0.0],
        "max": [1.0],
        "unit": ["nJy"]
    })
    prop.evaluate(thresholds)
    assert prop.cards["n_sources"].evaluation_value is True
    assert prop.cards["medae"].evaluation_value is True
    assert prop.cards["mae"].evaluation_value is True
    assert prop.cards["rmse"].evaluation_value is True

    thresholds = Table({
        "property_name": ["residual_medae"],
        "min": [0.0],
        "max": [0.1],
        "unit": ["nJy"]
    })
    prop.evaluate(thresholds)
    assert prop.cards["n_sources"].evaluation_value is True
    assert prop.cards["medae"].evaluation_value is False
    assert prop.cards["mae"].evaluation_value is True
    assert prop.cards["rmse"].evaluation_value is True

def test_property_evaluate_uses_suffix1_name_and_falls_back() -> None:
    """Check that suffix1-based threshold names are preferred and then fall back correctly.

    The Property object can attach a suffix such as "_bright" to summary names, and this
    test verifies the code first looks for the suffixed names before falling back to the
    base property names. It also ensures that a mismatch in the threshold table triggers a
    fail condition in the evaluation flags.
    """
    values = np.arange(1.0, 11.0)*u.nJy
    prop = Property("flux", "dist1d", suffix1='_bright')
    prop.compute(values)

    thresholds = Table({
        "property_name": ["flux_median_bright", "flux_nmad_bright"],
        "min": [-50.0, 100.0],
        "max": [-20.0, 200.0],
        "unit": ["nJy", "nJy"]
        })

    assert prop.cards["median"].data_name == "flux_median_bright"
    assert prop.cards["nmad"].data_name == "flux_nmad_bright"

    assert prop.cards["median"].evaluation_value is True
    assert prop.cards["nmad"].evaluation_value is True
    prop.evaluate(thresholds, suffix2='_v2')
    assert prop.cards["median"].evaluation_value is False
    assert prop.cards["nmad"].evaluation_value is False

    prop = Property("flux", "dist1d", suffix1="_bright")
    prop.compute(values)
    thresholds = Table({
        "property_name": ["flux_median", "flux_nmad"],
        "min": [-50.0, 100.0],
        "max": [-20.0, 200.0],
        "unit": ["nJy", "nJy"],
        })
    prop.evaluate(thresholds, suffix2="_v2")
    assert prop.cards["median"].evaluation_value is False
    assert prop.cards["nmad"].evaluation_value is False


def test_property_evaluate_uses_suffix2_name_and_falls_back() -> None:
    """Check that suffix2-based threshold names are preferred when supplied.

    This covers the case where the threshold table uses a second-level suffix, such as
    "_v2", and confirms the evaluator picks that name before falling back to the base
    property name. The test also verifies that a missing match still produces the expected
    fail state for the evaluated statistics.
    """
    values = np.arange(1.0, 11.0)*u.nJy
    prop = Property("flux", "dist1d")
    prop.compute(values)

    thresholds = Table({
        "property_name": ["flux_median_v2", "flux_nmad_v2"],
        "min": [-50.0, 100.0],
        "max": [-20.0, 200.0],
        "unit": ["nJy", "nJy"],
    })

    assert prop.cards["median"].data_name == "flux_median"
    assert prop.cards["nmad"].data_name == "flux_nmad"

    assert prop.cards["median"].evaluation_value is True
    assert prop.cards["nmad"].evaluation_value is True
    prop.evaluate(thresholds, suffix2='_v2')
    assert prop.cards["median"].evaluation_value is False
    assert prop.cards["nmad"].evaluation_value is False

    prop = Property("flux", "dist1d", suffix1="_bright")
    prop.compute(values)
    thresholds = Table({
        "property_name": ["flux_median", "flux_nmad"],
        "min": [-50.0, 100.0],
        "max": [-20.0, 200.0],
        "unit": ["nJy", "nJy"],
    })
    prop.evaluate(thresholds, suffix2="_v2")
    assert prop.cards["median"].evaluation_value is False
    assert prop.cards["nmad"].evaluation_value is False



def test_property_evaluate_uses_suffix1_suffix2_name_and_falls_back_to_suffix1_then_base() -> None:
    """Validate the full suffix fallback chain: suffix1+suffix2, then suffix1, then base name.

    The evaluator is expected to search for the most specific threshold labels first,
    progressively relaxing to less-specific names if needed. This test covers the full
    chain so the naming logic is documented and protected against regressions.
    """
    values = np.arange(1.0, 11.0)*u.nJy
    prop = Property("flux", "dist1d", suffix1="_bright")
    prop.compute(values)
    assert prop.cards["median"].evaluation_value is True
    assert prop.cards["nmad"].evaluation_value is True

    thresholds = Table({
        "property_name": ["flux_median_bright_v2", "flux_nmad_bright_v2"],
        "min": [-50.0, 100.0],
        "max": [-20.0, 200.0],
        "unit": ["nJy", "nJy"],
    })
    prop.evaluate(thresholds, suffix2="_v2")
    assert prop.cards["median"].evaluation_value is False
    assert prop.cards["nmad"].evaluation_value is False

    prop = Property("flux", "dist1d", suffix1="_bright")
    prop.compute(values)
    thresholds = Table({
        "property_name": ["flux_median_bright", "flux_nmad_bright"],
        "min": [-50.0, 100.0],
        "max": [-20.0, 200.0],
        "unit": ["nJy", "nJy"],
    })
    prop.evaluate(thresholds, suffix2="_v2")
    assert prop.cards["median"].evaluation_value is False
    assert prop.cards["nmad"].evaluation_value is False

    prop = Property("flux", "dist1d", suffix1="_bright")
    prop.compute(values)
    thresholds = Table({
        "property_name": ["flux_median", "flux_nmad"],
        "min": [-50.0, 100.0],
        "max": [-20.0, 200.0],
        "unit": ["nJy", "nJy"],
    })
    prop.evaluate(thresholds, suffix2="_v2")
    assert prop.cards["median"].evaluation_value is False
    assert prop.cards["nmad"].evaluation_value is False
    