import json
from dataclasses import asdict

import asdf
import pytest

from ..constants.codes import StatusCodes
from ..manager import MonitorManager
from ..monitors.noise_1f import Noise1fMonitor


def test_monitor_manager(monkeypatch):
    """
    Testing class MonitorManager
    """
    monitor_manager = MonitorManager(asdf.AsdfFile(), "noise_1f")
    monitor = monitor_manager.monitor_objects[0]
    monitor.append_data("sample_metric", 1, "")

    # set run to none so it doesn't actually execute any processing
    monkeypatch.setattr(monitor, "run", lambda: None)
    
    monitor_manager.process()
    assert monitor_manager.statusCode == StatusCodes.SUCCESS
    assert len(monitor_manager.monitor_objects) == 1
    assert monitor.monitor_name == "noise_1f"

    # Checking class MonitorManager.response_for_lambda()
    response = monitor_manager.response_for_lambda()
    assert response["statusCode"] == StatusCodes.SUCCESS
    data_cards = monitor_manager.monitor_objects[0].get_data_card("all")
    body = json.loads(response["body"])
    assert len(body) == len(data_cards)
    for i, data_card in enumerate(data_cards):
        assert body[i] == asdict(data_card)


def test_monitor_manager_constructor_errors():
    """Invalid input raises; unknown monitors are reported in the Lambda response."""
    with pytest.raises(ValueError, match="Invalid input file"):
        MonitorManager("", "base_monitor")

    monitor_manager = MonitorManager(asdf.AsdfFile(), "non_existing_monitor")
    assert monitor_manager.statusCode == StatusCodes.FAILURE
    response = monitor_manager.response_for_lambda()
    body = json.loads(response["body"])
    assert "error_message" in body[0]


def test_monitor_manager_double(monkeypatch):
    """
    Processing multiple monitors continues when each monitor succeeds.
    """
    monitor_config = {"astrometry": {"datadir": ""}}
    monitor_manager = MonitorManager(
        asdf.AsdfFile(), "astrometry", monitor_config=monitor_config
    )
    monitor_manager.monitor_objects.append(Noise1fMonitor(asdf.AsdfFile()))
    for monitor in monitor_manager.monitor_objects:
        monitor.append_data(f"{monitor.monitor_name}_metric", 1, "")
        monkeypatch.setattr(monitor, "run", lambda: None)

    assert len(monitor_manager.monitor_objects) == 2
    assert monitor_manager.monitor_objects[0].monitor_name == "astrometry"
    assert monitor_manager.monitor_objects[1].monitor_name == "noise_1f"

    monitor_manager.process()
    assert monitor_manager.statusCode == StatusCodes.SUCCESS
    data_cards = []
    for monitor_obj in monitor_manager.monitor_objects:
        data_cards.extend(monitor_obj.get_data_card("all"))
    assert len(data_cards) == 2


@pytest.mark.parametrize("error_type", [RuntimeError, ValueError])
def test_monitor_manager_process_propagates_monitor_errors(monkeypatch, error_type):
    """Monitor exceptions propagate; process() no longer translates them to status codes."""
    monitor_manager = MonitorManager(
        asdf.AsdfFile(),
        "astrometry",
        monitor_config={"astrometry": {"datadir": ""}},
    )
    monitor = monitor_manager.monitor_objects[0]

    def fail_run():
        raise error_type("monitor failed")

    monkeypatch.setattr(monitor, "run", fail_run)

    with pytest.raises(error_type, match="monitor failed"):
        monitor_manager.process()

    # Status/error conversion belongs to the caller now that process() does not catch.
    assert monitor_manager.statusCode == StatusCodes.SUCCESS
    assert monitor_manager.errors == []
