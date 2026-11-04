import os
from pathlib import Path
from unittest.mock import Mock, call

import pytest

from rdmt_spire.utilities.aws_utils import (
    fetch_parameters_from_path,
    file_exists,
    load_file_object,
)


@pytest.mark.parametrize("path", ["/app/config", "/app/config/"])
def test_fetch_parameters_requests_only_expected_names(path):
    client = Mock()
    client.get_parameters.return_value = {
        "Parameters": [
            {"Name": "/app/config/nested/second", "Value": "two"},
            {"Name": "/app/config/first", "Value": "one"},
        ],
        "InvalidParameters": [],
    }

    result = fetch_parameters_from_path(path, ["first", "nested/second"], client)

    assert result == {"first": "one", "nested/second": "two"}
    client.get_parameters.assert_called_once_with(
        Names=["/app/config/first", "/app/config/nested/second"],
        WithDecryption=True,
    )
    client.get_paginator.assert_not_called()


@pytest.mark.parametrize("count", [1, 10, 11, 21])
def test_fetch_parameters_batches_at_most_ten_names(count):
    client = Mock()
    names = [f"param{i}" for i in range(count)]

    def get_parameters(**kwargs):
        return {"Parameters": [{"Name": name, "Value": name} for name in kwargs["Names"]]}

    client.get_parameters.side_effect = get_parameters

    result = fetch_parameters_from_path("/config/", names, client)

    assert result == {name: f"/config/{name}" for name in names}
    client.get_parameters.assert_has_calls([
        call(Names=[f"/config/{name}" for name in names[start:start + 10]], WithDecryption=True)
        for start in range(0, count, 10)
    ])
    assert client.get_parameters.call_count == (count + 9) // 10


def test_fetch_parameters_raises_for_missing_names():
    client = Mock()
    client.get_parameters.return_value = {
        "Parameters": [{"Name": "/config/present", "Value": "value"}],
        "InvalidParameters": ["/config/missing"],
    }

    with pytest.raises(KeyError, match="Missing expected SSM parameters.*missing"):
        fetch_parameters_from_path("/config/", ["present", "missing"], client)


def test_fetch_parameters_empty_list_does_not_create_client(monkeypatch):
    create_client = Mock()
    monkeypatch.setattr("rdmt_spire.utilities.aws_utils.boto3.client", create_client)

    assert fetch_parameters_from_path("/config/", []) == {}
    create_client.assert_not_called()


def test_fetch_parameters_deduplicates_names_without_caching():
    client = Mock()
    client.get_parameters.side_effect = [
        {"Parameters": [{"Name": "/config/name", "Value": "first"}]},
        {"Parameters": [{"Name": "/config/name", "Value": "updated"}]},
    ]

    assert fetch_parameters_from_path("/config/", ["name", "name"], client) == {"name": "first"}
    assert fetch_parameters_from_path("/config/", ["name"], client) == {"name": "updated"}
    assert client.get_parameters.call_args_list == [
        call(Names=["/config/name"], WithDecryption=True),
        call(Names=["/config/name"], WithDecryption=True),
    ]


def test_load_file_object_local_file1(tmp_path):
    # Check a local file with path beginning with slash
    local_file = tmp_path / "mykey"
    local_file.write_text("content")

    result = load_file_object(str(local_file.parent), local_file.name)
    assert result.read() == b"content"

def test_load_file_object_local_file2(tmp_path, monkeypatch):
    # Check a local file whose path does not being with a slash (i.e., relative path)
    data_dir = tmp_path / "test_data"
    data_dir.mkdir()
    target = data_dir / "check.txt"
    target.write_text("abc")    
    monkeypatch.chdir(tmp_path)  # Change the current working directory to the temp path
    local_file = Path(os.path.join('test_data', 'check.txt'))

    result = load_file_object(str(local_file.parent), local_file.name)
    assert result.read() == b"abc"


def test_file_exists_local_file1(tmp_path, monkeypatch):
    # Check a local file whose path does not being with a slash (i.e., relative path)
    data_dir = tmp_path / "test_data"
    data_dir.mkdir()
    target = data_dir / "check.txt"
    target.write_text("abc")    
    monkeypatch.chdir(tmp_path)  # Change the current working directory to the temp path
    local_file = Path(os.path.join('test_data', 'check.txt'))
    assert file_exists(str(local_file.parent), local_file.name) is True

def test_file_exists_local_file2(tmp_path):
    # Check a local file with path beginning with slash
    local_file = tmp_path / "mykey"
    local_file.write_text("content")
    assert file_exists(str(local_file.parent), local_file.name) is True

