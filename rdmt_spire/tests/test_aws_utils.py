import os
from pathlib import Path

import pytest
from astropy.table import Table

from rdmt_spire.utilities.aws_utils import csv2qtable, file_exists, load_file_object


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

def test_file_exists_local_file2(tmp_path, monkeypatch):
    # Check a local file whose path does not being with a slash (i.e., relative path)
    data_dir = tmp_path / "test_data"
    data_dir.mkdir()
    monkeypatch.chdir(tmp_path)  # Change the current working directory to the temp path

    df=Table({"a": [1, 3], "b": [2, 4]})
    local_file = Path(os.path.join('test_data', 'check.ecsv'))
    df.write(str(local_file), format="ascii.ecsv")
    table = csv2qtable(str(local_file.parent), local_file.name)
    print(table)
    assert len(table) == 2
    assert 'a' in table.colnames
    assert 'b' in table.colnames

    local_file = Path(os.path.join('test_data', 'check.csv'))
    df.write(str(local_file), format="ascii.csv")
    table = csv2qtable(str(local_file.parent), local_file.name)
    print(table)
    assert len(table) == 2
    assert 'a' in table.colnames
    assert 'b' in table.colnames



@pytest.mark.skip(reason="S3 access required for this test")
def test_s3_access():
    bucket = "s3://stpubdata"
    key="gaia/gaia_dr3/public/hats/gaia/partition_info.csv"
    from botocore import UNSIGNED
    from botocore.config import Config

    assert file_exists(bucket, key,config=Config(signature_version=UNSIGNED)) is True

    content = load_file_object(bucket, key,config=Config(signature_version=UNSIGNED)).read()
    assert len(content) > 0

    table = csv2qtable(bucket, key,config=Config(signature_version=UNSIGNED))
    assert 'norder' in table.colnames
    assert 'npix' in table.colnames
    assert len(table) > 0


