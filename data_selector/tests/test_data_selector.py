import os
import pytest

from data_selector.data_selector import DataSelector


@pytest.mark.unit
def test_data_selector():
    ds = DataSelector(
        data_folders_filepath=os.path.join(os.path.dirname(__file__), 'data/data_folders.yaml'),
        dataset_config_filepath=os.path.join(os.path.dirname(__file__), 'data/dataset_config.yaml')
    )
    assert ds is not None
