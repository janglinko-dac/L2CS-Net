import os
import shutil
import pytest

from data_selector.data_selector import DataSelector


@pytest.mark.unit
def test_data_selector_init_ok():
    ds = DataSelector(
        data_folders_filepath=os.path.join(os.path.dirname(__file__), 'data/data_folders.yaml'),
        dataset_config_filepath=os.path.join(os.path.dirname(__file__), 'data/dataset_config.yaml')
    )
    assert ds is not None


@pytest.mark.unit
def test_data_selector_get_data_statistics_ok():
    ds = DataSelector(
        data_folders_filepath=os.path.join(os.path.dirname(__file__), 'data/data_folders.yaml'),
        dataset_config_filepath=os.path.join(os.path.dirname(__file__), 'data/dataset_config.yaml')
    )
    data_statistics = ds.get_data_statistics()
    assert data_statistics is not None
    assert len(data_statistics[4]) == 13


@pytest.mark.unit
def test_data_selector_generate_dataset_ok():
    train_dataset_folder = os.path.join(os.path.dirname(__file__), 'data/new_train_dataset')
    val_dataset_folder = os.path.join(os.path.dirname(__file__), 'data/new_val_dataset')

    # remove previous version of output folders (if present)
    if os.path.isdir(train_dataset_folder):
        shutil.rmtree(train_dataset_folder)

    if os.path.isdir(val_dataset_folder):
        shutil.rmtree(val_dataset_folder)

    ds = DataSelector(
        data_folders_filepath=os.path.join(os.path.dirname(__file__), 'data/data_folders.yaml'),
        dataset_config_filepath=os.path.join(os.path.dirname(__file__), 'data/dataset_config.yaml')
    )
    ds.generate_dataset(train_dataset_folder, val_dataset_folder)
