import math
from operator import attrgetter
import os
from pathlib import Path
import random
from typing import Tuple
import yaml

from data_selector.image_data import ImageData


class DataSelector():
    """Class that lets you generate bash scripts that will create "virtual" folders
    that can be used by the L2CS-Net data loaders. No data is copied anywhere, the original
    folders stay the same, but instead symbolic links are created and annotations files
    are accordingly modified.
    """

    ANNOTATIONS_FILENAME = "annotations.txt"

    def __init__(self, data_folders_filepath: str, dataset_config_filepath: str):
        """Class constructor

        Args:
            data_folders_filepath: YAML file that will contain paths to folders that include
                                   already prepared GazeCapture-compatible data folders
                                   (folders with images + annotations.txt)
            dataset_config_filepath: YAML file that contains description on how the new
                                     dataset should be selected for training and validation
                                     datasets
        """

        assert os.path.isfile(data_folders_filepath)
        assert os.path.isfile(dataset_config_filepath)

        # load contents of data folders YAML file
        with open(data_folders_filepath) as f:
            self._data_folders_dict = yaml.safe_load(f)

        # load contents of dataset config YAML file
        with open(dataset_config_filepath) as f:
            self._dataset_config_dict = yaml.safe_load(f)

        # create a list of all available images and corresponding yaw & pitch labels
        self._image_data = self.__generate_data_list()

    @staticmethod
    def __get_annotations_filepath(data_folderpath: str) -> str:
        """Extracts path to the annotations file corresponding to the data folder

        Args:
            data_folderpath: path to the data folder in which the annotations txt file should be stored

        Returns:
            filepath to annotations.txt file
        """

        annotations_filepath = os.path.join(data_folderpath, DataSelector.ANNOTATIONS_FILENAME)
        assert os.path.isfile(annotations_filepath)
        return annotations_filepath

    @staticmethod
    def __get_labels(image_path: str, annotations_filepath: str) -> Tuple[float, float]:
        """Extracts expected pitch & yaw labels for specific image path

        Args:
            image_path: path to image
            annotations_filepath: path to annotations file in which to look for labels

        Returns:
            _description_
        """

        # parse annotations file to look for a matching entry for image
        with open(annotations_filepath, 'r') as f:
            lines = f.readlines()

        entry = image_path.split('/')[-2:]
        entry = '/'.join(entry)
        matching_lines = [x for x in lines if entry in x]
        assert len(matching_lines) == 1
        pitch = float(matching_lines[0].split()[1])
        yaw = float(matching_lines[0].split()[2])

        return pitch, yaw

    def __generate_data_list(self) -> list:
        """Generates image data list

        Returns:
            list of ImageData objects containing image path, user ID, yaw & pitch, etc.
        """
        # parse all data folders to get the list of images and corresponding yaw and pitch labels

        image_data = []

        # images = [x for x in glob(f"{self._data_folders_dict['data_folders']}/*.png")]
        for data_folder in self._data_folders_dict['data_folders']:
            # extract annotations filepath
            annotations_filepath = self.__get_annotations_filepath(data_folder)
            for root, _, files in os.walk(data_folder):
                for file in files:
                    if file.endswith('.png'):
                        # save image path
                        image_path = os.path.join(root, file)
                        # extract user ID
                        user_id = os.path.basename(Path(image_path).parent)
                        # extract yaw and pitch label
                        pitch, yaw = self.__get_labels(image_path, annotations_filepath)
                        single_image_data = ImageData(image_path=image_path, user_id=user_id, pitch=pitch, yaw=yaw)
                        image_data.append(single_image_data)

        return image_data

    def __get_user_ids(self) -> list:
        """Gets a list of unique user IDs

        Returns:
            list of unique user IDs in all data folders
        """

        # get the list of unique user IDs
        unique_user_ids = set()
        [unique_user_ids.add(obj.user_id) or obj for obj in self._image_data if obj.user_id not in unique_user_ids]
        return list(unique_user_ids)

    def get_data_statistics(self) -> Tuple[float, float, float, float, list, int]:
        """Calculates a few statistics of the data

        Returns:
            min. value of pitch across the data, max. value of pitch, min. value of yaw, max. value of yaw
        """

        min_pitch = math.degrees(min(self._image_data, key=attrgetter('pitch')).pitch)
        max_pitch = math.degrees(max(self._image_data, key=attrgetter('pitch')).pitch)

        min_yaw = math.degrees(min(self._image_data, key=attrgetter('yaw')).yaw)
        max_yaw = math.degrees(max(self._image_data, key=attrgetter('yaw')).yaw)

        user_ids = self.__get_user_ids()
        total_number_of_images = len(self._image_data)

        return min_pitch, max_pitch, min_yaw, max_yaw, user_ids, total_number_of_images

    @staticmethod
    def __generate_symlinks_and_annotations(output_dataset_folder: str, image_data: list) -> None:
        """Generates symbolic links to images and annotations txt file necessary for training

        Args:
            output_dataset_folder: target folder where to place symbolic links annotations
            image_data: list of ImageData objects
        """

        # generate symbolic links & annotations files train and validation datasets
        with open(os.path.join(output_dataset_folder, DataSelector.ANNOTATIONS_FILENAME), 'w') as f:
            for single_image_data in image_data:
                src = os.path.abspath(single_image_data.image_path)
                dst = os.path.abspath(os.path.join(output_dataset_folder, single_image_data.user_id, os.path.basename(single_image_data.image_path)))

                # create parent folder for the symlink if it does not exist
                dst_parent = Path(dst).parent
                if not os.path.isdir(dst_parent):
                    Path(dst_parent).mkdir(parents=True, exist_ok=True)

                # create symbolic link to the image data
                os.symlink(src=src, dst=dst)

                # update annotations file
                image_path = f'{single_image_data.user_id}/{os.path.basename(single_image_data.image_path)}'
                f.write(f'{image_path} {single_image_data.pitch} {single_image_data.yaw}\n')

    def generate_dataset(self, train_dataset_folder: str, val_dataset_folder: str) -> None:
        """Generates new train and validation dataset based on the configuration file

        Args:
            train_dataset_folder: output folder to store train dataset symbolic links & annotations file
            val_dataset_folder: output folder to store validation dataset symbolic links & annotations file
        """

        assert train_dataset_folder != val_dataset_folder

        # check if the output folders exist, if they are not empty - abort
        if os.path.isdir(train_dataset_folder) and os.listdir(train_dataset_folder):
            raise RuntimeError(f'Output train dataset folder {train_dataset_folder} is not empty')

        if os.path.isdir(val_dataset_folder) and os.listdir(val_dataset_folder):
            raise RuntimeError(f'Output validation dataset folder {val_dataset_folder} is not empty')

        # create both output folders if they don't exist
        Path(train_dataset_folder).mkdir(parents=True, exist_ok=True)
        Path(val_dataset_folder).mkdir(parents=True, exist_ok=True)

        if self._dataset_config_dict['selection_strategy'] == 'random':
            # randomly select data from the pool
            # make sure that users that fall to train dataset are not present in the validation dataset

            assert self._dataset_config_dict['train_ratio'] < 1.0

            # select user IDs for training and validation dataset according to ratio
            user_ids = self.__get_user_ids()
            total_user_count = len(user_ids)
            train_user_count = int(round(total_user_count * self._dataset_config_dict['train_ratio']))

            # select user IDs for train & validation dataset
            train_user_ids = random.sample(user_ids, train_user_count)
            val_user_ids = [user_id for user_id in user_ids if user_id not in train_user_ids]

            assert len(train_user_ids) + len(val_user_ids) == len(user_ids)

            # extract image data that will go into train and validation dataset
            train_image_data = [image_data for image_data in self._image_data if image_data.user_id in train_user_ids]
            val_image_data = [image_data for image_data in self._image_data if image_data.user_id in val_user_ids]

            # generate symbolic links & annotations files train and validation datasets
            self.__generate_symlinks_and_annotations(train_dataset_folder, train_image_data)
            self.__generate_symlinks_and_annotations(val_dataset_folder, val_image_data)
