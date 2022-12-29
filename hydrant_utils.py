import argparse
from collections import OrderedDict
import numpy as np
import os
from PIL import Image
import torch
from torch import nn
from torchvision.models.efficientnet import efficientnet_b2, EfficientNet_B2_Weights
from torchvision.models.efficientnet import efficientnet_b0, EfficientNet_B0_Weights
from torchvision.models.efficientnet import efficientnet_b3, EfficientNet_B3_Weights
from torchvision import transforms
from torch.utils.data.dataset import Dataset


class Hydrant(nn.Module):
    def __init__(self, architecture: str, num_bins_yaw: int, num_bins_pitch: int):
        super().__init__()
        if architecture == 'efficientnet_b0':
            self.net = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
            self.n_features = self.net.classifier[1].in_features
            self.net.classifier = nn.Identity()
            self.net.fc_yaw_gaze = nn.Linear(self.n_features, num_bins_yaw)
            self.net.fc_pitch_gaze = nn.Linear(self.n_features, num_bins_pitch)
            self.net.fc_yaw_reg = nn.Sequential(OrderedDict([('dropout2', nn.Dropout(p=0.2)),
                                                             ('final', nn.Linear(self.n_features, 1))]))
            self.net.fc_pitch_reg = nn.Sequential(OrderedDict([('dropout2', nn.Dropout(p=0.2)),
                                                               ('final', nn.Linear(self.n_features, 1))]))
        elif architecture == 'efficientnet_b2':
            self.net = efficientnet_b2(weights=EfficientNet_B2_Weights.IMAGENET1K_V1)
            self.n_features = self.net.classifier[1].in_features
            self.net.classifier = nn.Identity()
            self.net.fc_yaw_gaze = nn.Linear(self.n_features, num_bins_yaw)
            self.net.fc_pitch_gaze = nn.Linear(self.n_features, num_bins_pitch)
        elif architecture == 'efficientnet_b3':
            self.net = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
            self.n_features = self.net.classifier[1].in_features
            self.net.classifier = nn.Identity()
            self.net.fc_yaw_gaze = nn.Linear(self.n_features, num_bins_yaw)
            self.net.fc_pitch_gaze = nn.Linear(self.n_features, num_bins_pitch)
            self.net.fc_yaw_reg = nn.Sequential(OrderedDict([('dropout2', nn.Dropout(p=0.2)),
                                                             ('final', nn.Linear(self.n_features, 1))]))
            self.net.fc_pitch_reg = nn.Sequential(OrderedDict([('dropout2', nn.Dropout(p=0.2)),
                                                               ('final', nn.Linear(self.n_features, 1))]))

        else:
            raise ValueError(f"{architecture} is not implemented yet.")


    def forward(self, x):
        # TODO: Check if double pass is required or self.net(x) can be computed once
        yaw_head = self.net.fc_yaw_gaze(self.net(x))
        pitch_head = self.net.fc_pitch_gaze(self.net(x))
        yaw_head_reg = self.net.fc_yaw_reg(self.net(x))
        pitch_head_reg = self.net.fc_pitch_reg(self.net(x))


        return yaw_head, pitch_head, yaw_head_reg, pitch_head_reg

class GazeCaptureDifferentRanges(Dataset):
    '''
    GazeCapture DataLoader.
    '''
    def __init__(self, annotations: str, root: str, transform: transforms.Compose =None, flip_signs=False,
                 pitch_angle_lower_range: int =-42, pitch_angle_upper_range: int =42, pitch_degrees_per_bin: int =3,
                 yaw_angle_lower_range: int =-42, yaw_angle_upper_range: int =42, yaw_degrees_per_bin: int =3):
        '''
        Initialization.

        Parameters

        Input:
        annotations: str Annotations filepath
        root: str Path to the dataset base directory.
        transform: torchvision.transforms.Compose Image transform. Can be None
        flip_signs: flip signs in yaw and pitch labels
        '''

        self._root = root
        self._transform = transform
        self._flip_signs = flip_signs

        self._pitch_angle_lower_range = pitch_angle_lower_range
        self._pitch_angle_upper_range = pitch_angle_upper_range
        self._pitch_degrees_per_bin = pitch_degrees_per_bin

        self._yaw_angle_lower_range = yaw_angle_lower_range
        self._yaw_angle_upper_range = yaw_angle_upper_range
        self._yaw_degrees_per_bin = yaw_degrees_per_bin


        # Read Annotations [filepath.png yaw pitch]
        with open(annotations, 'r') as f:
            self._data = f.readlines()
        # Remove \n from the end of each line and empty last line
        self._data = list(map(str.strip, self._data))[:-1]

    def __len__(self):
        # Length of annotations
        return len(self._data)

    def __getitem__(self, idx):
        # Get the annotation
        annotation = self._data[idx]

        # Split [filepath.png yaw pitch]
        img_path, yaw, pitch = annotation.split(" ")

        # Convert to Tensor
        label = np.array([yaw, pitch]).astype("float")
        if self._flip_signs:
            label[0] *= -1
            label[1] *= -1
        label = torch.from_numpy(label).type(torch.FloatTensor)

        # Load image
        img = Image.open(os.path.join(self._root, img_path))
        # Apply Transform if not None
        if self._transform:
            img = self._transform(img)

        # Convert yaw and pitch to angles
        yaw = label[0] * 180 / np.pi
        pitch = label[1] * 180 / np.pi

        # Binarize Values
        bins_pitch = np.array(range(self._pitch_angle_lower_range, self._pitch_angle_upper_range, self._pitch_degrees_per_bin))
        bins_yaw = np.array(range(self._yaw_angle_lower_range, self._yaw_angle_upper_range, self._yaw_degrees_per_bin))

        binned_pitch = np.digitize([pitch], bins_pitch) - 1
        binned_yaw = np.digitize([yaw], bins_yaw) - 1
        binned_pose_stacked = np.hstack((binned_yaw, binned_pitch))


        labels = binned_pose_stacked
        cont_labels = torch.FloatTensor([yaw, pitch])

        return img, labels, cont_labels

def gauss(x: np.ndarray, mu: float, sigma: float):
    '''
    Computes gaussian function for given values of X.

    Input:
    x: np.ndarray domain
    mu: float expected value
    sigma: float standard deviatian
    '''
    return 1 / np.sqrt(2*np.pi * sigma * sigma) * np.exp((-(x - mu)**2)/(2*sigma*sigma))

def smooth_labels(bin_numbers: np.ndarray,  sigma: float, bin_count: int, threshold: float =1e-3):
    '''
    Applies gaussian smoothing to the classification ground truth.

    Input:
    bin_numbers: np.ndarray ground truth labels (expected values for gaussian)
    sigma: float standard deviation
    bin_count: total number of classes
    threshold: float computed smooth values lower than the threshold are set to 0

    Output:
    tensor of size bin_numbers x bin_count

    '''

    x = np.arange(0, bin_count, 1)
    smoothed_labels = []
    for bin_number in bin_numbers:
        smoothed_bins = (gauss(x, bin_number, sigma) > threshold).astype(int) * gauss(x, bin_number, sigma)
        smoothed_bins /= sum(smoothed_bins)
        smoothed_labels.append(smoothed_bins)

    smoothed_labels = np.array(smoothed_labels)
    return torch.from_numpy(smoothed_labels)
