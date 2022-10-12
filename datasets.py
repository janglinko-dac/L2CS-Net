import os
import numpy as np
import cv2


import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from PIL import Image, ImageFilter


class Gaze360(Dataset):
    def __init__(self, path, root, transform, angle, binwidth, train=True):
        self.transform = transform
        self.root = root
        self.orig_list_len = 0
        self.angle = angle
        if train==False:
          angle=90
        self.binwidth=binwidth
        self.lines = []
        if isinstance(path, list):
            for i in path:
                with open(i) as f:
                    print("here")
                    line = f.readlines()
                    line.pop(0)
                    self.lines.extend(line)
        else:
            with open(path) as f:
                lines = f.readlines()
                lines.pop(0)
                self.orig_list_len = len(lines)
                for line in lines:
                    gaze2d = line.strip().split(" ")[5]
                    label = np.array(gaze2d.split(",")).astype("float")
                    if abs((label[0]*180/np.pi)) <= angle and abs((label[1]*180/np.pi)) <= angle:
                        self.lines.append(line)


        print("{} items removed from dataset that have an angle > {}".format(self.orig_list_len-len(self.lines), angle))

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        line = self.lines[idx]
        line = line.strip().split(" ")

        face = line[0]
        lefteye = line[1]
        righteye = line[2]
        name = line[3]
        gaze2d = line[5]
        label = np.array(gaze2d.split(",")).astype("float")
        label = torch.from_numpy(label).type(torch.FloatTensor)

        pitch = label[0]* 180 / np.pi
        yaw = label[1]* 180 / np.pi

        img = Image.open(os.path.join(self.root, face))

        # fimg = cv2.imread(os.path.join(self.root, face))
        # fimg = cv2.resize(fimg, (448, 448))/255.0
        # fimg = fimg.transpose(2, 0, 1)
        # img=torch.from_numpy(fimg).type(torch.FloatTensor)

        if self.transform:
            img = self.transform(img)

        # Bin values
        bins = np.array(range(-1*self.angle, self.angle, self.binwidth))
        binned_pose = np.digitize([pitch, yaw], bins) - 1

        labels = binned_pose
        cont_labels = torch.FloatTensor([pitch, yaw])


        return img, labels, cont_labels, name

class Mpiigaze(Dataset):
  def __init__(self, pathorg, root, transform, train, angle,fold=0):
    self.transform = transform
    self.root = root
    self.orig_list_len = 0
    self.lines = []
    path=pathorg.copy()
    if train==True:
      # path.pop(fold) # Nie usuwamy użytkownika
      pass
    else:
      path=path[fold]
    if isinstance(path, list):
        for i in path:
            with open(i) as f:
                lines = f.readlines()
                lines.pop(0)
                self.orig_list_len += len(lines)
                for line in lines:
                    gaze2d = line.strip().split(" ")[7]
                    label = np.array(gaze2d.split(",")).astype("float")
                    if abs((label[0]*180/np.pi)) <= angle and abs((label[1]*180/np.pi)) <= angle:
                        self.lines.append(line)
    else:
      with open(path) as f:
        lines = f.readlines()
        lines.pop(0)
        self.orig_list_len += len(lines)
        for line in lines:
            gaze2d = line.strip().split(" ")[7]
            label = np.array(gaze2d.split(",")).astype("float")
            if abs((label[0]*180/np.pi)) <= 42 and abs((label[1]*180/np.pi)) <= 42:
                self.lines.append(line)

    print("{} items removed from dataset that have an angle > {}".format(self.orig_list_len-len(self.lines),angle))

  def __len__(self):
    return len(self.lines)

  def __getitem__(self, idx):
    line = self.lines[idx]
    line = line.strip().split(" ")

    name = line[3]
    gaze2d = line[7]
    head2d = line[8]
    lefteye = line[1]
    righteye = line[2]
    face = line[0]

    label = np.array(gaze2d.split(",")).astype("float")
    label = torch.from_numpy(label).type(torch.FloatTensor)


    pitch = label[0]* 180 / np.pi
    yaw = label[1]* 180 / np.pi

    img = Image.open(os.path.join(self.root, face))

    # fimg = cv2.imread(os.path.join(self.root, face))
    # fimg = cv2.resize(fimg, (448, 448))/255.0
    # fimg = fimg.transpose(2, 0, 1)
    # img=torch.from_numpy(fimg).type(torch.FloatTensor)

    if self.transform:
        img = self.transform(img)

    # Bin values
    bins = np.array(range(-42, 42,3))
    binned_pose = np.digitize([pitch, yaw], bins) - 1

    labels = binned_pose
    cont_labels = torch.FloatTensor([pitch, yaw])


    return img, labels, cont_labels, name


class GazeCapture(Dataset):
    '''
    GazeCapture DataLoader.
    '''
    def __init__(self, annotations: str, root: str, transform: transforms.Compose =None, flip_signs=False):
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

        # Read Annotations [filepath.png pitch yaw]
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
        label = np.array([pitch, yaw]).astype("float")
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
        pitch = label[0] * 180 / np.pi
        yaw = label[1] * 180 / np.pi

        # Binarize Values
        bins = np.array(range(-42, 42, 3))
        binned_pose = np.digitize([pitch, yaw], bins) - 1

        labels = binned_pose
        cont_labels = torch.FloatTensor([pitch, yaw])

        return img, labels, cont_labels
