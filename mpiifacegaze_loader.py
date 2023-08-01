import pandas as pd
from torch.utils.data import Dataset
import os
import random
import numpy as np
import cv2
import torch
import torchvision.transforms as t

class MpiiFaceGazeMetaLoader(Dataset):

    LABELS_DIR = "Label"
    IMAGES_DIR = "Image"
    IMAGE_COLUMN_ID = 0
    GAZE2D_COLUMN_ID = 1

    def __init__(self, base_path: str, support_size: int=20, query_size: int=500, transforms=None, exclude=False, one_out=None) -> None:
        super().__init__()
        self._base_path = base_path
        self._support_size = support_size
        self._query_size = query_size
        self._annotations = self.read_annotations(base_path, exclude, one_out)

        self._transforms=transforms

    def read_annotations(self, path, exclude=False, one_out=None):
        YAW_RANGE = 8
        PITCH_RANGE = 13
        labels_path = os.path.join(path, self.LABELS_DIR)
        users = os.listdir(labels_path)
        users = [user for user in users if not user.startswith("_")]
        users = sorted(users)
        if one_out is not None:
            users = [one_out]
        print("USERS: ", users)

        if not exclude:
            return [pd.read_csv(os.path.join(labels_path, user), delimiter=" ") for user in users]
        else:
            annotations = []
            for user in users:
                data = pd.read_csv(os.path.join(labels_path, user), delimiter=" ")
                yaw_pitch = list(data["2DGaze"])
                yaw_pitch = [[float(p.split(",")[0]), float(p.split(",")[1])] for p in yaw_pitch]
                yaw = np.array(yaw_pitch)[:, 0]
                pitch = np.array(yaw_pitch)[:, 1]
                condition = (np.abs(yaw*180/np.pi) < YAW_RANGE) & (np.abs(pitch*180/np.pi) < PITCH_RANGE)
                print(f"Number of valid images for yaw range: {YAW_RANGE}, pitch range: {PITCH_RANGE}: {np.count_nonzero(condition)}/{len(condition)}")
                annotations.append(data[condition])
            return annotations

    def __len__(self):
        return len(self._annotations)

    def __getitem__(self, index):
        annotations = self._annotations[index]
        # query = annotations[self._support_size:self._query_size+self._support_size]
        query = annotations[-self._query_size:]
        support = annotations[:self._support_size]
        # support_indices = random.sample(range(len(support)), self._support_size)
        # support = support.iloc[support_indices, [self.IMAGE_COLUMN_ID, self.GAZE2D_COLUMN_ID]]

        query_images_paths = list(query["Face"])
        query_annotations = [[float(q.split(",")[0]), float(q.split(",")[1])] for q in list(query["2DGaze"])]
        support_images_paths = list(support["Face"])
        support_annotations = [[float(q.split(",")[0]), float(q.split(",")[1])] for q in list(support["2DGaze"])]

        # indexes = list(range(len(annotations)))
        # support_indexes = random.sample(indexes, self._support_size)
        # rest_indexes = list(set(indexes) - set(support_indexes))
        # query_indexes = random.sample(rest_indexes, self._query_size)

        # support = annotations.iloc[support_indexes, [self.IMAGE_COLUMN_ID, self.GAZE2D_COLUMN_ID]]
        # query = annotations.iloc[query_indexes, [self.IMAGE_COLUMN_ID, self.GAZE2D_COLUMN_ID]]
        # query_images_paths = list(query["Face"])
        # query_annotations = [[float(q.split(",")[0]), float(q.split(",")[1])] for q in list(query["2DGaze"])]
        # support_images_paths = list(support["Face"])
        # support_annotations = [[float(q.split(",")[0]), float(q.split(",")[1])] for q in list(support["2DGaze"])]

        support_cont = []
        support_binned = []
        support_images = []

        #    ANNOTATIONS:
        #   yaw     pitch

        for support_img_path, support_annotation in zip(support_images_paths, support_annotations):
            # label
            # Convert yaw and pitch to angles
            pitch = -support_annotation[1] * 180 / np.pi
            yaw = -support_annotation[0] * 180 / np.pi
            bins = np.array(range(-42, 42, 3))
            binned_pose = np.digitize([pitch, yaw], bins) - 1

            cont_labels = [pitch, yaw]

            # image
            img_path = os.path.join(self._base_path, self.IMAGES_DIR, support_img_path)
            try:
                support_images.append(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0)
            except:
                print("SUPPORT SET IMAGE LOADER ERROR")
                continue

            support_cont.append(cont_labels)
            support_binned.append(binned_pose)

        query_cont = []
        query_binned = []
        query_images = []

        for query_img_path, query_annotation in zip(query_images_paths, query_annotations):
            # label
            # Convert yaw and pitch to angles
            # image
            img_path = os.path.join(self._base_path, self.IMAGES_DIR, query_img_path)
            try:
                query_images.append(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0)
            except:
                print("QUERY SET IMAGE LOADER ERROR")
                continue
            pitch = -query_annotation[1] * 180 / np.pi
            yaw = -query_annotation[0] * 180 / np.pi
            bins = np.array(range(-42, 42, 3))
            binned_pose = np.digitize([pitch, yaw], bins) - 1

            cont_labels = [pitch, yaw]

            query_cont.append(cont_labels)
            query_binned.append(binned_pose)

        support_cont = np.array(support_cont)
        support_cont = torch.FloatTensor(support_cont)
        query_cont = np.array(query_cont)
        query_cont = torch.FloatTensor(query_cont)

        support_binned = np.array(support_binned)
        support_binned = torch.FloatTensor(support_binned)
        query_binned = np.array(query_binned)
        query_binned = torch.FloatTensor(query_binned)

        support_images = np.array(support_images)
        support_images = torch.from_numpy(support_images).type(torch.FloatTensor)
        support_images = torch.permute(support_images, (0, 3, 1, 2))
        query_images = np.array(query_images)
        query_images = torch.from_numpy(query_images).type(torch.FloatTensor)
        query_images = torch.permute(query_images, (0, 3, 1, 2))

        if self._transforms is not None:
            support_images = self._transforms(support_images)
            query_images = self._transforms(query_images)

        return support_images, support_cont, support_binned,\
               query_images, query_cont, query_binned
