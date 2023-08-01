import cv2
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
import torchvision.transforms as t
import random

class WETMetaLoader(Dataset):

    def __init__(self, root: str, annotations: str,
                 nshot_support: int, n_query: int,
                 angles_range: int =28, bin_resolution: int =3,
                 transforms: t.Compose =None) -> None:
        '''
        WET Meta-Learning data loader. Each video is treated as a separate task.

        @Params
        root: str Path to the dataset base directory, where tasks are stored.
        annotations: str Path to the annotations file
        nshot_support: int Number of training samples in the support set (inner loop)
        n_query: int Number of query samples (outer loop)
        transforms: torchvision.transforms.Compose Transforms compose
        '''
        super().__init__()
        # set seed
        np.random.seed(42)

        self._transforms = transforms
        self._root = root

        # Read Annotations [filepath.png yaw pitch]
        self._annotations = pd.read_csv(annotations, header=None,
                                        delimiter=" ", index_col=0,
                                        names=['yaw', 'pitch'])

        self._nshot_support = nshot_support
        self._n_query = n_query

        self._angles_range = angles_range
        self._bin_resolution = bin_resolution

        # read all tasks from the directory
        self._tasks = None
        self._get_tasks()

    def __len__(self):
        return len(self._tasks)

    def __getitem__(self, index):
        task_id = self._tasks[index]
        data_points = os.listdir(os.path.join(self._root, task_id))
        # support_size = round(self._nshot_support / len(data_points), 2)
        # query_set, support_set = train_test_split(data_points,
        #                                           test_size=support_size,
        #                                           random_state=42)
        # query_set = query_set[:self._n_query]
        # support_set = support_set[:self._nshot_support]
        # #! Not random, first support then query
        # support_set = data_points[:self._nshot_support]
        # #! For the benchmarking, take always the same ammount, from the end
        # query_set = data_points[-self._n_query:]
        #! Try random sampling - these are tasks
        support_set = random.sample(data_points, self._nshot_support)
        query_set = list(set(data_points) - set(support_set))
        if len(query_set) > self._n_query:
            query_set = random.sample(query_set, self._n_query)
        # query_set = data_points[self._nshot_support:(self._nshot_support+self._n_query)]


        support_cont = []
        support_binned = []
        support_images = []
        for s in support_set:
            # label
            row_name = os.path.join(task_id, s)
            yaw, pitch = self._annotations.loc[row_name].values
            # Convert yaw and pitch to angles
            pitch = pitch * 180 / np.pi
            yaw = yaw * 180 / np.pi
            bins = np.array(range(-42, 42, 3))
            binned_pose = np.digitize([pitch, yaw], bins) - 1

            cont_labels = [pitch, yaw]


            # image
            img_path = os.path.join(self._root, row_name)
            support_images.append(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0)
            support_cont.append(cont_labels)
            support_binned.append(binned_pose)


        query_cont = []
        query_binned = []
        query_images = []
        for q in query_set:
            # label
            row_name = os.path.join(task_id, q)
            yaw, pitch = self._annotations.loc[row_name].values
            # Convert yaw and pitch to angles
            pitch = pitch * 180 / np.pi
            yaw = yaw * 180 / np.pi
            bins = np.array(range(-42, 42, 3))
            binned_pose = np.digitize([pitch, yaw], bins) - 1

            cont_labels = [pitch, yaw]

            query_cont.append(cont_labels)
            query_binned.append(binned_pose)
            # image
            img_path = os.path.join(self._root, row_name)
            query_images.append(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0)


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

    def _get_tasks(self):
        self._tasks = os.listdir(self._root)
        self._tasks.remove("annotations.txt")
