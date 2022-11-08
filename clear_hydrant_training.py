import argparse
import clearml
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
from torch.utils.data import DataLoader
from torch.autograd import Variable



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


train_transformations = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                 std=[0.229, 0.224, 0.225])])
                                                           

val_transformations = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                           std=[0.229, 0.224, 0.225])])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Gaze estimation using Hydrant.')
    # ------DATASET ARGS------
    parser.add_argument('--train-dir', help='Directory path for training dataset.', required=True)
    parser.add_argument('--val-dir', help='Directory path for validation dataset.', required=True)
    # ------NETWORK ARCHITECTURE------
    parser.add_argument('--architecture', help='Root network architecture', required=True)
    # ------BINS ARGUMENTS------
    parser.add_argument('--pitch-lower-range', help='Pitch lower range', default=-42, type=int)
    parser.add_argument('--pitch-upper-range', help='Pitch upper range', default=42, type=int)
    parser.add_argument('--pitch-resolution', help='Bin size in degrees for pitch', default=3, type=int)
    parser.add_argument('--yaw-lower-range', help='Yaw lower range', default=-42, type=int)
    parser.add_argument('--yaw-upper-range', help='Yaw upper range', default=42, type=int)
    parser.add_argument('--yaw-resolution', help='Bin size in degrees for yaw', default=3, type=int)
    # ------LOSS ARGUMENTS------
    parser.add_argument('--pitch-cls-scale', help='Scaling factor for pitch classification loss', default=1.0, type=float)
    parser.add_argument('--pitch-reg-scale', help='Scaling factor for pitch regression loss', default=1.0, type=float)
    parser.add_argument('--yaw-cls-scale', help='Scaling factor for yaw classification loss', default=1.0, type=float)
    parser.add_argument('--yaw-reg-scale', help='Scaling factor for yaw regression loss', default=1.0, type=float)
    # ------LABEL SMOOTHING ARGUMENTS------
    parser.add_argument('--label-smoothing', help='Enables label smoothing', action='store_true')
    parser.add_argument('--smoothing-sigma', help='Standard deviation for gaussian smoothing', default=0.45, type=float)
    parser.add_argument('--smoothing-threshold', help='Threshold for non-zeroing smoothed values', default=1e-3, type=float)
    # ------CLEARML ARGUMENTS------
    parser.add_argument('--clearml-experiment', help='Name of the experiment', required=True)
    parser.add_argument('--clearml-tags', nargs='*', help='tags that will be used by ClearML', required=True)
    # ------TRAINING ARGUMENTS------
    parser.add_argument('--lr', help='Learning ratio', default=1e-5, type=float)
    parser.add_argument('--epochs', help='Number of epochs', default=20, type=int)
    parser.add_argument('--batch-size', help='Batch size', default=8, type=int)
    # ------OUTPUT ARGUMENTS------
    parser.add_argument('--output-folder', help='Folder where models will be saved', required=True)

    args = parser.parse_args()


    task = clearml.Task.init(project_name="WET", task_name=args.clearml_experiment, tags=args.clearml_tags)
    logger = task.get_logger()


    # Calculate bins number
    if (args.pitch_upper_range - args.pitch_lower_range) % args.pitch_resolution:
        raise ValueError("Pitch range must be divisible by pitch resolution")
    else:
        bin_number_pitch = int((args.pitch_upper_range - args.pitch_lower_range) / args.pitch_resolution)
    
    if (args.yaw_upper_range - args.yaw_lower_range) % args.yaw_resolution:
        raise ValueError("yaw range must be divisible by Yaw resolution")
    else:
        bin_number_yaw = int((args.yaw_upper_range - args.yaw_lower_range) / args.yaw_resolution)


    # Define device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Build model
    model = Hydrant(args.architecture, bin_number_yaw, bin_number_pitch)
    model.to(device)

    # Define datasets
    train_dataset=GazeCaptureDifferentRanges(os.path.join(args.train_dir, 'annotations.txt'),
                                             args.train_dir,
                                             train_transformations, 
                                             False,
                                             args.pitch_lower_range, 
                                             args.pitch_upper_range,
                                             args.pitch_resolution, 
                                             args.yaw_lower_range,
                                             args.yaw_upper_range,
                                             args.yaw_resolution)

    val_dataset=GazeCaptureDifferentRanges(os.path.join(args.val_dir, 'annotations.txt'),
                                           args.val_dir,
                                           val_transformations, 
                                           False,
                                           args.pitch_lower_range, 
                                           args.pitch_upper_range,
                                           args.pitch_resolution, 
                                           args.yaw_lower_range,
                                           args.yaw_upper_range,
                                           args.yaw_resolution)

    # Define dataloaders
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=1,
                                  pin_memory=True)

    val_dataloader = DataLoader(dataset=val_dataset,
                                batch_size=args.batch_size,
                                shuffle=True,
                                num_workers=1,
                                pin_memory=True)

    # Define losses
    classification_loss = nn.CrossEntropyLoss()
    regression_loss = nn.L1Loss()
    quadratic_loss = nn.MSELoss()
    
    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Define training parameters
    n_epochs = args.epochs
    alpha_yaw = args.yaw_cls_scale
    alpha_pitch = args.pitch_cls_scale
    beta_yaw = args.yaw_reg_scale
    beta_pitch = args.pitch_reg_scale

    idx_tensor_yaw = [idx for idx in range(bin_number_yaw)]
    idx_tensor_yaw = Variable(torch.FloatTensor(idx_tensor_yaw)).to(device)
    idx_tensor_pitch = [idx for idx in range(bin_number_pitch)]
    idx_tensor_pitch = Variable(torch.FloatTensor(idx_tensor_pitch)).to(device)

    softmax = nn.Softmax(dim=1).to(device)


    for epoch in range(n_epochs):
        print(f"#### Epoch {epoch + 1} ####")
        model.train()
        total_yaw_reg = 0
        total_pitch_reg = 0
        total_yaw_bins = 0
        total_pitch_bins = 0 
        total_yaw_combined = 0
        total_pitch_combined = 0

        iters_number = 0

        for i, (images_gaze, labels_gaze, cont_labels_gaze) in enumerate(train_dataloader):
            
            iters_number += 1
            
            images_gaze = Variable(images_gaze).to(device)
            
            label_yaw_cont_gaze = Variable(cont_labels_gaze[:, 0]).to(device)
            label_pitch_cont_gaze = Variable(cont_labels_gaze[:, 1]).to(device)
                

            label_yaw_gaze = labels_gaze[:, 0]
            label_pitch_gaze = labels_gaze[:, 1]

            if args.label_smoothing:
                label_yaw_gaze = label_yaw_gaze.numpy() 
                label_yaw_gaze = smooth_labels(label_yaw_gaze, args.smoothing_sigma, bin_number_yaw)
                label_pitch_gaze = label_pitch_gaze.numpy() 
                label_pitch_gaze = smooth_labels(label_pitch_gaze, args.smoothing_sigma, bin_number_pitch)

            label_yaw_gaze = label_yaw_gaze.to(device)
            label_pitch_gaze = label_pitch_gaze.to(device)
            
            yaw, pitch, yaw_reg, pitch_reg = model(images_gaze)
            
            loss_yaw_gaze = classification_loss(yaw, label_yaw_gaze)
            loss_pitch_gaze = classification_loss(pitch, label_pitch_gaze)

            yaw_reg = yaw_reg.view(-1)
            pitch_reg = pitch_reg.view(-1)

            loss_yaw_reg = regression_loss(label_yaw_cont_gaze, yaw_reg)
            loss_pitch_reg = regression_loss(label_pitch_cont_gaze, pitch_reg)

            loss = alpha_pitch * loss_pitch_gaze + alpha_yaw * loss_yaw_gaze + beta_pitch * loss_pitch_reg + beta_yaw * loss_yaw_reg
            loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                yaw_predicted = softmax(yaw)
                pitch_predicted = softmax(pitch)
                
                yaw_predicted = torch.sum(yaw_predicted * idx_tensor_yaw, 1) * args.yaw_resolution + args.yaw_lower_range
                pitch_predicted = torch.sum(pitch_predicted * idx_tensor_pitch, 1) * args.pitch_resolution + args.pitch_lower_range
                
                loss_yaw_reg_bins = regression_loss(label_yaw_cont_gaze, yaw_predicted)
                loss_pitch_reg_bins = regression_loss(label_pitch_cont_gaze, pitch_predicted)

                loss_yaw_combined = regression_loss((yaw_predicted + yaw_reg)/2, label_yaw_cont_gaze)
                loss_pitch_combined = regression_loss((pitch_predicted + pitch_reg)/2, label_pitch_cont_gaze)
            
            
            total_yaw_reg += regression_loss(label_yaw_cont_gaze, yaw_predicted).detach()
            total_pitch_reg += regression_loss(label_pitch_cont_gaze, pitch_predicted).detach()

            total_yaw_bins += loss_yaw_reg_bins.detach()
            total_pitch_bins += loss_pitch_reg_bins.detach()

            total_yaw_combined += loss_yaw_combined.detach()
            total_pitch_combined += loss_pitch_combined.detach()


        logger.report_scalar("MAE", "Regression Yaw", iteration=epoch, value=(total_yaw_reg/iters_number))
        logger.report_scalar("MAE", "Regression Pitch", iteration=epoch, value=(total_pitch_reg/iters_number))
        logger.report_scalar("MAE", "Bins Yaw", iteration=epoch, value=(total_yaw_bins/iters_number))
        logger.report_scalar("MAE", "Bins Pitch", iteration=epoch, value=(total_pitch_bins/iters_number))
        logger.report_scalar("MAE", "Combined Yaw", iteration=epoch, value=(total_yaw_combined/iters_number))
        logger.report_scalar("MAE", "Combined Pitch", iteration=epoch, value=(total_pitch_combined/iters_number))

        total_yaw_reg = 0
        total_pitch_reg = 0
        total_yaw_bins = 0
        total_pitch_bins = 0 
        total_yaw_combined = 0
        total_pitch_combined = 0

        val_iters = 0
        
        model.eval()
        with torch.no_grad():
            for i, (images_gaze, labels_gaze, cont_labels_gaze) in enumerate(val_dataloader):
                val_iters += 1
                
                label_yaw_cont_gaze = Variable(cont_labels_gaze[:, 0]).to(device)
                label_pitch_cont_gaze = Variable(cont_labels_gaze[:, 1]).to(device)
                
                images_gaze = Variable(images_gaze).to(device)

                yaw, pitch, yaw_reg, pitch_reg = model(images_gaze)
                
                pitch_predicted = softmax(pitch)
                yaw_predicted = softmax(yaw)
                
                yaw_predicted = torch.sum(yaw_predicted * idx_tensor_yaw, 1) * args.yaw_resolution + args.yaw_lower_range
                pitch_predicted = torch.sum(pitch_predicted * idx_tensor_pitch, 1) * args.pitch_resolution + args.pitch_lower_range
                
                loss_pitch_bins = regression_loss(label_pitch_cont_gaze, pitch_predicted)
                loss_yaw_bins = regression_loss(label_yaw_cont_gaze, yaw_predicted)
                
                pitch_reg = pitch_reg.view(-1)
                yaw_reg = yaw_reg.view(-1)

                loss_pitch_reg = regression_loss(label_pitch_cont_gaze, pitch_reg)
                loss_yaw_reg = regression_loss(label_yaw_cont_gaze, yaw_reg)

                loss_pitch_combined = regression_loss(label_pitch_cont_gaze, (pitch_reg + pitch_predicted)/2)
                loss_yaw_combined = regression_loss(label_yaw_cont_gaze, (yaw_reg + yaw_predicted)/2)


                total_pitch_reg += loss_pitch_reg.detach()
                total_yaw_reg += loss_yaw_reg.detach()

                total_pitch_bins += loss_pitch_bins.detach()
                total_yaw_bins += loss_yaw_bins.detach()

                total_pitch_combined += loss_pitch_combined.detach()
                total_yaw_combined += loss_yaw_combined.detach()

        logger.report_scalar("MAE", "[VAL] Regression Yaw", iteration=epoch, value=(total_yaw_reg/val_iters))
        logger.report_scalar("MAE", "[VAL] Regression Pitch", iteration=epoch, value=(total_pitch_reg/val_iters))
        logger.report_scalar("MAE", "[VAL] Bins Yaw", iteration=epoch, value=(total_yaw_bins/val_iters))
        logger.report_scalar("MAE", "[VAL] Bins Pitch", iteration=epoch, value=(total_pitch_bins/val_iters))
        logger.report_scalar("MAE", "[VAL] Combined Yaw", iteration=epoch, value=(total_yaw_combined/val_iters))
        logger.report_scalar("MAE", "[VAL] Combined Pitch", iteration=epoch, value=(total_pitch_combined/val_iters))

        torch.save(model, os.path.join(args.output_folder, f"model_epoch{epoch+1}.pkl"))
