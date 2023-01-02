import argparse
import clearml
import os
import torch
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.autograd import Variable
from hydrant_utils import Hydrant, GazeCaptureDifferentRanges, gauss, smooth_labels
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np

def reset_weights(m):
  '''
    Try resetting model weights to avoid
    weight leakage.
  '''
  for layer in m.children():
   if hasattr(layer, 'reset_parameters'):
    print(f'Reset trainable parameters of layer = {layer}')
    layer.reset_parameters()

train_transformations = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                 std=[0.229, 0.224, 0.225])])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Gaze estimation using Hydrant.')
    # ------DATASET ARGS------
    parser.add_argument('--train-dir', help='Directory path for training dataset.', required=True)
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
    parser.add_argument('--k-fold', help='Number of folds', default=5, type=int)
    # ------OUTPUT ARGUMENTS------
    parser.add_argument('--output-folder', help='Folder where models will be saved', default='')

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

    # Define dataset
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

    # Set fixed random number seed
    torch.manual_seed(42)
    # Define KFold cross validator
    kfold = KFold(n_splits=args.k_fold, shuffle=True)

    regression_yaw = []
    regression_pitch = []
    bins_yaw = []
    bins_pitch = []
    combined_yaw = []
    combined_pitch = []

    regression_yaw_val = []
    regression_pitch_val = []
    bins_yaw_val = []
    bins_pitch_val = []
    combined_yaw_val = []
    combined_pitch_val = []

    # K-fold Cross Validation model evaluation
    for fold, (train_ids, test_ids) in enumerate(kfold.split(train_dataset)):
        print(f'FOLD {fold}')
        print(50*'-')
         # Sample elements randomly from a given list of ids, no replacement.
        train_subsampler = SubsetRandomSampler(train_ids)
        test_subsampler = SubsetRandomSampler(test_ids)

         # Define data loaders for training and testing data in this fold
        train_dataloader = DataLoader(train_dataset,
                                batch_size=args.batch_size,
                                num_workers=1,
                                pin_memory=True,
                                sampler=train_subsampler)
        val_dataloader = DataLoader(train_dataset,
                            batch_size=args.batch_size,
                            num_workers=1,
                            pin_memory=True,
                            sampler= test_subsampler)

        # Build model
        model = Hydrant(args.architecture, bin_number_yaw, bin_number_pitch)

        # TODO: Discuss if the weight should be reset to avoid weight leakage. IMO not necessary
        # cause we use pretrained weights
        # model.apply(reset_weights)
        model.to(device)

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

        # training loop
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


            logger.report_scalar(f"MAE fold no. {fold}", "Regression Yaw", iteration=epoch, value=(total_yaw_reg/iters_number))
            logger.report_scalar(f"MAE fold no. {fold}", "Regression Pitch", iteration=epoch, value=(total_pitch_reg/iters_number))
            logger.report_scalar(f"MAE fold no. {fold}", "Bins Yaw", iteration=epoch, value=(total_yaw_bins/iters_number))
            logger.report_scalar(f"MAE fold no. {fold}", "Bins Pitch", iteration=epoch, value=(total_pitch_bins/iters_number))
            logger.report_scalar(f"MAE fold no. {fold}", "Combined Yaw", iteration=epoch, value=(total_yaw_combined/iters_number))
            logger.report_scalar(f"MAE fold no. {fold}", "Combined Pitch", iteration=epoch, value=(total_pitch_combined/iters_number))

            # save results for the last epoch for each k-fold
            if epoch == n_epochs - 1:
                regression_yaw.append((total_yaw_reg/iters_number).cpu().numpy())
                regression_pitch.append((total_pitch_reg/iters_number).cpu().numpy())
                bins_yaw.append((total_yaw_bins/iters_number).cpu().numpy())
                bins_pitch.append((total_pitch_bins/iters_number).cpu().numpy())
                combined_yaw.append((total_yaw_combined/iters_number).cpu().numpy())
                combined_pitch.append((total_pitch_combined/iters_number).cpu().numpy())

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

            logger.report_scalar(f"MAE fold no. {fold}", "[VAL] Regression Yaw", iteration=epoch, value=(total_yaw_reg/val_iters))
            logger.report_scalar(f"MAE fold no. {fold}", "[VAL] Regression Pitch", iteration=epoch, value=(total_pitch_reg/val_iters))
            logger.report_scalar(f"MAE fold no. {fold}", "[VAL] Bins Yaw", iteration=epoch, value=(total_yaw_bins/val_iters))
            logger.report_scalar(f"MAE fold no. {fold}", "[VAL] Bins Pitch", iteration=epoch, value=(total_pitch_bins/val_iters))
            logger.report_scalar(f"MAE fold no. {fold}", "[VAL] Combined Yaw", iteration=epoch, value=(total_yaw_combined/val_iters))
            logger.report_scalar(f"MAE fold no. {fold}", "[VAL] Combined Pitch", iteration=epoch, value=(total_pitch_combined/val_iters))

            if epoch == n_epochs - 1:
                regression_yaw_val.append((total_yaw_reg/iters_number).cpu().numpy())
                regression_pitch_val.append((total_pitch_reg/iters_number).cpu().numpy())
                bins_yaw_val.append((total_yaw_bins/iters_number).cpu().numpy())
                bins_pitch_val.append((total_pitch_bins/iters_number).cpu().numpy())
                combined_yaw_val.append((total_yaw_combined/iters_number).cpu().numpy())
                combined_pitch_val.append((total_pitch_combined/iters_number).cpu().numpy())

        if args.output_folder:
            # create output directory
            os.makedirs(args.output_folder, exist_ok=True)
            torch.save(model, os.path.join(args.output_folder, f"model_fold_{fold}_epoch{epoch+1}.pkl"))

    # calculate metrics for k-fold

    df = pd.DataFrame(
    {
        "mean": [np.mean(regression_yaw), np.mean(regression_pitch), np.mean(bins_yaw), np.mean(bins_pitch), np.mean(combined_yaw), np.mean(combined_pitch)],
        "std": [np.std(regression_yaw), np.std(regression_pitch), np.std(bins_yaw), np.std(bins_pitch), np.std(combined_yaw), np.std(combined_pitch)],

    },
    index=["regression_yaw", "regression_pitch", "bins_yaw","bins_pitch", "combined_yaw", "combined_pitch"],
    )
    df.index.name = "loss"
    logger.report_table(
        "K-fold training for last epoch",
        "K-fold training",
        iteration=0,
        table_plot=df
    )
    df = pd.DataFrame(
    {
        "mean": [np.mean(regression_yaw_val), np.mean(regression_pitch_val), np.mean(bins_yaw_val), np.mean(bins_pitch_val), np.mean(combined_yaw_val), np.mean(combined_pitch_val)],
        "std": [np.std(regression_yaw_val), np.std(regression_pitch_val), np.std(bins_yaw_val), np.std(bins_pitch_val), np.std(combined_yaw_val), np.std(combined_pitch_val)],

    },
    index=["regression_yaw", "regression_pitch", "bins_yaw","bins_pitch", "combined_yaw", "combined_pitch"],
    )
    df.index.name = "loss"
    logger.report_table(
        "K-fold validation for last epoch",
        "K-fold validation",
        iteration=0,
        table_plot=df
)