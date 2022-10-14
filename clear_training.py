import clearml
import os

import torch.utils.model_zoo as model_zoo
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.backends.cudnn as cudnn

from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import MultiStepLR


# Custom Packages
import datasets
from utils import select_device
from clear_training_utils import (parse_args,
                                  get_fc_params,
                                  get_non_ignored_params,
                                  get_ignored_params,
                                  getArch_weights,
                                  load_filtered_state_dict)


BIN_COUNT = 28
ANGLE_RANGE = 42  # +/- angle range in degrees
DEGREES_PER_BIN = 2 * ANGLE_RANGE // BIN_COUNT


def train(model, optimizer, train_dataset_length, train_dataloader, reg_criterion, cls_criterion, gpu, idx_tensor, epoch, writer, val_dataloader, writing_frequency=50):
    model.train()
    sum_loss_pitch_gaze = sum_loss_yaw_gaze = iter_gaze = 0
    for i, (images_gaze, labels_gaze, cont_labels_gaze) in enumerate(train_dataloader):
        # tensor of shape (N, 3, H, W), where N is the batch size (number of normalized face images),
        # 3 is the number of channels, W is the image height, W is the image width
        images_gaze = Variable(images_gaze).cuda(gpu)

        # Binned labels
        # label_pitch_gaze and label_yaw_gaze are both tensors of shape (N, ), where N is the batch size
        # they hold index of the angle bin in the range from 0 to 27
        label_pitch_gaze = Variable(labels_gaze[:, 0]).cuda(gpu)
        label_yaw_gaze = Variable(labels_gaze[:, 1]).cuda(gpu)

        # Continuous labels
        # label_pitch_cont_gaze and label_yaw_cont_gaze are both tensors of shape (N, ), where N is the batch size
        # they hold actual float angle value for a given image in batch
        label_pitch_cont_gaze = Variable(cont_labels_gaze[:, 0]).cuda(gpu)
        label_yaw_cont_gaze = Variable(cont_labels_gaze[:, 1]).cuda(gpu)

        # pitch and yaw is a (N, 28) tensor and it holds probabilities of the yaw and pitch angles
        # to be within one of the 28 bins
        pitch, yaw = model(images_gaze)

        # Cross entropy loss
        # loss_pitch_gaze and loss_yaw_gaze are scalar values of cross-entropy loss
        # for all N images in the batch
        loss_pitch_gaze = cls_criterion(pitch, label_pitch_gaze)
        loss_yaw_gaze = cls_criterion(yaw, label_yaw_gaze)

        # MSE loss
        # pitch_predicted and yaw_predicted are tensors of shape (N, 28) holding normalized probabilities
        pitch_predicted = softmax(pitch)
        yaw_predicted = softmax(yaw)

        # after the accumulation we end up with pitch_predicted and yaw_predicted tensors of shape (N, )
        # convert from bin index to actual float angle
        # there are 28 bins that cover 84 degrees (-42, 42), so each bin covers 3 degrees horizontally and vertically
        pitch_predicted = \
            torch.sum(pitch_predicted * idx_tensor, 1) * DEGREES_PER_BIN - ANGLE_RANGE
        yaw_predicted = \
            torch.sum(yaw_predicted * idx_tensor, 1) * DEGREES_PER_BIN - ANGLE_RANGE

        # loss_reg_pitch and loss_reg_yaw will hold a single scalar value
        loss_reg_pitch = reg_criterion(
            pitch_predicted, label_pitch_cont_gaze)
        loss_reg_yaw = reg_criterion(
            yaw_predicted, label_yaw_cont_gaze)

        # Total loss
        loss_pitch_gaze += alpha * loss_reg_pitch
        loss_yaw_gaze += alpha * loss_reg_yaw

        sum_loss_pitch_gaze += loss_pitch_gaze.detach()
        sum_loss_yaw_gaze += loss_yaw_gaze.detach()

        # perform back-propagation
        loss_seq = [loss_pitch_gaze, loss_yaw_gaze]
        # TODO: Is this grad_seq really needed to be se to all 1.0? Can't we just set it to None?
        grad_seq = \
            [torch.tensor(1.0).cuda(gpu) for _ in range(len(loss_seq))]

        # when we set the gradient to None instead of 0?
        optimizer.zero_grad(set_to_none=True)
        torch.autograd.backward(loss_seq, grad_seq)
        optimizer.step()

        iter_gaze += 1

        # evaluate model on the validation data and log some values
        if (i+1) % writing_frequency == 0:
            writer.add_scalar('Train_Loss/pitch', sum_loss_pitch_gaze/iter_gaze, epoch*train_dataset_length//batch_size + i)
            writer.add_scalar('Train_Loss/yaw', sum_loss_yaw_gaze/iter_gaze, epoch*train_dataset_length//batch_size + i)

            val_loss_pitch, val_loss_yaw, val_pitch_mae, val_yaw_mae = eval(model, val_dataloader, reg_criterion, criterion, gpu, idx_tensor, epoch)
            writer.add_scalar('Val_Loss/pitch', val_loss_pitch, epoch*train_dataset_length//batch_size + i)
            writer.add_scalar('Val_Loss/yaw', val_loss_yaw, epoch*train_dataset_length//batch_size + i)
            writer.add_scalar('Val_MAE/pitch', val_pitch_mae, epoch*train_dataset_length//batch_size + i)
            writer.add_scalar('Val_MAE/yaw', val_yaw_mae, epoch*train_dataset_length//batch_size + i)

    print(f"Epoch {epoch + 1}/{num_epochs} train losses. Yaw: {sum_loss_yaw_gaze/iter_gaze}, Pitch: {sum_loss_pitch_gaze/iter_gaze}")

    # when a single epoch is complete, evaluate the trained model on validation dataset
    val_loss_pitch, val_loss_yaw, val_pitch_mae, val_yaw_mae = eval(model, val_dataloader, reg_criterion, criterion, gpu, idx_tensor, epoch)

    print(f"Epoch {epoch + 1}/{num_epochs} validation losses. Yaw: {val_loss_yaw}, Pitch: {val_loss_pitch}")
    print(f"Epoch {epoch + 1}/{num_epochs} validation MAE. Yaw: {val_yaw_mae}, Pitch: {val_pitch_mae}")

    return val_loss_pitch.item(), val_loss_yaw.item(), val_pitch_mae.item(), val_yaw_mae.item()


def eval(model, val_dataloader, reg_criterion, cls_criterion, gpu, idx_tensor, epoch):
    model.eval()
    val_sum_loss_pitch_gaze = val_sum_loss_yaw_gaze = val_iter_gaze = val_yaw_mae = val_pitch_mae = 0
    with torch.no_grad():
        for k, (images_gaze_val, labels_gaze_val, cont_labels_gaze_val) in enumerate(val_dataloader):
            images_gaze_val = Variable(images_gaze_val).cuda(gpu)
            label_pitch_gaze = Variable(labels_gaze_val[:, 0]).cuda(gpu)
            label_yaw_gaze = Variable(labels_gaze_val[:, 1]).cuda(gpu)
            # Continuous labels
            label_pitch_cont_gaze = Variable(cont_labels_gaze_val[:, 0]).cuda(gpu)
            label_yaw_cont_gaze = Variable(cont_labels_gaze_val[:, 1]).cuda(gpu)

            pitch, yaw = model(images_gaze_val)

            # Cross entropy loss
            loss_pitch_gaze = cls_criterion(pitch, label_pitch_gaze)
            loss_yaw_gaze = cls_criterion(yaw, label_yaw_gaze)
            # MAE loss
            pitch_predicted = softmax(pitch)
            yaw_predicted = softmax(yaw)

            pitch_predicted = \
                torch.sum(pitch_predicted * idx_tensor, 1) * DEGREES_PER_BIN - ANGLE_RANGE
            yaw_predicted = \
                torch.sum(yaw_predicted * idx_tensor, 1) * DEGREES_PER_BIN - ANGLE_RANGE

            # Add MAE metrics
            val_yaw_mae += abs(label_yaw_cont_gaze.detach() - yaw_predicted.detach())
            val_pitch_mae += abs(label_pitch_cont_gaze.detach() - pitch_predicted.detach())

            loss_reg_pitch = reg_criterion(
                pitch_predicted, label_pitch_cont_gaze)
            loss_reg_yaw = reg_criterion(
                yaw_predicted, label_yaw_cont_gaze)

            # Total loss
            loss_pitch_gaze += alpha * loss_reg_pitch.detach()
            loss_yaw_gaze += alpha * loss_reg_yaw.detach()

            val_sum_loss_pitch_gaze += loss_pitch_gaze.detach()
            val_sum_loss_yaw_gaze += loss_yaw_gaze.detach()

            val_iter_gaze += 1

        return val_sum_loss_pitch_gaze/val_iter_gaze, val_sum_loss_yaw_gaze/val_iter_gaze, val_pitch_mae/val_iter_gaze, val_yaw_mae/val_iter_gaze


if __name__ == '__main__':

    # Parse arguments
    args = parse_args()
    # TODO: Add tags as input arguments
    task = clearml.Task.init(project_name="WET", task_name=args.tb, tags=[])

    # Enable cuda
    cudnn.enabled = True
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    gpu = select_device(args.gpu_id, batch_size=args.batch_size)
    data_set=args.dataset
    # TODO: what is the impact of alpha on accuracy?
    alpha = args.alpha
    output=args.output

    # define softmax
    softmax = nn.Softmax(dim=1).cuda(gpu)

    # Define transforms
    transformations = transforms.Compose([
        transforms.Resize(448),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225] # RGB
        )
    ])

    # Create Tensorboard writer
    writer = SummaryWriter(f"runs/{args.tb}")
    writer.add_hparams({"architecture": args.arch, "lr": args.lr, "batch_size": args.batch_size,
                        "epochs": args.num_epochs, "alpha": args.alpha, "experiment_name": args.tb}, {'hparams/accuracy': 0})

    # instantiate model based on pre-trained weights
    model, pre_url = getArch_weights(args.arch, BIN_COUNT)
    load_filtered_state_dict(model, model_zoo.load_url(pre_url))
    model = nn.DataParallel(model)
    model.to(gpu)

    if data_set == "gazecapture":

        train_dataset=datasets.GazeCapture(args.gazecapture_ann,
                                           args.gazecapture_dir,
                                           transformations)

        val_dataset = datasets.GazeCapture(args.validation_ann,
                                           args.validation_dir,
                                           transformations)

        train_loader_gaze = DataLoader(dataset=train_dataset,
                                       batch_size=int(batch_size),
                                       shuffle=True,
                                       num_workers=1,
                                       pin_memory=True)

        val_dataloader = DataLoader(dataset=val_dataset,
                                    batch_size=1,
                                    shuffle=False,
                                    num_workers=1,
                                    pin_memory=True)

        torch.backends.cudnn.benchmark = True

        criterion = nn.CrossEntropyLoss().cuda(gpu)
        reg_criterion = nn.MSELoss().cuda(gpu)
        # softmax = nn.Softmax(dim=1).cuda(gpu)
        idx_tensor = [idx for idx in range(BIN_COUNT)]
        idx_tensor = Variable(torch.FloatTensor(idx_tensor)).cuda(gpu)

        # Optimizer gaze
        optimizer_gaze = torch.optim.Adam([
            {'params': get_ignored_params(model), 'lr': 0},
            {'params': get_non_ignored_params(model), 'lr': args.lr},
            {'params': get_fc_params(model), 'lr': args.lr}
        ], args.lr)

        scheduler = MultiStepLR(optimizer_gaze, milestones=[30, 40], gamma=0.1) # Old parameters for Gaze Capture, steps should be bigger for smaller dataset

        for epoch in range(num_epochs):

            is_best_model = False

            val_loss_pitch, \
            val_loss_yaw, \
            val_pitch_mae, \
            val_yaw_mae = train(
                model,
                optimizer_gaze,
                len(train_dataset),
                train_loader_gaze,
                reg_criterion,
                criterion,
                gpu,
                idx_tensor,
                epoch,
                writer,
                val_dataloader,
                writing_frequency=50)
            scheduler.step()

            if epoch == 0:
                min_val_pitch_yaw_mae = (val_pitch_mae + val_yaw_mae) / 2
                is_best_model = True
            else:
                val_loss_pitch_yaw = (val_loss_pitch + val_loss_yaw) / 2
                val_pitch_yaw_mae = (val_pitch_mae + val_yaw_mae) / 2

                # give MAE a priority when selecting best model
                if val_pitch_mae < min_val_pitch_yaw_mae:
                    print(f'Found new best model with pitch & yaw MAE: {val_pitch_yaw_mae}')
                    min_val_pitch_yaw_mae = val_pitch_mae
                    is_best_model = True

            # create the folder to store checkpoints (if not already created)
            if not os.path.isdir(output+args.tb):
                os.makedirs(output+args.tb, exist_ok=True)

            # save the current snapshot and print its contents
            print('Taking snapshot...')
            snapshot_filepath = os.path.join(output, args.tb, f'_epoch_{epoch + 1}.pkl')
            torch.save(model.state_dict(), snapshot_filepath)
            print(f'Snapshot saved under: {snapshot_filepath}')

            if is_best_model:
                print(f'Best model found @ epoch {epoch + 1}')
                torch.save(model.state_dict(), os.path.join(output, args.tb, f'best.pkl'))

    else:
        raise ValueError(f'Unsupported dataset type: {data_set}')
