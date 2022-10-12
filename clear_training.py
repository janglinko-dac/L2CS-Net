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




def train(model, optimizer, train_dataset_length, train_dataloader, reg_criterion, cls_criterion, device, idx_tensor, epoch, writer, val_dataloader, writing_frequency=50):
    # TODO: Why the train fucntion uses general 'device' while the eval uses 'gpu' for softmax?
    softmax = nn.Softmax(dim=1).cuda(device)
    model.train()
    sum_loss_pitch_gaze = sum_loss_yaw_gaze = iter_gaze = 0
    for i, (images_gaze, labels_gaze, cont_labels_gaze) in enumerate(train_dataloader):
        # tensor of shape (N, 3, H, W), where N is the batch size (number of normalized face images),
        # 3 is the number of channels, W is the image height, W is the image width
        images_gaze = Variable(images_gaze).cuda(device)

        # Binned labels
        # label_pitch_gaze and label_yaw_gaze are both tensors of shape (N, ), where N is the batch size
        # they hold index of the angle bin in the range from 0 to 27
        label_pitch_gaze = Variable(labels_gaze[:, 0]).cuda(device)
        label_yaw_gaze = Variable(labels_gaze[:, 1]).cuda(device)

        # Continuous labels
        # label_pitch_cont_gaze and label_yaw_cont_gaze are both tensors of shape (N, ), where N is the batch size
        # they hold actual float angle value for a given image in batch
        label_pitch_cont_gaze = Variable(cont_labels_gaze[:, 0]).cuda(device)
        label_yaw_cont_gaze = Variable(cont_labels_gaze[:, 1]).cuda(device)

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

        # TODO: What is this strange math?
        # after the accumulation we end up with pitch_predicted and yaw_predicted tensors of shape (N, )
        pitch_predicted = \
            torch.sum(pitch_predicted * idx_tensor, 1) * 3 - 42
        yaw_predicted = \
            torch.sum(yaw_predicted * idx_tensor, 1) * 3 - 42

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
            [torch.tensor(1.0).cuda(device) for _ in range(len(loss_seq))]

        # TODO: why we set the gradient to None instead of 0?
        optimizer.zero_grad(set_to_none=True)
        # TODO: why this grad_seq is actually never used later on?
        torch.autograd.backward(loss_seq, grad_seq)
        optimizer.step()

        iter_gaze += 1

        # evaluate model on the validation data and log some values
        if (i+1) % writing_frequency == 0:
            writer.add_scalar('Loss/pitch_train', sum_loss_pitch_gaze/iter_gaze, epoch*train_dataset_length//batch_size + i)
            writer.add_scalar('Loss/yaw_train', sum_loss_yaw_gaze/iter_gaze, epoch*train_dataset_length//batch_size + i)
            print('Epoch [%d/%d], Iter [%d/%d] Losses: '
                'Gaze Yaw %.4f,Gaze Pitch %.4f' % (
                    epoch+1,
                    num_epochs,
                    i+1,
                    train_dataset_length//batch_size,
                    sum_loss_pitch_gaze/iter_gaze,
                    sum_loss_yaw_gaze/iter_gaze
                )
                )
            val_loss_pitch, val_loss_yaw, val_pitch_mae, val_yaw_mae = eval(model, val_dataloader, reg_criterion, criterion, gpu, idx_tensor, epoch)
            writer.add_scalar('Val/Loss_pitch', val_loss_pitch, epoch*train_dataset_length//batch_size + i)
            writer.add_scalar('Val/Loss_yaw', val_loss_yaw, epoch*train_dataset_length//batch_size + i)
            writer.add_scalar('Val/MAE_pitch', val_pitch_mae, epoch*train_dataset_length//batch_size + i)
            writer.add_scalar('Val/MAE_yaw', val_yaw_mae, epoch*train_dataset_length//batch_size + i)


def eval(model, val_dataloader, reg_criterion, cls_criterion, gpu, idx_tensor, epoch):
    softmax = nn.Softmax(dim=1).cuda(gpu)
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
                torch.sum(pitch_predicted * idx_tensor, 1) * 3 - 42
            yaw_predicted = \
                torch.sum(yaw_predicted * idx_tensor, 1) * 3 - 42

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

        print(f"Epoch {epoch} validation losses. Yaw: {val_sum_loss_yaw_gaze/val_iter_gaze}  Pitch: {val_sum_loss_pitch_gaze/val_iter_gaze}")
        print(f"Epoch {epoch} validation MAE. Yaw: {val_yaw_mae/val_iter_gaze}  Pitch: {val_pitch_mae/val_iter_gaze}")

        # writer.add_scalar('Val/Loss_epochs_pitch', val_sum_loss_pitch_gaze/val_iter_gaze, epoch)
        # writer.add_scalar('Val/Loss_epochs_yaw', val_sum_loss_yaw_gaze/val_iter_gaze, epoch)
        # writer.add_scalar('Val/MAE_epochs_yaw', val_sum_loss_yaw_gaze/val_iter_gaze, epoch)
        # writer.add_scalar('Val/MAE_epochs_yaw', val_sum_loss_yaw_gaze/val_iter_gaze, epoch)
        return val_sum_loss_pitch_gaze/val_iter_gaze, val_sum_loss_yaw_gaze/val_iter_gaze, val_pitch_mae/val_iter_gaze, val_yaw_mae/val_iter_gaze


if __name__ == '__main__':

    # Parse arguments
    args = parse_args()
    task = clearml.Task.init(project_name="WET", task_name=args.tb)

    # Enable cuda
    cudnn.enabled = True
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    gpu = select_device(args.gpu_id, batch_size=args.batch_size)
    data_set=args.dataset
    # TODO: what is the impact of alpha on accuracy?
    alpha = args.alpha
    output=args.output

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
    model, pre_url = getArch_weights(args.arch, 28)
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
        # TODO: What is this magic number '28'?
        idx_tensor = [idx for idx in range(28)]
        idx_tensor = Variable(torch.FloatTensor(idx_tensor)).cuda(gpu)

        # Optimizer gaze
        optimizer_gaze = torch.optim.Adam([
            {'params': get_ignored_params(model), 'lr': 0},
            {'params': get_non_ignored_params(model), 'lr': args.lr},
            {'params': get_fc_params(model), 'lr': args.lr}
        ], args.lr)

        scheduler = MultiStepLR(optimizer_gaze, milestones=[30, 40], gamma=0.1) # Old parameters for Gaze Capture, steps should be bigger for smaller dataset

        for epoch in range(num_epochs):
            train(
                model,
                optimizer_gaze,
                len(train_dataset),
                train_loader_gaze,
                reg_criterion,
                criterion, gpu,
                idx_tensor,
                epoch,
                writer,
                val_dataloader,
                writing_frequency=50)
            scheduler.step()

            # create the folder to store checkpoints (if not already created)
            if not os.path.isdir(output+args.tb):
                os.makedirs(output+args.tb, exist_ok=True)

            # save the current snapshot and print its contents
            print('Taking snapshot...')
            snapshot_filepath = os.path.join(output, args.tb, f'_epoch_{epoch + 1}.pkl')
            torch.save(model.state_dict(), snapshot_filepath)
            print(f'Snapshot saved under: {snapshot_filepath}')

    else:
        raise ValueError(f'Unsupported dataset type: {data_set}')
