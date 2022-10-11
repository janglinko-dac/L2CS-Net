import os
import argparse
import time

import torch.utils.model_zoo as model_zoo
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchvision

from torch.utils.tensorboard import SummaryWriter

import datasets
from model import L2CS
from utils import select_device

import clearml

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Gaze estimation using L2CSNet.')
    # Gaze360
    parser.add_argument(
        '--gaze360image_dir', dest='gaze360image_dir', help='Directory path for gaze images.',
        default='datasets/Gaze360/Image', type=str)
    parser.add_argument(
        '--gaze360label_dir', dest='gaze360label_dir', help='Directory path for gaze labels.',
        default='datasets/Gaze360/Label/train.label', type=str)
    # mpiigaze
    parser.add_argument(
        '--gazeMpiimage_dir', dest='gazeMpiimage_dir', help='Directory path for gaze images.',
        default='datasets/MPIIFaceGaze/Image', type=str)
    parser.add_argument(
        '--gazeMpiilabel_dir', dest='gazeMpiilabel_dir', help='Directory path for gaze labels.',
        default='datasets/MPIIFaceGaze/Label', type=str)
    # GazeCapture
    parser.add_argument(
        '--gazecapture-dir', help='Root path of dataset.', type=str
    )
    parser.add_argument(
        '--gazecapture-ann', help='Annotations filepath.', type=str
    )
    # Validation Dataset
    parser.add_argument(
        '--validation-dir', help='Root path of the dataset.', type=str
    )
    parser.add_argument(
        '--validation-ann', help='Annotations filepath.', type=str
    )

    # Important args -------------------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------------
    parser.add_argument(
        '--dataset', dest='dataset', help='mpiigaze, rtgene, gaze360, ethgaze, gazecapture',
        default= "gaze360", type=str)
    parser.add_argument(
        '--output', dest='output', help='Path of output models.',
        default='output/snapshots/', type=str)
    parser.add_argument(
        '--snapshot', dest='snapshot', help='Path of model snapshot.',
        default='', type=str)
    parser.add_argument(
        '--gpu', dest='gpu_id', help='GPU device id to use [0] or multiple 0,1,2,3',
        default='0', type=str)
    parser.add_argument(
        '--num_epochs', dest='num_epochs', help='Maximum number of training epochs.',
        default=60, type=int)
    parser.add_argument(
        '--batch_size', dest='batch_size', help='Batch size.',
        default=1, type=int)
    parser.add_argument(
        '--arch', dest='arch', help='Network architecture, can be: ResNet18, ResNet34, [ResNet50], ''ResNet101, ResNet152, Squeezenet_1_0, Squeezenet_1_1, MobileNetV2',
        default='ResNet50', type=str)
    parser.add_argument(
        '--alpha', dest='alpha', help='Regression loss coefficient.',
        default=1, type=float)
    parser.add_argument(
        '--lr', dest='lr', help='Base learning rate.',
        default=0.00001, type=float)
    # ---------------------------------------------------------------------------------------------------------------------
    # Important args ------------------------------------------------------------------------------------------------------
    # Tensorboard args
    parser.add_argument(
        '--tb', help='name of the output folder that will be containing data about an experiment.', required=True)
    args = parser.parse_args()
    return args

def get_ignored_params(model):
    # Generator function that yields ignored params.
    b = [model.module.conv1, model.module.bn1, model.module.fc_finetune]
    for i in range(len(b)):
        for module_name, module in b[i].named_modules():
            if 'bn' in module_name:
                module.eval()
            for name, param in module.named_parameters():
                yield param

def get_non_ignored_params(model):
    # Generator function that yields params that will be optimized.
    b = [model.module.layer1, model.module.layer2, model.module.layer3, model.module.layer4]
    for i in range(len(b)):
        for module_name, module in b[i].named_modules():
            if 'bn' in module_name:
                module.eval()
            for name, param in module.named_parameters():
                yield param

def get_fc_params(model):
    # Generator function that yields fc layer params.
    b = [model.module.fc_yaw_gaze, model.module.fc_pitch_gaze]
    for i in range(len(b)):
        for module_name, module in b[i].named_modules():
            for name, param in module.named_parameters():
                yield param

def load_filtered_state_dict(model, snapshot):
    # By user apaszke from discuss.pytorch.org
    model_dict = model.state_dict()
    snapshot = {k: v for k, v in snapshot.items() if k in model_dict}
    model_dict.update(snapshot)
    model.load_state_dict(model_dict)


def getArch_weights(arch, bins):
    if arch == 'ResNet18':
        model = L2CS(torchvision.models.resnet.BasicBlock, [2, 2, 2, 2], bins)
        pre_url = 'https://download.pytorch.org/models/resnet18-5c106cde.pth'
    elif arch == 'ResNet34':
        model = L2CS(torchvision.models.resnet.BasicBlock, [3, 4, 6, 3], bins)
        pre_url = 'https://download.pytorch.org/models/resnet34-333f7ec4.pth'
    elif arch == 'ResNet101':
        model = L2CS(torchvision.models.resnet.Bottleneck, [3, 4, 23, 3], bins)
        pre_url = 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth'
    elif arch == 'ResNet152':
        model = L2CS(torchvision.models.resnet.Bottleneck,[3, 8, 36, 3], bins)
        pre_url = 'https://download.pytorch.org/models/resnet152-b121ed2d.pth'
    else:
        if arch != 'ResNet50':
            print('Invalid value for architecture is passed! '
                  'The default value of ResNet50 will be used instead!')
        model = L2CS(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], bins)
        pre_url = 'https://download.pytorch.org/models/resnet50-19c8e357.pth'

    return model, pre_url

if __name__ == '__main__':
    # task = clearml.Task.init(project_name="WET", task_name="Task02")
    # task.execute_remotely(queue_name='initial')

    args = parse_args()
    cudnn.enabled = True
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    gpu = select_device(args.gpu_id, batch_size=args.batch_size)
    data_set=args.dataset
    alpha = args.alpha
    output=args.output


    transformations = transforms.Compose([
        transforms.Resize(448),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225] # RGB
        )
    ])

    writer = SummaryWriter(f"runs/{args.tb}")
    
    if data_set=="gaze360":
        model, pre_url = getArch_weights(args.arch, 90)
        if args.snapshot == '':
            load_filtered_state_dict(model, model_zoo.load_url(pre_url))
        else:
            saved_state_dict = torch.load(args.snapshot)
            model.load_state_dict(saved_state_dict)


        model.cuda(gpu)
        dataset=datasets.Gaze360(args.gaze360label_dir, args.gaze360image_dir, transformations, 180, 4)
        print('Loading data.')
        train_loader_gaze = DataLoader(
            dataset=dataset,
            batch_size=int(batch_size),
            shuffle=True,
            num_workers=0,
            pin_memory=True)
        torch.backends.cudnn.benchmark = True

        summary_name = '{}_{}'.format('L2CS-gaze360-', int(time.time()))
        output=os.path.join(output, summary_name)
        if not os.path.exists(output):
            os.makedirs(output)


        criterion = nn.CrossEntropyLoss().cuda(gpu)
        reg_criterion = nn.MSELoss().cuda(gpu)
        softmax = nn.Softmax(dim=1).cuda(gpu)
        idx_tensor = [idx for idx in range(90)]
        idx_tensor = Variable(torch.FloatTensor(idx_tensor)).cuda(gpu)


        # Optimizer gaze
        optimizer_gaze = torch.optim.Adam([
            {'params': get_ignored_params(model), 'lr': 0},
            {'params': get_non_ignored_params(model), 'lr': args.lr},
            {'params': get_fc_params(model), 'lr': args.lr}
        ], args.lr)


        configuration = f"\ntrain configuration, gpu_id={args.gpu_id}, batch_size={batch_size}, model_arch={args.arch}\nStart testing dataset={data_set}, loader={len(train_loader_gaze)}------------------------- \n"
        print(configuration)
        
        # Build tensorboard
        

        for epoch in range(num_epochs):
            sum_loss_pitch_gaze = sum_loss_yaw_gaze = iter_gaze = 0

            for i, (images_gaze, labels_gaze, cont_labels_gaze,name) in enumerate(train_loader_gaze):
                images_gaze = Variable(images_gaze).cuda(gpu)

                # Binned labels
                label_pitch_gaze = Variable(labels_gaze[:, 0]).cuda(gpu)
                label_yaw_gaze = Variable(labels_gaze[:, 1]).cuda(gpu)

                # Continuous labels
                label_pitch_cont_gaze = Variable(cont_labels_gaze[:, 0]).cuda(gpu)
                label_yaw_cont_gaze = Variable(cont_labels_gaze[:, 1]).cuda(gpu)

                pitch, yaw = model(images_gaze)

                # Cross entropy loss
                loss_pitch_gaze = criterion(pitch, label_pitch_gaze)
                loss_yaw_gaze = criterion(yaw, label_yaw_gaze)

                # MSE loss
                pitch_predicted = softmax(pitch)
                yaw_predicted = softmax(yaw)

                pitch_predicted = \
                    torch.sum(pitch_predicted * idx_tensor, 1) * 4 - 180
                yaw_predicted = \
                    torch.sum(yaw_predicted * idx_tensor, 1) * 4 - 180

                loss_reg_pitch = reg_criterion(
                    pitch_predicted, label_pitch_cont_gaze)
                loss_reg_yaw = reg_criterion(
                    yaw_predicted, label_yaw_cont_gaze)

                # Total loss
                loss_pitch_gaze += alpha * loss_reg_pitch
                loss_yaw_gaze += alpha * loss_reg_yaw

                sum_loss_pitch_gaze += loss_pitch_gaze
                sum_loss_yaw_gaze += loss_yaw_gaze

                loss_seq = [loss_pitch_gaze, loss_yaw_gaze]
                grad_seq = [torch.tensor(1.0).cuda(gpu) for _ in range(len(loss_seq))]
                optimizer_gaze.zero_grad(set_to_none=True)
                torch.autograd.backward(loss_seq, grad_seq)
                optimizer_gaze.step()
                # scheduler.step()

                iter_gaze += 1

                if (i+1) % 100 == 0:
                    writer.add_scalar('Loss/pitch_train', sum_loss_pitch_gaze/iter_gaze, epoch*len(dataset)//batch_size + i)
                    writer.add_scalar('Loss/yaw_train', sum_loss_yaw_gaze/iter_gaze, epoch*len(dataset)//batch_size + i)

                    print('Epoch [%d/%d], Iter [%d/%d] Losses: '
                        'Gaze Yaw %.4f,Gaze Pitch %.4f' % (
                            epoch+1,
                            num_epochs,
                            i+1,
                            len(dataset)//batch_size,
                            sum_loss_pitch_gaze/iter_gaze,
                            sum_loss_yaw_gaze/iter_gaze
                        )
                        ) 


            if epoch % 1 == 0 and epoch < num_epochs:
                writer.add_scalar('Loss/epochs_pitch_train', sum_loss_pitch_gaze/iter_gaze, epoch)
                writer.add_scalar('Loss/epochs_yaw_train', sum_loss_yaw_gaze/iter_gaze, epoch)

                print('Taking snapshot...',
                    torch.save(model.state_dict(),
                                output +'/'+
                                '_epoch_' + str(epoch+1) + '.pkl')
                    )



    elif data_set=="mpiigaze":
        writer.add_hparams({"architecture": args.arch, "lr": args.lr, "batch_size": args.batch_size,
                        "dataset": args.gazeMpiimage_dir, "experiment_name": args.tb}, {'hparams/accuracy': 0})

        folder = os.listdir(args.gazeMpiilabel_dir)
        folder.sort()
        testlabelpathombined = [os.path.join(args.gazeMpiilabel_dir, j) for j in folder]
        for fold in range(1):
            model, pre_url = getArch_weights(args.arch, 28)
            load_filtered_state_dict(model, model_zoo.load_url(pre_url))
            model = nn.DataParallel(model)
            model.to(gpu)
            print('Loading data.')
            dataset=datasets.Mpiigaze(testlabelpathombined,args.gazeMpiimage_dir, transformations, True, 42, fold)
            train_loader_gaze = DataLoader(
                dataset=dataset,
                batch_size=int(batch_size),
                shuffle=True,
                num_workers=4,
                pin_memory=True)
            torch.backends.cudnn.benchmark = True

            summary_name = '{}_{}'.format('L2CS-mpiigaze', int(time.time()))


            if not os.path.exists(os.path.join(output+'/{}'.format(summary_name),'fold' + str(fold))):
                os.makedirs(os.path.join(output+'/{}'.format(summary_name),'fold' + str(fold)))


            criterion = nn.CrossEntropyLoss().cuda(gpu)
            reg_criterion = nn.MSELoss().cuda(gpu)
            softmax = nn.Softmax(dim=1).cuda(gpu)
            idx_tensor = [idx for idx in range(28)]
            idx_tensor = Variable(torch.FloatTensor(idx_tensor)).cuda(gpu)

            # Optimizer gaze
            optimizer_gaze = torch.optim.Adam([
                {'params': get_ignored_params(model), 'lr': 0},
                {'params': get_non_ignored_params(model), 'lr': args.lr},
                {'params': get_fc_params(model), 'lr': args.lr}
            ], args.lr)



            configuration = f"\ntrain configuration, gpu_id={args.gpu_id}, batch_size={batch_size}, model_arch={args.arch}\n Start training dataset={data_set}, loader={len(train_loader_gaze)}, fold={fold}--------------\n"
            print(configuration)
            for epoch in range(num_epochs):
                sum_loss_pitch_gaze = sum_loss_yaw_gaze = iter_gaze = 0


                for i, (images_gaze, labels_gaze, cont_labels_gaze,name) in enumerate(train_loader_gaze):
                    images_gaze = Variable(images_gaze).cuda(gpu)

                    # Binned labels
                    label_pitch_gaze = Variable(labels_gaze[:, 0]).cuda(gpu)
                    label_yaw_gaze = Variable(labels_gaze[:, 1]).cuda(gpu)

                    # Continuous labels
                    label_pitch_cont_gaze = Variable(cont_labels_gaze[:, 0]).cuda(gpu)
                    label_yaw_cont_gaze = Variable(cont_labels_gaze[:, 1]).cuda(gpu)

                    pitch, yaw = model(images_gaze)

                    # Cross entropy loss
                    loss_pitch_gaze = criterion(pitch, label_pitch_gaze)
                    loss_yaw_gaze = criterion(yaw, label_yaw_gaze)

                    # MSE loss
                    pitch_predicted = softmax(pitch)
                    yaw_predicted = softmax(yaw)

                    pitch_predicted = \
                        torch.sum(pitch_predicted * idx_tensor, 1) * 3 - 42
                    yaw_predicted = \
                        torch.sum(yaw_predicted * idx_tensor, 1) * 3 - 42

                    loss_reg_pitch = reg_criterion(
                        pitch_predicted, label_pitch_cont_gaze)
                    loss_reg_yaw = reg_criterion(
                        yaw_predicted, label_yaw_cont_gaze)

                    # Total loss
                    loss_pitch_gaze += alpha * loss_reg_pitch
                    loss_yaw_gaze += alpha * loss_reg_yaw

                    sum_loss_pitch_gaze += loss_pitch_gaze
                    sum_loss_yaw_gaze += loss_yaw_gaze

                    loss_seq = [loss_pitch_gaze, loss_yaw_gaze]
                    grad_seq = \
                        [torch.tensor(1.0).cuda(gpu) for _ in range(len(loss_seq))]

                    optimizer_gaze.zero_grad(set_to_none=True)
                    torch.autograd.backward(loss_seq, grad_seq)
                    optimizer_gaze.step()

                    iter_gaze += 1

                    if (i+1) % 100 == 0:
                        writer.add_scalar('Loss/pitch_train', sum_loss_pitch_gaze/iter_gaze, epoch*len(dataset)//batch_size + i)
                        writer.add_scalar('Loss/yaw_train', sum_loss_yaw_gaze/iter_gaze, epoch*len(dataset)//batch_size + i)
                        print('Epoch [%d/%d], Iter [%d/%d] Losses: '
                            'Gaze Yaw %.4f,Gaze Pitch %.4f' % (
                                epoch+1,
                                num_epochs,
                                i+1,
                                len(dataset)//batch_size,
                                sum_loss_pitch_gaze/iter_gaze,
                                sum_loss_yaw_gaze/iter_gaze
                            )
                            )



                # Save models at numbered epochs.
                if epoch % 1 == 0 and epoch < num_epochs:
                    writer.add_scalar('Loss/epochs_pitch_train', sum_loss_pitch_gaze/iter_gaze, epoch)
                    writer.add_scalar('Loss/epochs_yaw_train', sum_loss_yaw_gaze/iter_gaze, epoch)
                    if not os.path.isdir(output+'fold' + str(fold)):
                        os.mkdir(output+'fold' + str(fold))
                    print('Taking snapshot...',
                        torch.save(model.state_dict(),
                                    output+'fold' + str(fold) +'/'+
                                    '_epoch_' + str(epoch+1) + '.pkl')
                        )

    elif data_set=="gazecapture":
        model, pre_url = getArch_weights(args.arch, 28)
        load_filtered_state_dict(model, model_zoo.load_url(pre_url))
        model = nn.DataParallel(model)
        model.to(gpu)
        print('Loading data.')
        dataset=datasets.GazeCapture(args.gazecapture_ann, args.gazecapture_dir, transformations)
        train_loader_gaze = DataLoader(
            dataset=dataset,
            batch_size=int(batch_size),
            shuffle=True,
            num_workers=4,
            pin_memory=True)
        val_dataset = datasets.GazeCapture(args.validation_ann,
                                           args.validation_dir,
                                           transformations)

        val_dataloader = DataLoader(
            dataset=val_dataset,
            batch_size=2,
            shuffle=True,
            num_workers=4,
            pin_memory=True)

        torch.backends.cudnn.benchmark = True

        criterion = nn.CrossEntropyLoss().cuda(gpu)
        reg_criterion = nn.MSELoss().cuda(gpu)
        softmax = nn.Softmax(dim=1).cuda(gpu)
        idx_tensor = [idx for idx in range(28)]
        idx_tensor = Variable(torch.FloatTensor(idx_tensor)).cuda(gpu)

        # Optimizer gaze
        optimizer_gaze = torch.optim.Adam([
            {'params': get_ignored_params(model), 'lr': 0},
            {'params': get_non_ignored_params(model), 'lr': args.lr},
            {'params': get_fc_params(model), 'lr': args.lr}
        ], args.lr)



        configuration = f"\ntrain configuration, gpu_id={args.gpu_id}, batch_size={batch_size}, model_arch={args.arch}\n Start training dataset={data_set}, loader={len(train_loader_gaze)}--------------\n"
        print(configuration)
        for epoch in range(num_epochs):
            sum_loss_pitch_gaze = sum_loss_yaw_gaze = iter_gaze = 0

            for i, (images_gaze, labels_gaze, cont_labels_gaze) in enumerate(train_loader_gaze):
                images_gaze = Variable(images_gaze).cuda(gpu)

                # Binned labels
                label_pitch_gaze = Variable(labels_gaze[:, 0]).cuda(gpu)
                label_yaw_gaze = Variable(labels_gaze[:, 1]).cuda(gpu)

                # Continuous labels
                label_pitch_cont_gaze = Variable(cont_labels_gaze[:, 0]).cuda(gpu)
                label_yaw_cont_gaze = Variable(cont_labels_gaze[:, 1]).cuda(gpu)

                pitch, yaw = model(images_gaze)

                # Cross entropy loss
                loss_pitch_gaze = criterion(pitch, label_pitch_gaze)
                loss_yaw_gaze = criterion(yaw, label_yaw_gaze)

                # MSE loss
                pitch_predicted = softmax(pitch)
                yaw_predicted = softmax(yaw)

                pitch_predicted = \
                    torch.sum(pitch_predicted * idx_tensor, 1) * 3 - 42
                yaw_predicted = \
                    torch.sum(yaw_predicted * idx_tensor, 1) * 3 - 42

                loss_reg_pitch = reg_criterion(
                    pitch_predicted, label_pitch_cont_gaze)
                loss_reg_yaw = reg_criterion(
                    yaw_predicted, label_yaw_cont_gaze)

                # Total loss
                loss_pitch_gaze += alpha * loss_reg_pitch
                loss_yaw_gaze += alpha * loss_reg_yaw

                sum_loss_pitch_gaze += loss_pitch_gaze
                sum_loss_yaw_gaze += loss_yaw_gaze

                loss_seq = [loss_pitch_gaze, loss_yaw_gaze]
                grad_seq = \
                    [torch.tensor(1.0).cuda(gpu) for _ in range(len(loss_seq))]

                optimizer_gaze.zero_grad(set_to_none=True)
                torch.autograd.backward(loss_seq, grad_seq)
                optimizer_gaze.step()

                iter_gaze += 1

                if (i+1) % 100 == 0:
                    writer.add_scalar('Loss/pitch_train', sum_loss_pitch_gaze/iter_gaze, epoch*len(dataset)//batch_size + i)
                    writer.add_scalar('Loss/yaw_train', sum_loss_yaw_gaze/iter_gaze, epoch*len(dataset)//batch_size + i)
                    print('Epoch [%d/%d], Iter [%d/%d] Losses: '
                        'Gaze Yaw %.4f,Gaze Pitch %.4f' % (
                            epoch+1,
                            num_epochs,
                            i+1,
                            len(dataset)//batch_size,
                            sum_loss_pitch_gaze/iter_gaze,
                            sum_loss_yaw_gaze/iter_gaze
                        )
                        )
                    
            # Save models at numbered epochs.
            if epoch % 1 == 0 and epoch < num_epochs:
                # DO VALIDATION EVERY EPOCH
                print("DOING VALIDATION")
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
                        loss_pitch_gaze = criterion(pitch, label_pitch_gaze)
                        loss_yaw_gaze = criterion(yaw, label_yaw_gaze)

                        pitch_predicted = softmax(pitch)
                        yaw_predicted = softmax(yaw)

                        pitch_predicted = \
                            torch.sum(pitch_predicted * idx_tensor, 1) * 3 - 42
                        yaw_predicted = \
                            torch.sum(yaw_predicted * idx_tensor, 1) * 3 - 42

                        # Add MAE metrics
                        val_yaw_mae += abs(label_yaw_cont_gaze - yaw_predicted)
                        val_pitch_mae += abs(label_pitch_cont_gaze - pitch_predicted)

                        loss_reg_pitch = reg_criterion(
                            pitch_predicted, label_pitch_cont_gaze)
                        loss_reg_yaw = reg_criterion(
                            yaw_predicted, label_yaw_cont_gaze)

                        # Total loss
                        loss_pitch_gaze += alpha * loss_reg_pitch
                        loss_yaw_gaze += alpha * loss_reg_yaw

                        val_sum_loss_pitch_gaze += loss_pitch_gaze
                        val_sum_loss_yaw_gaze += loss_yaw_gaze

                        val_iter_gaze += 1

                writer.add_scalar('Loss/epochs_pitch_train', sum_loss_pitch_gaze/iter_gaze, epoch)
                writer.add_scalar('Loss/epochs_yaw_train', sum_loss_yaw_gaze/iter_gaze, epoch)
                
                print(f"Epoch {epoch} validation losses. Pitch: {val_sum_loss_pitch_gaze/val_iter_gaze}, Yaw: {val_sum_loss_yaw_gaze/val_iter_gaze}")

                writer.add_scalar('Val/Loss_epochs_pitch', val_sum_loss_pitch_gaze/val_iter_gaze, epoch)
                writer.add_scalar('Val/Loss_epochs_yaw', val_sum_loss_yaw_gaze/val_iter_gaze, epoch)
                writer.add_scalar('Val/MAE_epochs_yaw', val_sum_loss_yaw_gaze/val_iter_gaze, epoch)
                writer.add_scalar('Val/MAE_epochs_yaw', val_sum_loss_yaw_gaze/val_iter_gaze, epoch)

                if not os.path.isdir(output+args.tb):
                    os.mkdir(output+args.tb)
                print('Taking snapshot...',
                    torch.save(model.state_dict(),
                                output+args.tb+'/'+
                                '_epoch_' + str(epoch+1) + '.pkl')
                    )
