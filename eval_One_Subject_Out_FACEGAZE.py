import cv2
import numpy as np
import os
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.transforms as t
import torchvision
import math
# from meta_training_hydranet import Hydrant
from torch.utils.data import DataLoader, Subset, random_split
from meta_dataloader import WETMetaLoader
from mpiifacegaze_loader import MpiiFaceGazeMetaLoader
import xgboost as xg



class L2CS(nn.Module):
    def __init__(self, block, layers, num_bins):
        self.inplanes = 64
        super(L2CS, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc_yaw_gaze = nn.Linear(512 * block.expansion, num_bins)
        self.fc_pitch_gaze = nn.Linear(512 * block.expansion, num_bins)

       # Vestigial layer from previous experiments
        self.fc_finetune = nn.Linear(512 * block.expansion + 3, 3)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        # gaze
        pre_yaw_gaze =  self.fc_yaw_gaze(x)
        pre_pitch_gaze = self.fc_pitch_gaze(x)
        return pre_yaw_gaze, pre_pitch_gaze


def getArch(arch,bins):
    # Base network structure
    if arch == 'ResNet18':
        model = L2CS( torchvision.models.resnet.BasicBlock,[2, 2,  2, 2], bins)
    elif arch == 'ResNet34':
        model = L2CS( torchvision.models.resnet.BasicBlock,[3, 4,  6, 3], bins)
    elif arch == 'ResNet101':
        model = L2CS( torchvision.models.resnet.Bottleneck,[3, 4, 23, 3], bins)
    elif arch == 'ResNet152':
        model = L2CS( torchvision.models.resnet.Bottleneck,[3, 8, 36, 3], bins)
    else:
        if arch != 'ResNet50':
            print('Invalid value for architecture is passed! '
                'The default value of ResNet50 will be used instead!')
        model = L2CS( torchvision.models.resnet.Bottleneck, [3, 4, 6,  3], bins)
    return model


if __name__ == '__main__':
    INNER_LOOP_LR = 1e-3
    INNER_STEPS = 20

    BATCH1 = False

    SUPPORT_SIZE = 200
    QUERY_SIZE = 500

    APPLY_BOOSTING = True

    device = torch.device('cuda')
    softmax = nn.Softmax(dim=1).cuda(device)

    transforms = t.Compose([
        t.Resize(448),
        t.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225] # RGB
        )
    ])

    criterion = nn.CrossEntropyLoss().cuda(device)
    reg_criterion = nn.L1Loss().cuda(device)

    idx_tensor = [idx for idx in range(28)]
    idx_tensor = Variable(torch.FloatTensor(idx_tensor)).cuda(device)


    pitch_steps = []
    yaw_steps = []
    for user in range(15):
        data = MpiiFaceGazeMetaLoader(
                            base_path="/home/janek/MPIIFaceGazeNorm",
                            support_size=SUPPORT_SIZE, query_size=QUERY_SIZE,
                            transforms=transforms, exclude=False, one_out=f"p{user:02d}.label")


        meta_test_loader = DataLoader(dataset=data,
                                    batch_size=1,
                                    shuffle=True,
                                    num_workers=1,
                                    pin_memory=False)

        for i, (b_s_im, b_s_lc, b_s_lb, b_q_im, b_q_lc, b_q_lb) in enumerate(meta_test_loader):
            print(f"Iter {i+1} / {len(meta_test_loader.dataset)}")
            model = torch.load(f"/home/janek/software/L2CS-Net/models/mpii_{user:02d}/model_epoch499.pkl")
            model = nn.DataParallel(model, device_ids=[0])
            model.train()
            model.to(device)
            optimizer = torch.optim.SGD(model.parameters(), INNER_LOOP_LR)


            for s_im, s_lc, s_lb, q_im, q_lc, q_lb in zip(b_s_im, b_s_lc, b_s_lb, b_q_im, b_q_lc, b_q_lb):
            # Prepare data
                s_im = torch.squeeze(s_im, 0)
                s_im = s_im.to(device)
                s_lc = torch.squeeze(s_lc, 0)
                s_lc = s_lc.to(device)
                s_lb = torch.squeeze(s_lb, 0)
                s_lb = s_lb.to(device)
                q_im = torch.squeeze(q_im, 0)
                q_im = q_im.to(device)
                q_lc = torch.squeeze(q_lc, 0)
                q_lc = q_lc.to(device)

                s_label_pitch_gaze = Variable(s_lb[:, 0]).cuda(device)
                s_label_yaw_gaze = Variable(s_lb[:, 1]).cuda(device)
                s_label_pitch_cont_gaze = Variable(s_lc[:, 0]).cuda(device)
                s_label_yaw_cont_gaze = Variable(s_lc[:, 1]).cuda(device)

                q_label_pitch_cont_gaze = Variable(q_lc[:, 0]).cuda(device)
                q_label_yaw_cont_gaze = Variable(q_lc[:, 1]).cuda(device)


                for step_number in range(INNER_STEPS):
                    optimizer.zero_grad()
                    pitch, yaw = model(s_im)

                    s_loss_pitch_gaze = criterion(pitch, s_label_pitch_gaze.to(torch.long))
                    s_loss_yaw_gaze = criterion(yaw, s_label_yaw_gaze.to(torch.long))
                    pitch_predicted = softmax(pitch)
                    yaw_predicted = softmax(yaw)
                    pitch_predicted = \
                    torch.sum(pitch_predicted * idx_tensor, 1) * 3 - 42
                    yaw_predicted = \
                    torch.sum(yaw_predicted * idx_tensor, 1) * 3 - 42

                    s_loss_reg_pitch = reg_criterion(pitch_predicted, s_label_pitch_cont_gaze)
                    s_loss_reg_yaw = reg_criterion(yaw_predicted, s_label_yaw_cont_gaze)

                    # Total loss
                    s_loss_pitch_gaze += s_loss_reg_pitch
                    s_loss_yaw_gaze += s_loss_reg_yaw

                    loss_seq = s_loss_pitch_gaze + s_loss_yaw_gaze
                    loss_seq.backward()
                    # torch.autograd.backward(loss_seq, grad_seq)
                    optimizer.step()

                if APPLY_BOOSTING:
                    with torch.no_grad():
                        pitch, yaw = model(s_im)
                        pitch_predicted = softmax(pitch)
                        yaw_predicted = softmax(yaw)
                        pitch_predicted = \
                        torch.sum(pitch_predicted * idx_tensor, 1) * 3 - 42
                        yaw_predicted = \
                        torch.sum(yaw_predicted * idx_tensor, 1) * 3 - 42
                    pitch_boost = xg.XGBRegressor(objective ='reg:squarederror', gamma=0.2)
                    yaw_boost = xg.XGBRegressor(objective ='reg:squarederror', gamma=0.2)

                    pitch_train = pitch_predicted.cpu().detach().numpy()
                    yaw_train = yaw_predicted.cpu().detach().numpy()
                    pitch_label = s_label_pitch_cont_gaze.cpu().detach().numpy()
                    yaw_label = s_label_yaw_cont_gaze.cpu().detach().numpy()

                    pitch_boost.fit(pitch_train.reshape(-1, 1), pitch_label.reshape(-1, 1))
                    yaw_boost.fit(yaw_train.reshape(-1, 1), yaw_label.reshape(-1, 1))


                with torch.no_grad():
                    for b in range(2):
                        q_label_pitch_cont_gaze_batch = q_label_pitch_cont_gaze[b:(b+1)*250]
                        q_label_yaw_cont_gaze_batch = q_label_yaw_cont_gaze[b:(b+1)*250]
                        q_im_batch = q_im[b:(b+1)*250]
                        pitch, yaw = model(q_im_batch)

                        pitch_predicted = softmax(pitch)
                        yaw_predicted = softmax(yaw)

                        pitch_predicted = \
                        torch.sum(pitch_predicted * idx_tensor, 1) * 3 - 42
                        yaw_predicted = \
                        torch.sum(yaw_predicted * idx_tensor, 1) * 3 - 42

                        q_loss_reg_pitch = reg_criterion(pitch_predicted, q_label_pitch_cont_gaze_batch).item()
                        q_loss_reg_yaw = reg_criterion(yaw_predicted, q_label_yaw_cont_gaze_batch).item()
                        if APPLY_BOOSTING:
                            val_pitch_input = pitch_predicted.detach().cpu().numpy()
                            val_yaw_input = yaw_predicted.detach().cpu().numpy()

                            val_pitch_boosted = pitch_boost.predict(val_pitch_input.reshape(-1, 1))
                            val_yaw_boosted = yaw_boost.predict(val_yaw_input.reshape(-1, 1))

                            val_pitch_label = q_label_pitch_cont_gaze_batch.detach().cpu().numpy()
                            val_yaw_label = q_label_yaw_cont_gaze_batch.detach().cpu().numpy()

                            pitch_steps.append(np.mean(np.abs(val_pitch_boosted - val_pitch_label)))
                            yaw_steps.append(np.mean(np.abs(val_yaw_boosted - val_yaw_label)))
                        else:
                            pitch_steps.append(q_loss_reg_pitch)
                            yaw_steps.append(q_loss_reg_yaw)

            del model
    print("### EVALUATION ###")
    print("Yaw: ", sum(yaw_steps)/len(yaw_steps))
    print("Pitch: ", sum(pitch_steps)/len(pitch_steps))
    print("finish")
