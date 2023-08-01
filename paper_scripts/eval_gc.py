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
from meta_training_hydranet import Hydrant




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
    CALIBRATION_SET_SIZE = 30
    INNER_LOOP_LR = 1e-5
    INNER_STEPS = 10

    BATCH1 = False


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

    dataset_base_path = "/home/janek/gaze_capture_normalized"
    annotations_path = os.path.join(dataset_base_path, "annotations.txt")
    annotations = pd.read_csv(annotations_path, header=None,
                              delimiter=" ", index_col=0,
                              names=['yaw', 'pitch'])

    subjects = [s for s in sorted(os.listdir(dataset_base_path)) if not s.endswith(".txt")]


    pitch_steps = []
    yaw_steps = []
    # steps = [2, 3, 5, 10, 15, 20, 30]
    steps = [20, 30]


    for model_id in range(79, 80):
        losses_pitch = []
        losses_yaw = []
        for subject in subjects:
            images = sorted(os.listdir(os.path.join(dataset_base_path, subject)))
            if len(images) < 140:
                continue

            calibration_samples = images[:CALIBRATION_SET_SIZE]
            test_samples = images[CALIBRATION_SET_SIZE:CALIBRATION_SET_SIZE+100]

            support_cont = []
            support_binned = []
            support_images = []
            for s in calibration_samples:
                # label
                row_name = os.path.join(subject, s)
                yaw, pitch = annotations.loc[row_name].values
                # Convert yaw and pitch to angles
                pitch = pitch * 180 / np.pi
                yaw = yaw * 180 / np.pi
                bins = np.array(range(-28, 28, 3))
                binned_pose = np.digitize([pitch, yaw], bins) - 1
                cont_labels = [pitch, yaw]

                # image
                img_path = os.path.join(dataset_base_path, row_name)
                support_images.append(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB).astype(np.float32)/255.0)
                support_cont.append(cont_labels)
                support_binned.append(binned_pose)


            query_cont = []
            query_binned = []
            query_images = []
            for q in test_samples:
                # label
                row_name = os.path.join(subject, q)
                yaw, pitch = annotations.loc[row_name].values
                # Convert yaw and pitch to angles
                pitch = pitch * 180 / np.pi
                yaw = yaw * 180 / np.pi
                bins = np.array(range(-28, 28, 3))
                binned_pose = np.digitize([pitch, yaw], bins) - 1

                cont_labels = [pitch, yaw]

                query_cont.append(cont_labels)
                query_binned.append(binned_pose)
                # image
                img_path = os.path.join(dataset_base_path, row_name)
                query_images.append(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB).astype(np.float32)/255.0)


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

            support_images = transforms(support_images)
            query_images = transforms(query_images)

            model = torch.load(f"/home/janek/software/L2CS-Net/models/l2cs_maml_random_sample_user/model_epoch{str(model_id)}.pkl")
            model = nn.DataParallel(model, device_ids=[0])
            model.train()
            model.to(device)

            #! L2CS-Net
            # model= getArch("ResNet18", 28)
            # model = torch.nn.DataParallel(model, device_ids=[0])
            # state_dict = torch.load("/home/janek/software/L2CS-Net/models/meta/model_epoch29_state_dict2.pkl")
            # for key in list(state_dict.keys()):
            #     if 'module.module.' in key:
            #         state_dict[key.replace('module.module.', '')] = state_dict[key]
            #         del state_dict[key]
            # model.load_state_dict(state_dict)
            # model.train()
            # model.to(device)

            #! Hydrant (ours)
            # model = Hydrant("efficientnet_b0")
            # state_dict = torch.load("/home/janek/software/L2CS-Net/models/hydrant_effb3/model_epoch23_state_dict.pkl")
            # model.load_state_dict(state_dict)
            # model.train()
            # model.to(device)

            # model = getArch("ResNet18", 28)
            # saved_state_dict = torch.load("/home/janek/software/L2CS-Net/output/snapshots/legacy_verification/best.pkl")
            # model = nn.DataParallel(model, device_ids=[0])
            # model.load_state_dict(saved_state_dict)
            # model.cuda(device)

            optimizer = torch.optim.SGD(model.parameters(), INNER_LOOP_LR)

            s_im = torch.squeeze(support_images, 0)
            s_im = s_im.to(device)
            s_lc = torch.squeeze(support_cont, 0)
            s_lc = s_lc.to(device)
            s_lb = torch.squeeze(support_binned, 0)
            s_lb = s_lb.to(device)
            q_im = torch.squeeze(query_images, 0)
            q_im = q_im.to(device)
            q_lc = torch.squeeze(query_cont, 0)
            q_lc = q_lc.to(device)
            q_lb = torch.squeeze(query_binned, 0)
            q_lb = q_lb.to(device)

            s_label_pitch_gaze = Variable(s_lb[:, 0]).cuda(device)
            s_label_yaw_gaze = Variable(s_lb[:, 1]).cuda(device)
            s_label_pitch_cont_gaze = Variable(s_lc[:, 0]).cuda(device)
            s_label_yaw_cont_gaze = Variable(s_lc[:, 1]).cuda(device)

            q_label_pitch_gaze = Variable(q_lb[:, 0]).cuda(device)
            q_label_yaw_gaze = Variable(q_lb[:, 1]).cuda(device)
            q_label_pitch_cont_gaze = Variable(q_lc[:, 0]).cuda(device)
            q_label_yaw_cont_gaze = Variable(q_lc[:, 1]).cuda(device)
            for _ in range(INNER_STEPS):
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

                optimizer.step()

            # model.eval()
            # for m in model.modules():
            #         for child in m.children():
            #             if type(child) == nn.BatchNorm2d:
            #                 child.track_running_stats = False
            #                 child.running_mean = None
            #                 child.running_var = None

            if BATCH1:
                for q_image, q_label_pitch_cont, q_label_yaw_cont in zip(q_im, q_label_pitch_cont_gaze, q_label_yaw_cont_gaze):
                    with torch.no_grad():
                        q_image = q_image.unsqueeze(0)
                        pitch, yaw = model(q_image)

                        pitch_predicted = softmax(pitch)
                        yaw_predicted = softmax(yaw)

                        pitch_predicted = \
                        torch.sum(pitch_predicted * idx_tensor, 1) * 3 - 42
                        yaw_predicted = \
                        torch.sum(yaw_predicted * idx_tensor, 1) * 3 - 42

                        q_loss_reg_pitch = reg_criterion(pitch_predicted, q_label_pitch_cont.unsqueeze(0))
                        q_loss_reg_yaw = reg_criterion(yaw_predicted, q_label_yaw_cont.unsqueeze(0))
                        losses_pitch.append(q_loss_reg_pitch.detach())
                        losses_yaw.append(q_loss_reg_yaw.detach())
            else:
                pitch, yaw = model(q_im)
                pitch_predicted = softmax(pitch)
                yaw_predicted = softmax(yaw)

                pitch_predicted = \
                torch.sum(pitch_predicted * idx_tensor, 1) * 3 - 42
                yaw_predicted = \
                torch.sum(yaw_predicted * idx_tensor, 1) * 3 - 42

                q_loss_reg_pitch = reg_criterion(pitch_predicted, q_label_pitch_cont_gaze)
                q_loss_reg_yaw = reg_criterion(yaw_predicted, q_label_yaw_cont_gaze)
                losses_pitch.append(q_loss_reg_pitch.detach())
                losses_yaw.append(q_loss_reg_yaw.detach())

            del model

        print("### EVALUATION ###")
        print("Yaw: ", sum(losses_yaw)/len(losses_yaw))
        print("Pitch: ", sum(losses_pitch)/len(losses_pitch))
        pitch_steps.append(sum(losses_pitch)/len(losses_pitch))
        yaw_steps.append(sum(losses_yaw)/len(losses_yaw))

    print("Finish")