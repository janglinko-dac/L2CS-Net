
import matplotlib.pyplot as plt
import math
import numpy as np
import torch
from torch.autograd import Variable
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import torchvision

from meta_dataloader import WETMetaLoader
from clear_training_utils import (parse_args,
                                  get_fc_params,
                                  get_non_ignored_params,
                                  get_ignored_params,
                                  getArch_weights,
                                  load_filtered_state_dict)


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

    INNER_LOOP_LR = 5e-6
    INNER_STEPS = 7
    SUPPORT_SIZE = 20
    QUERY_SIZE = 50

    # task = clearml.Task.init(project_name="meta", task_name="final", tags="v2")
    # logger = task.get_logger()
    # parameters = task.connect({})
    # parameters['meta_batch_size'] = META_BATCH_SIZE
    # parameters['meta_lr'] = META_LR
    # parameters['inner_loop_lr'] = INNER_LOOP_LR
    # parameters['epochs'] = EPOCHS
    # parameters['inner_steps'] = INNER_STEPS
    # parameters['support_size'] = SUPPORT_SIZE
    # parameters['query_size'] = QUERY_SIZE


    transformations = transforms.Compose([
        transforms.Resize(448),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225] # RGB
        )
    ])
    device = torch.device('cuda')
    softmax = nn.Softmax(dim=1).cuda(device)

    # model, pre_url = getArch_weights("ResNet18", 28)
    # load_filtered_state_dict(model, model_zoo.load_url(pre_url))
    # model = nn.DataParallel(model)
    # model.to(device)

    criterion = nn.CrossEntropyLoss().cuda(device)
    reg_criterion = nn.L1Loss().cuda(device)

    idx_tensor = [idx for idx in range(28)]
    idx_tensor = Variable(torch.FloatTensor(idx_tensor)).cuda(device)



    pitch_steps = []
    yaw_steps = []

    support_sizes = [2, 3, 5, 10, 15, 20]
    for inner_steps in range(1):

        data = WETMetaLoader(annotations="/home/janek/software/L2CS-Net/meta_dataset_normalized/annotations.txt",
                         root="/home/janek/software/L2CS-Net/meta_dataset_normalized",
                         nshot_support=SUPPORT_SIZE, n_query=QUERY_SIZE,
                         transforms=transformations)

        proportions = [.8, .2]
        lengths = [int(p * len(data)) for p in proportions]
        lengths[-1] = len(data) - sum(lengths[:-1])
        meta_train, meta_test = random_split(data, lengths, generator=torch.Generator().manual_seed(42))
        # meta_train, meta_test = random_split(data, [0.8, 0.2], generator=torch.Generator().manual_seed(42))

        meta_train_loader = DataLoader(dataset=meta_train,
                                    batch_size=1,
                                    shuffle=True,
                                    num_workers=1,
                                    pin_memory=False)

        meta_test_loader = DataLoader(dataset=meta_test,
                                    batch_size=1,
                                    shuffle=True,
                                    num_workers=1,
                                    pin_memory=False)

        losses_pitch = []
        losses_yaw = []
        for i, (b_s_im, b_s_lc, b_s_lb, b_q_im, b_q_lc, b_q_lb) in enumerate(meta_test_loader):
            # if i == 100:
            #     break
            # model = torch.load("/home/janek/software/L2CS-Net/models/meta/model_epoch29.pkl")
            # model.eval()
            # model.to(device)
            model = getArch("ResNet18", 28)
            saved_state_dict = torch.load("/home/janek/software/L2CS-Net/output/snapshots/legacy_verification/best.pkl")
            model = nn.DataParallel(model, device_ids=[0])
            model.load_state_dict(saved_state_dict)
            model.cuda(device)
            # Have to evaluate model at the end to run predictions
            # model.eval()

            for m in model.modules():
                for child in m.children():
                    if type(child) == nn.BatchNorm2d:
                        child.track_running_stats = False
                        child.running_mean = None
                        child.running_var = None

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
                q_lb = torch.squeeze(q_lb, 0)
                q_lb = q_lb.to(device)

                s_label_pitch_gaze = Variable(s_lb[:, 0]).cuda(device)
                s_label_yaw_gaze = Variable(s_lb[:, 1]).cuda(device)
                s_label_pitch_cont_gaze = Variable(s_lc[:, 0]).cuda(device)
                s_label_yaw_cont_gaze = Variable(s_lc[:, 1]).cuda(device)

                q_label_pitch_gaze = Variable(q_lb[:, 0]).cuda(device)
                q_label_yaw_gaze = Variable(q_lb[:, 1]).cuda(device)
                q_label_pitch_cont_gaze = Variable(q_lc[:, 0]).cuda(device)
                q_label_yaw_cont_gaze = Variable(q_lc[:, 1]).cuda(device)

                # setup model
                # model.zero_grad()

                for _ in range(inner_steps):
                    pitch, yaw = model(s_im)

                    s_loss_pitch_gaze = criterion(pitch, s_label_pitch_gaze.to(torch.long))
                    s_loss_yaw_gaze = criterion(yaw, s_label_pitch_gaze.to(torch.long))
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
