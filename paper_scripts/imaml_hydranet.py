
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

import clearml
import higher
import numpy as np
import os
import torch
from torch.autograd import Variable
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, random_split
import torch.utils.model_zoo as model_zoo
from torchvision import transforms


from meta_dataloader import WETMetaLoader


from meta_training_torchmeta_multistep import get_per_step_loss_importance_vector


class Hydrant(nn.Module):
    def __init__(self, architecture: str):
        super().__init__()
        if architecture == 'efficientnet_b0':
            self.net = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
            self.n_features = self.net.classifier[1].in_features
            self.net.classifier = nn.Identity()
            self.net.fc_yaw_reg = nn.Sequential(OrderedDict([('dropout2', nn.Dropout(p=0.2)),
                                                             ('final', nn.Linear(self.n_features, 1))]))
            self.net.fc_pitch_reg = nn.Sequential(OrderedDict([('dropout2', nn.Dropout(p=0.2)),
                                                               ('final', nn.Linear(self.n_features, 1))]))

        elif architecture == 'efficientnet_b3':
            self.net = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
            self.n_features = self.net.classifier[1].in_features
            self.net.classifier = nn.Identity()
            self.net.fc_yaw_reg = nn.Sequential(OrderedDict([('dropout2', nn.Dropout(p=0.2)),
                                                             ('final', nn.Linear(self.n_features, 1))]))
            self.net.fc_pitch_reg = nn.Sequential(OrderedDict([('dropout2', nn.Dropout(p=0.2)),
                                                               ('final', nn.Linear(self.n_features, 1))]))

        else:
            raise ValueError(f"{architecture} is not implemented yet.")


    def forward(self, x):
        # TODO: Check if double pass is required or self.net(x) can be computed once
        logits = self.net(x)
        yaw_head_reg = self.net.fc_yaw_reg(logits)
        pitch_head_reg = self.net.fc_pitch_reg(logits)


        return yaw_head_reg, pitch_head_reg

def hv_prod(in_grad, x, params, lamb):
        hv = torch.autograd.grad(in_grad, params, retain_graph=True, grad_outputs=x)
        hv = torch.nn.utils.parameters_to_vector(hv).detach()
        # precondition with identity matrix
        return hv/lamb + x

def cg(in_grad, outer_grad, params, model, lamb, n_cg):
        x = outer_grad.clone().detach()
        r = outer_grad.clone().detach() - hv_prod(in_grad, x, params, lamb)
        p = r.clone().detach()
        for i in range(n_cg):
            Ap = hv_prod(in_grad, p, params, lamb)
            alpha = (r @ r)/(p @ Ap)
            x = x + alpha * p
            r_new = r - alpha * Ap
            beta = (r_new @ r_new)/(r @ r)
            p = r_new + beta * p
            r = r_new.clone().detach()
        return vec_to_grad(x, model)

def vec_to_grad(vec, model):
    pointer = 0
    res = []
    for param in model.parameters():
        num_param = param.numel()
        res.append(vec[pointer:pointer+num_param].view_as(param).data)
        pointer += num_param
    return res

def mix_grad(grad_list, weight_list):
    '''
    calc weighted average of gradient
    '''
    mixed_grad = []
    for g_list in zip(*grad_list):
        g_list = torch.stack([weight_list[i] * g_list[i] for i in range(len(weight_list))])
        mixed_grad.append(torch.sum(g_list, dim=0))
    return mixed_grad

def apply_grad(model, grad):
    '''
    assign gradient to model(nn.Module) instance. return the norm of gradient
    '''
    grad_norm = 0
    for p, g in zip(model.parameters(), grad):
        if p.grad is None:
            p.grad = g
        else:
            p.grad += g
        grad_norm += torch.sum(g**2)
    grad_norm = grad_norm ** (1/2)
    return grad_norm.item()

if __name__ == '__main__':

    META_BATCH_SIZE = 1
    META_LR = 5e-5
    INNER_LOOP_LR = 1e-4
    EPOCHS = 30
    INNER_STEPS = 10
    SUPPORT_SIZE = 20
    QUERY_SIZE = 30

    LAMB = 2
    N_CG = 5

    task = clearml.Task.init(project_name="meta", task_name="hydrant_b0_imaml", tags="v2")
    logger = task.get_logger()
    parameters = task.connect({})
    parameters['meta_batch_size'] = META_BATCH_SIZE
    parameters['meta_lr'] = META_LR
    parameters['inner_loop_lr'] = INNER_LOOP_LR
    parameters['epochs'] = EPOCHS
    parameters['inner_steps'] = INNER_STEPS
    parameters['support_size'] = SUPPORT_SIZE
    parameters['query_size'] = QUERY_SIZE


    transformations = transforms.Compose([
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225] # RGB
        )
    ])
    device = torch.device('cuda')

    model = Hydrant("efficientnet_b0")
    model.to(device)

    reg_criterion = nn.L1Loss().cuda(device)

    meta_optimizer = torch.optim.Adam(model.parameters(), META_LR)


    data = WETMetaLoader(annotations="/home/janek/software/L2CS-Net/meta_dataset_normalized/annotations.txt",
                         root="/home/janek/software/L2CS-Net/meta_dataset_normalized",
                         nshot_support=SUPPORT_SIZE, n_query=QUERY_SIZE,
                         transforms=transformations)

    meta_train = Subset(data, range(int(.8*len(data))))
    meta_test = Subset(data, range(int(.8*len(data)), len(data)))


    meta_train_loader = DataLoader(dataset=meta_train,
                                   batch_size=META_BATCH_SIZE,
                                   shuffle=True,
                                   num_workers=1,
                                   pin_memory=False)

    meta_test_loader = DataLoader(dataset=meta_test,
                                   batch_size=1,
                                   shuffle=True,
                                   num_workers=1,
                                   pin_memory=False)

    meta_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(meta_optimizer, T_max=len(meta_train_loader), eta_min=0)

    iters = 0
    for epoch in range(EPOCHS):
        model.train()

        for i, (b_s_im, b_s_lc, b_s_lb, b_q_im, b_q_lc, b_q_lb) in enumerate(meta_train_loader):
            print(f"Iter {i+1} / {len(meta_train_loader.dataset)}")
            inner_optimizer = torch.optim.SGD(model.parameters(), INNER_LOOP_LR)
            meta_optimizer.zero_grad()

            qry_accs = []

            grad_list = []
            for s_im, s_lc, s_lb, q_im, q_lc, q_lb in zip(b_s_im, b_s_lc, b_s_lb, b_q_im, b_q_lc, b_q_lb):
                iters += 1
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

                s_label_pitch_cont_gaze = Variable(s_lc[:, 0]).cuda(device)
                s_label_yaw_cont_gaze = Variable(s_lc[:, 1]).cuda(device)

                q_label_pitch_cont_gaze = Variable(q_lc[:, 0]).cuda(device)
                q_label_yaw_cont_gaze = Variable(q_lc[:, 1]).cuda(device)

                # setup model
                # model.zero_grad()
                importance_vector = get_per_step_loss_importance_vector(epoch, device, INNER_STEPS, EPOCHS)
                losses = []
                with higher.innerloop_ctx(model, inner_optimizer, track_higher_grads=False) as (fnet, diffopt):
                    for step_number in range(INNER_STEPS):
                        pitch_s, yaw_s = fnet(s_im)

                        s_loss_reg_pitch = reg_criterion(pitch_s.ravel(), s_label_pitch_cont_gaze)
                        s_loss_reg_yaw = reg_criterion(yaw_s.ravel(), s_label_yaw_cont_gaze)


                        loss_seq = s_loss_reg_pitch + s_loss_reg_yaw
                        # torch.autograd.backward(loss_seq, grad_seq)
                        diffopt.step(loss_seq)

                    pitch_s, yaw_s = fnet(s_im)
                    s_loss_reg_pitch = reg_criterion(pitch_s.ravel(), s_label_pitch_cont_gaze)
                    s_loss_reg_yaw = reg_criterion(yaw_s.ravel(), s_label_yaw_cont_gaze)
                    train_loss = s_loss_reg_pitch + s_loss_reg_yaw

                    pitch_q, yaw_q = fnet(q_im)
                    q_loss_reg_pitch = reg_criterion(pitch_q.ravel(), q_label_pitch_cont_gaze)
                    q_loss_reg_yaw = reg_criterion(yaw_q.ravel(), q_label_yaw_cont_gaze)
                    test_loss = q_loss_reg_pitch + q_loss_reg_yaw

                    qry_accs.append(test_loss)

                    params = list(fnet.parameters(time=-1))
                    in_grad = torch.nn.utils.parameters_to_vector(torch.autograd.grad(train_loss, params, create_graph=True))
                    outer_grad = torch.nn.utils.parameters_to_vector(torch.autograd.grad(test_loss, params))
                    implicit_grad = cg(in_grad, outer_grad, params, model, LAMB, N_CG)
                    grad_list.append(implicit_grad)

                meta_optimizer.zero_grad()
                weight = torch.ones(len(grad_list))
                weight = weight / torch.sum(weight)
                meta_optimizer.step()
                grad = mix_grad(grad_list, weight)
                grad_log = apply_grad(model, grad)
                meta_optimizer.step()


            print(sum(qry_accs) / len(qry_accs))
            logger.report_scalar("Query", "Loss", iteration=iters, value=(sum(qry_accs) / len(qry_accs)))
            logger.report_scalar("Params", "LR", iteration=iters, value=meta_scheduler.get_last_lr()[0])
            # meta_scheduler.step()

        torch.save(model.state_dict(), os.path.join("/home/janek/software/L2CS-Net/models/imaml_hydrant_effb0", f"model_epoch{epoch}_state_dict.pkl"))
