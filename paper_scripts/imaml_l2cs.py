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
from clear_training_utils import (parse_args,
                                  get_fc_params,
                                  get_non_ignored_params,
                                  get_ignored_params,
                                  getArch_weights,
                                  load_filtered_state_dict)

from meta_dataloader import WETMetaLoader


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
    META_LR = 1e-5
    INNER_LOOP_LR = 1e-5
    EPOCHS = 30
    INNER_STEPS = 30
    SUPPORT_SIZE = 20
    QUERY_SIZE = 30

    LAMB = 2
    N_CG = 5

    task = clearml.Task.init(project_name="meta", task_name="l2cs_imaml", tags="v2")
    logger = task.get_logger()
    parameters = task.connect({})
    parameters['meta_batch_size'] = META_BATCH_SIZE
    parameters['meta_lr'] = META_LR
    parameters['inner_loop_lr'] = INNER_LOOP_LR
    parameters['epochs'] = EPOCHS
    parameters['inner_steps'] = INNER_STEPS
    parameters['support_size'] = SUPPORT_SIZE
    parameters['query_size'] = QUERY_SIZE
    parameters['lamb'] = LAMB
    parameters['n_cg'] = N_CG



    transformations = transforms.Compose([
        transforms.Resize(448),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225] # RGB
        )
    ])
    device = torch.device('cuda')

    softmax = nn.Softmax(dim=1).cuda(device)

    model, pre_url = getArch_weights("ResNet18", 28)
    load_filtered_state_dict(model, model_zoo.load_url(pre_url))
    # model = torch.load("/home/janek/software/L2CS-Net/models/meta/model_epoch14.pkl")
    model = nn.DataParallel(model)
    model.to(device)

    criterion = nn.CrossEntropyLoss().cuda(device)
    reg_criterion = nn.L1Loss().cuda(device)

    idx_tensor = [idx for idx in range(28)]
    idx_tensor = Variable(torch.FloatTensor(idx_tensor)).cuda(device)

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
                losses = []
                with higher.innerloop_ctx(model, inner_optimizer, track_higher_grads=False) as (fnet, diffopt):
                    for step_number in range(INNER_STEPS):
                        pitch, yaw = fnet(s_im)

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
                        diffopt.step(loss_seq)

                    pitch_s, yaw_s = fnet(s_im)

                    s_loss_pitch_gaze = criterion(pitch_s, s_label_pitch_gaze.to(torch.long))
                    s_loss_yaw_gaze = criterion(yaw_s, s_label_yaw_gaze.to(torch.long))
                    pitch_predicted_s = softmax(pitch_s)
                    yaw_predicted_s = softmax(yaw_s)
                    pitch_predicted_s = \
                    torch.sum(pitch_predicted_s * idx_tensor, 1) * 3 - 42
                    yaw_predicted_s = \
                    torch.sum(yaw_predicted_s * idx_tensor, 1) * 3 - 42

                    s_loss_reg_pitch = reg_criterion(pitch_predicted_s, s_label_pitch_cont_gaze)
                    s_loss_reg_yaw = reg_criterion(yaw_predicted_s, s_label_yaw_cont_gaze)

                    # Total loss
                    s_loss_pitch_gaze += s_loss_reg_pitch
                    s_loss_yaw_gaze += s_loss_reg_yaw

                    train_loss = s_loss_pitch_gaze + s_loss_yaw_gaze

                    pitch_q, yaw_q = fnet(q_im)

                    q_loss_pitch_gaze = criterion(pitch_q, q_label_pitch_gaze.to(torch.long))
                    q_loss_yaw_gaze = criterion(yaw_q, q_label_yaw_gaze.to(torch.long))
                    pitch_predicted_q = softmax(pitch_q)
                    yaw_predicted_q = softmax(yaw_q)
                    pitch_predicted_q = \
                    torch.sum(pitch_predicted_q * idx_tensor, 1) * 3 - 42
                    yaw_predicted_q = \
                    torch.sum(yaw_predicted_q * idx_tensor, 1) * 3 - 42

                    q_loss_reg_pitch = reg_criterion(pitch_predicted_q, q_label_pitch_cont_gaze)
                    q_loss_reg_yaw = reg_criterion(yaw_predicted_q, q_label_yaw_cont_gaze)

                    # Total loss
                    q_loss_pitch_gaze += q_loss_reg_pitch
                    q_loss_yaw_gaze += q_loss_reg_yaw

                    test_loss = q_loss_pitch_gaze + q_loss_yaw_gaze

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

        torch.save(model.state_dict(), os.path.join("/home/janek/software/L2CS-Net/models/imaml_l2cs", f"model_epoch{epoch}_state_dict.pkl"))
