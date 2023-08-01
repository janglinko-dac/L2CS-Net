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


def get_per_step_loss_importance_vector(current_epoch, device,
                                        number_of_training_steps_per_iter,
                                        multi_step_loss_num_epochs):
        """
        Generates a tensor of dimensionality (num_inner_loop_steps) indicating the importance of each step's target
        loss towards the optimization loss.
        :return: A tensor to be used to compute the weighted average of the loss, useful for
        the MSL (Multi Step Loss) mechanism.
        """
        number_of_training_steps_per_iter = number_of_training_steps_per_iter
        multi_step_loss_num_epochs = 15


        loss_weights = np.ones(shape=(number_of_training_steps_per_iter)) * (
                1.0 / number_of_training_steps_per_iter)
        decay_rate = 1.0 / number_of_training_steps_per_iter / multi_step_loss_num_epochs
        min_value_for_non_final_losses = 0.03 / number_of_training_steps_per_iter
        for i in range(len(loss_weights) - 1):
            curr_value = np.maximum(loss_weights[i] - (current_epoch * decay_rate), min_value_for_non_final_losses)
            loss_weights[i] = curr_value

        curr_value = np.minimum(
            loss_weights[-1] + (current_epoch * (number_of_training_steps_per_iter - 1) * decay_rate),
            1.0 - ((number_of_training_steps_per_iter - 1) * min_value_for_non_final_losses))
        loss_weights[-1] = curr_value
        loss_weights = torch.Tensor(loss_weights).to(device=device)
        return loss_weights


if __name__ == '__main__':

    META_BATCH_SIZE = 1
    META_LR = 5e-5
    INNER_LOOP_LR = 1e-5
    EPOCHS = 30
    INNER_STEPS = 3
    SUPPORT_SIZE = 20
    QUERY_SIZE = 30

    task = clearml.Task.init(project_name="meta", task_name="l2cs_maml_gc_portrait_mode", tags="v2")
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
    # model = torch.load("/home/janek/software/L2CS-Net/models/l2cs_maml_random_sample_user/model_epoch79.pkl")
    model = nn.DataParallel(model)
    model.to(device)

    criterion = nn.CrossEntropyLoss().cuda(device)
    reg_criterion = nn.L1Loss().cuda(device)

    idx_tensor = [idx for idx in range(28)]
    idx_tensor = Variable(torch.FloatTensor(idx_tensor)).cuda(device)

    # Optimizer gaze
    # meta_optimizer = torch.optim.Adam([
    #     {'params': get_ignored_params(model), 'lr': 0},
    #     {'params': get_non_ignored_params(model), 'lr': 1e-6},
    #     {'params': get_fc_params(model), 'lr': 1e-6}
    # ], 1e-6)
    meta_optimizer = torch.optim.Adam(model.parameters(), META_LR)

    # inner_optimizer = torch.optim.SGD([
    #                             {'params': get_ignored_params(model), 'lr': 0},
    #                             {'params': get_non_ignored_params(model), 'lr': 5e-4},
    #                             {'params': get_fc_params(model), 'lr': 5e-4}
    #                             ], 5e-4)

    # data = WETMetaLoader(annotations="/home/janek/software/L2CS-Net/meta_dataset_normalized/annotations.txt",
    #                      root="/home/janek/software/L2CS-Net/meta_dataset_normalized",
    #                      nshot_support=SUPPORT_SIZE, n_query=QUERY_SIZE,
    #                      transforms=transformations)

    data = WETMetaLoader(annotations="/home/janek/GazeCaptureNormalizedTrain/annotations.txt",
                         root="/home/janek/GazeCaptureNormalizedTrain",
                         nshot_support=SUPPORT_SIZE, n_query=QUERY_SIZE,
                         transforms=transformations)

    # proportions = [.9, .1]
    # lengths = [int(p * len(data)) for p in proportions]
    # lengths[-1] = len(data) - sum(lengths[:-1])
    # meta_train, meta_test = random_split(data, lengths, generator=torch.Generator().manual_seed(42))
    # meta_train = Subset(data, range(int(.9*len(data))))
    # meta_test = Subset(data, range(int(.1*len(data)), len(data)))

    # meta_train, meta_test = random_split(data, [0.8, 0.2], generator=torch.Generator().manual_seed(42))

    meta_train_loader = DataLoader(dataset=data,
                                   batch_size=META_BATCH_SIZE,
                                   shuffle=True,
                                   num_workers=1,
                                   pin_memory=False)

    # meta_test_loader = DataLoader(dataset=meta_test,
    #                                batch_size=1,
    #                                shuffle=True,
    #                                num_workers=1,
    #                                pin_memory=False)

    meta_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(meta_optimizer, T_max=len(meta_train_loader), eta_min=0)

    iters = 0
    for epoch in range(EPOCHS):
        model.train()

        for i, (b_s_im, b_s_lc, b_s_lb, b_q_im, b_q_lc, b_q_lb) in enumerate(meta_train_loader):
            print(f"Iter {i+1} / {len(meta_train_loader.dataset)}")
            inner_optimizer = torch.optim.SGD(model.parameters(), INNER_LOOP_LR)
            meta_optimizer.zero_grad()

            qry_accs = []

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
                importance_vector = get_per_step_loss_importance_vector(epoch, device, INNER_STEPS, EPOCHS)
                losses = []
                with higher.innerloop_ctx(model, inner_optimizer, copy_initial_weights=False) as (fnet, diffopt):
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

                        # loss_reg_pitch = F.mse_loss(pitch_predicted, label_pitch_cont_gaze)
                        # loss_reg_yaw = F.mse_loss(yaw_predicted, label_yaw_cont_gaze)
                        s_loss_reg_pitch = reg_criterion(pitch_predicted, s_label_pitch_cont_gaze)
                        s_loss_reg_yaw = reg_criterion(yaw_predicted, s_label_yaw_cont_gaze)

                        # Total loss
                        s_loss_pitch_gaze += s_loss_reg_pitch
                        s_loss_yaw_gaze += s_loss_reg_yaw

                        # loss_seq = [s_loss_pitch_gaze, s_loss_yaw_gaze]
                        # grad_seq = [torch.tensor(1.0).cuda(device) for _ in range(len(loss_seq))]
                        loss_seq = s_loss_pitch_gaze + s_loss_yaw_gaze
                        # torch.autograd.backward(loss_seq, grad_seq)
                        diffopt.step(loss_seq)
                        #? I dont know how to pass the griadients sequence here

                        pitch, yaw = fnet(q_im)
                        # loss_pitch_gaze = F.cross_entropy(pitch, label_pitch_gaze)
                        # loss_yaw_gaze = F.cross_entropy(yaw, label_yaw_gaze)
                        q_loss_pitch_gaze = criterion(pitch, q_label_pitch_gaze.to(torch.long))
                        q_loss_yaw_gaze = criterion(yaw, q_label_yaw_gaze.to(torch.long))

                        pitch_predicted = softmax(pitch)
                        yaw_predicted = softmax(yaw)

                        pitch_predicted = \
                        torch.sum(pitch_predicted * idx_tensor, 1) * 3 - 42
                        yaw_predicted = \
                        torch.sum(yaw_predicted * idx_tensor, 1) * 3 - 42

                        # loss_reg_pitch = F.mse_loss(pitch_predicted, label_pitch_cont_gaze)
                        # loss_reg_yaw = F.mse_loss(yaw_predicted, label_yaw_cont_gaze)
                        q_loss_reg_pitch = reg_criterion(pitch_predicted, q_label_pitch_cont_gaze)
                        q_loss_reg_yaw = reg_criterion(yaw_predicted, q_label_yaw_cont_gaze)

                        # Total loss
                        q_loss_pitch_gaze += q_loss_reg_pitch
                        q_loss_yaw_gaze += q_loss_reg_yaw
                        losses.append(q_loss_pitch_gaze + q_loss_yaw_gaze)
                        # q_loss_seq = 0
                        # for s in range(step_number+1):
                        #     q_loss_seq += importance_vector[s]*losses[s]
                        # if step_number != INNER_STEPS:
                        #     q_loss_seq.backward(retain_graph=True)
                        # else:
                        #     q_loss_seq.backward()
                        # meta_optimizer.step()
                train_mae = q_loss_reg_pitch.item() + q_loss_reg_yaw.item()
                outer_loss = 0
                for s in range(INNER_STEPS):
                    outer_loss += importance_vector[s]*losses[s]
                outer_loss.backward()
                meta_optimizer.step()
                    # pitch, yaw = fnet(q_im)
                    # # loss_pitch_gaze = F.cross_entropy(pitch, label_pitch_gaze)
                    # # loss_yaw_gaze = F.cross_entropy(yaw, label_yaw_gaze)
                    # q_loss_pitch_gaze_local = criterion(pitch, q_label_pitch_gaze.to(torch.long))
                    # q_loss_yaw_gaze_local = criterion(yaw, q_label_yaw_gaze.to(torch.long))

                    # pitch_predicted = softmax(pitch)
                    # yaw_predicted = softmax(yaw)

                    # pitch_predicted = \
                    # torch.sum(pitch_predicted * idx_tensor, 1) * 3 - 42
                    # yaw_predicted = \
                    # torch.sum(yaw_predicted * idx_tensor, 1) * 3 - 42

                    # # loss_reg_pitch = F.mse_loss(pitch_predicted, label_pitch_cont_gaze)
                    # # loss_reg_yaw = F.mse_loss(yaw_predicted, label_yaw_cont_gaze)
                    # q_loss_reg_pitch_local = reg_criterion(pitch_predicted, q_label_pitch_cont_gaze)
                    # q_loss_reg_yaw_local = reg_criterion(yaw_predicted, q_label_yaw_cont_gaze)

                    # # Total loss
                    # q_loss_pitch_gaze_local += q_loss_reg_pitch_local
                    # q_loss_yaw_gaze_local += q_loss_reg_yaw_local
                    # q_loss_local = q_loss_pitch_gaze_local + q_loss_yaw_gaze_local
                qry_accs.append(losses[INNER_STEPS-1].detach())
            # loss_seq = [q_loss_pitch_gaze, q_loss_yaw_gaze]
            # grad_seq = [torch.tensor(1.0).cuda(device) for _ in range(len(loss_seq))]
            # torch.autograd.backward(loss_seq, grad_seq)

            print(sum(qry_accs) / len(qry_accs))
            logger.report_scalar("Query", "Loss", iteration=iters, value=(sum(qry_accs) / len(qry_accs)))
            logger.report_scalar("Query", "MAE", iteration=iters, value=train_mae)
            logger.report_scalar("Params", "LR", iteration=iters, value=meta_scheduler.get_last_lr()[0])
            meta_scheduler.step()

        torch.save(model, os.path.join("/home/janek/software/L2CS-Net/models/l2cs_maml_gc_portrait_mode", f"model_epoch{epoch}.pkl"))
        # test_pitch_loss = 0
        # test_yaw_loss = 0

        # for i, (s_im, s_lc, s_lb, q_im, q_lc, q_lb) in enumerate(meta_test_loader):
        #     print(f"Iter {i+1} / {len(meta_test_loader.dataset)}")
        #     # Prepare data
        #     s_im = torch.squeeze(s_im, 0)
        #     s_im = s_im.to(device)
        #     s_lc = torch.squeeze(s_lc, 0)
        #     s_lc = s_lc.to(device)
        #     s_lb = torch.squeeze(s_lb, 0)
        #     s_lb = s_lb.to(device)
        #     q_im = torch.squeeze(q_im, 0)
        #     q_im = q_im.to(device)
        #     q_lc = torch.squeeze(q_lc, 0)
        #     q_lc = q_lc.to(device)
        #     q_lb = torch.squeeze(q_lb, 0)
        #     q_lb = q_lb.to(device)

        #     s_label_pitch_gaze = Variable(s_lb[:, 0]).cuda(device)
        #     s_label_yaw_gaze = Variable(s_lb[:, 1]).cuda(device)
        #     s_label_pitch_cont_gaze = Variable(s_lc[:, 0]).cuda(device)
        #     s_label_yaw_cont_gaze = Variable(s_lc[:, 1]).cuda(device)

        #     q_label_pitch_gaze = Variable(q_lb[:, 0]).cuda(device)
        #     q_label_yaw_gaze = Variable(q_lb[:, 1]).cuda(device)
        #     q_label_pitch_cont_gaze = Variable(q_lc[:, 0]).cuda(device)
        #     q_label_yaw_cont_gaze = Variable(q_lc[:, 1]).cuda(device)

        #     # setup model
        #     model.train()
        #     model.zero_grad()

        #     with higher.innerloop_ctx(model, inner_optimizer, copy_initial_weights=False) as (fnet, diffopt):
        #         for _ in range(1):
        #             pitch, yaw = fnet(s_im)

        #             s_loss_pitch_gaze = criterion(pitch, s_label_pitch_gaze.to(torch.long))
        #             s_loss_yaw_gaze = criterion(yaw, s_label_pitch_gaze.to(torch.long))
        #             pitch_predicted = softmax(pitch)
        #             yaw_predicted = softmax(yaw)
        #             pitch_predicted = \
        #             torch.sum(pitch_predicted * idx_tensor, 1) * 3 - 42
        #             yaw_predicted = \
        #             torch.sum(yaw_predicted * idx_tensor, 1) * 3 - 42

        #             # loss_reg_pitch = F.mse_loss(pitch_predicted, label_pitch_cont_gaze)
        #             # loss_reg_yaw = F.mse_loss(yaw_predicted, label_yaw_cont_gaze)
        #             s_loss_reg_pitch = reg_criterion(pitch_predicted, s_label_pitch_cont_gaze)
        #             s_loss_reg_yaw = reg_criterion(yaw_predicted, s_label_yaw_cont_gaze)

        #             # Total loss
        #             s_loss_pitch_gaze += s_loss_reg_pitch
        #             s_loss_yaw_gaze += s_loss_reg_yaw

        #             # loss_seq = [s_loss_pitch_gaze, s_loss_yaw_gaze]
        #             # grad_seq = [torch.tensor(1.0).cuda(device) for _ in range(len(loss_seq))]
        #             loss_seq = s_loss_pitch_gaze + s_loss_yaw_gaze
        #             # torch.autograd.backward(loss_seq, grad_seq)
        #             diffopt.step(loss_seq)

        #         # setup model for evaluation
        #         fnet.eval()
        #         pitch, yaw = fnet(q_im)
        #         # loss_pitch_gaze = F.cross_entropy(pitch, label_pitch_gaze)
        #         # loss_yaw_gaze = F.cross_entropy(yaw, label_yaw_gaze)
        #         q_loss_pitch_gaze = criterion(pitch, q_label_pitch_gaze.to(torch.long))
        #         q_loss_yaw_gaze = criterion(yaw, q_label_yaw_gaze.to(torch.long))

        #         pitch_predicted = softmax(pitch)
        #         yaw_predicted = softmax(yaw)

        #         pitch_predicted = \
        #         torch.sum(pitch_predicted * idx_tensor, 1) * 3 - 42
        #         yaw_predicted = \
        #         torch.sum(yaw_predicted * idx_tensor, 1) * 3 - 42

        #         # loss_reg_pitch = F.mse_loss(pitch_predicted, label_pitch_cont_gaze)
        #         # loss_reg_yaw = F.mse_loss(yaw_predicted, label_yaw_cont_gaze)
        #         q_loss_reg_pitch = reg_criterion(pitch_predicted, q_label_pitch_cont_gaze)
        #         q_loss_reg_yaw = reg_criterion(yaw_predicted, q_label_yaw_cont_gaze)

        #         # Total loss
        #         # q_loss_pitch_gaze += q_loss_reg_pitch
        #         # q_loss_yaw_gaze += q_loss_reg_yaw
        #         test_pitch_loss += q_loss_reg_pitch.detach()
        #         test_yaw_loss += q_loss_reg_yaw.detach()

        #     #! No outer optimization
        #     # loss_seq = [q_loss_pitch_gaze, q_loss_yaw_gaze]
        #     # grad_seq = [torch.tensor(1.0).cuda(device) for _ in range(len(loss_seq))]
        #     # torch.autograd.backward(loss_seq, grad_seq)
        #     # meta_optimizer.step()
        # print(test_pitch_loss / len(meta_test_loader.dataset))
        # print(test_yaw_loss / len(meta_test_loader.dataset))
        # logger.report_scalar("MSE", "[VAL] Yaw", iteration=epoch, value=(test_yaw_loss / len(meta_test_loader.dataset)))
        # logger.report_scalar("MSE", "[VAL] Pitch", iteration=epoch, value=(test_pitch_loss / len(meta_test_loader.dataset)))
