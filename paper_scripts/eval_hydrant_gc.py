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
import tqdm

if __name__ == '__main__':
    CALIBRATION_SET_SIZE = 30
    INNER_LOOP_LR = 1e-5
    INNER_STEPS = 20

    BATCH1 = False


    device = torch.device('cuda')

    transforms = t.Compose([
        t.Resize(224),
        t.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225] # RGB
        )
    ])

    reg_criterion = nn.L1Loss().cuda(device)


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


    for model_id in range(29, 30):
        losses_pitch = []
        losses_yaw = []
        for subject in tqdm.tqdm(subjects):
            images = sorted(os.listdir(os.path.join(dataset_base_path, subject)))
            if len(images) < 140:
                continue

            calibration_samples = images[:CALIBRATION_SET_SIZE]
            test_samples = images[CALIBRATION_SET_SIZE:CALIBRATION_SET_SIZE+100]

            support_cont = []
            support_images = []
            for s in calibration_samples:
                # label
                row_name = os.path.join(subject, s)
                yaw, pitch = annotations.loc[row_name].values
                # Convert yaw and pitch to angles
                pitch = pitch * 180 / np.pi
                yaw = yaw * 180 / np.pi

                cont_labels = [pitch, yaw]

                # image
                img_path = os.path.join(dataset_base_path, row_name)
                support_images.append(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB).astype(np.float32)/255.0)
                support_cont.append(cont_labels)


            query_cont = []
            query_images = []
            for q in test_samples:
                # label
                row_name = os.path.join(subject, q)
                yaw, pitch = annotations.loc[row_name].values
                # Convert yaw and pitch to angles
                pitch = pitch * 180 / np.pi
                yaw = yaw * 180 / np.pi

                cont_labels = [pitch, yaw]

                query_cont.append(cont_labels)
                # image
                img_path = os.path.join(dataset_base_path, row_name)
                query_images.append(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB).astype(np.float32)/255.0)


            support_cont = np.array(support_cont)
            support_cont = torch.FloatTensor(support_cont)
            query_cont = np.array(query_cont)
            query_cont = torch.FloatTensor(query_cont)

            support_images = np.array(support_images)
            support_images = torch.from_numpy(support_images).type(torch.FloatTensor)
            support_images = torch.permute(support_images, (0, 3, 1, 2))
            query_images = np.array(query_images)
            query_images = torch.from_numpy(query_images).type(torch.FloatTensor)
            query_images = torch.permute(query_images, (0, 3, 1, 2))

            support_images = transforms(support_images)
            query_images = transforms(query_images)


            #! Hydrant (ours)
            model = Hydrant("regnet")
            state_dict = torch.load(f"/home/janek/software/L2CS-Net/models/hydrant_regnet/model_epoch{str(model_id)}_state_dict.pkl")
            model.load_state_dict(state_dict)
            model.train()
            model.to(device)


            optimizer = torch.optim.SGD(model.parameters(), INNER_LOOP_LR)

            s_im = torch.squeeze(support_images, 0)
            s_im = s_im.to(device)
            s_lc = torch.squeeze(support_cont, 0)
            s_lc = s_lc.to(device)
            q_im = torch.squeeze(query_images, 0)
            q_im = q_im.to(device)
            q_lc = torch.squeeze(query_cont, 0)
            q_lc = q_lc.to(device)

            s_label_pitch_cont_gaze = Variable(s_lc[:, 0]).cuda(device)
            s_label_yaw_cont_gaze = Variable(s_lc[:, 1]).cuda(device)

            q_label_pitch_cont_gaze = Variable(q_lc[:, 0]).cuda(device)
            q_label_yaw_cont_gaze = Variable(q_lc[:, 1]).cuda(device)
            for step_nr in range(INNER_STEPS):
                optimizer.zero_grad()
                pitch_s, yaw_s = model(s_im)


                s_loss_reg_pitch = reg_criterion(pitch_s.ravel(), s_label_pitch_cont_gaze)
                s_loss_reg_yaw = reg_criterion(yaw_s.ravel(), s_label_yaw_cont_gaze)

                # Total loss
                loss_seq = s_loss_reg_pitch + s_loss_reg_yaw
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
                        pitch_q, yaw_q = model(q_image)

                        q_loss_reg_pitch = reg_criterion(pitch_q.ravel(), q_label_pitch_cont.unsqueeze(0))
                        q_loss_reg_yaw = reg_criterion(yaw_q.ravel(), q_label_yaw_cont.unsqueeze(0))
                        losses_pitch.append(q_loss_reg_pitch.detach())
                        losses_yaw.append(q_loss_reg_yaw.detach())
            else:
                pitch_q, yaw_q = model(q_im)

                q_loss_reg_pitch = reg_criterion(pitch_q.ravel(), q_label_pitch_cont_gaze)
                q_loss_reg_yaw = reg_criterion(yaw_q.ravel(), q_label_yaw_cont_gaze)
                losses_pitch.append(q_loss_reg_pitch.detach())
                losses_yaw.append(q_loss_reg_yaw.detach())

            del model

        print("### EVALUATION ###")
        print("Yaw: ", sum(losses_yaw)/len(losses_yaw))
        print("Pitch: ", sum(losses_pitch)/len(losses_pitch))
        pitch_steps.append(sum(losses_pitch)/len(losses_pitch))
        yaw_steps.append(sum(losses_yaw)/len(losses_yaw))

    print("Finish")