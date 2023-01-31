import cv2
import numpy as np
import os
import pandas as pd
import torchvision
import math
import matplotlib.pyplot as plt


if __name__ == "__main__":

    dataset_base_path = "/home/janek/gaze_capture_normalized"
    annotations_path = os.path.join(dataset_base_path, "annotations.txt")
    annotations = pd.read_csv(annotations_path, header=None,
                              delimiter=" ", index_col=0,
                              names=['yaw', 'pitch'])

    subjects = [s for s in sorted(os.listdir(dataset_base_path)) if not s.endswith(".txt")]


    pitch_steps = []
    yaw_steps = []
    kernel_sizes = [2, 3, 5, 10, 15, 20, 30, 40, 50, 70, 90, 120, 150, 180]

    for kernel_size in kernel_sizes:
        losses_pitch = []
        losses_yaw = []
        for subject in subjects:
            images = sorted(os.listdir(os.path.join(dataset_base_path, subject)))
            if len(images) < 140:
                continue

            test_samples = images[30:30+100]

            support_cont = []
            support_binned = []
            support_images = []


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
                img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
                #! Do FFT filtration
                dft = np.fft.fft2(img, axes=(0, 1))
                dft_shift = np.fft.fftshift(dft)
                mask = np.zeros_like(img)
                cy = mask.shape[0] // 2
                cx = mask.shape[1] // 2
                cv2.circle(mask, (cx, cy), kernel_size, (255, 255, 255), -1)
                gaussian_radius = kernel_size // 2
                if not gaussian_radius % 2:
                    gaussian_radius += 1
                mask = cv2.GaussianBlur(mask, (gaussian_radius, gaussian_radius), 0)
                # mask = mask / 255
                dft_shift_masked = np.multiply(dft_shift, mask) / 255
                back_ishift = np.fft.ifftshift(dft_shift_masked)
                img_back = np.fft.ifft2(back_ishift, axes=(0, 1))
                img_back = np.abs(img_back).clip(0, 255).astype(np.uint8)

                dft_back = np.fft.fft2(img_back, axes=(0, 1))
                dft_shift_back = np.fft.fftshift(dft_back)

                #! End of FFT filtration
                query_images.append(img_back)

                # create plot
                fig, axs = plt.subplots(2, 3)
                axs[0, 0].imshow(img)
                axs[0, 0].set_title('Image')
                axs[0, 1].imshow(img_back)
                axs[0, 1].set_title('Filtered Image', )
                axs[0, 2].imshow(mask)
                axs[0, 2].set_title('Mask')
                axs[1, 0].imshow(np.mean(20*np.log(np.abs(dft_shift)), axis=2) / np.max(np.mean(20*np.log(np.abs(dft_shift)), axis=2)), cmap='gray')
                axs[1, 0].set_title('Original Image DFT')
                axs[1, 0].title.set_size(8)
                axs[1, 1].imshow(np.mean(20*np.log(np.abs(dft_shift_masked)), axis=2) / np.max(np.mean(20*np.log(np.abs(dft_shift_masked)), axis=2)), cmap='gray')
                axs[1, 1].set_title('Filtered Image DFT')
                axs[1, 1].title.set_size(8)
                axs[1, 2].imshow(np.mean(20*np.log(np.abs(dft_shift_back)), axis=2) / np.max(np.mean(20*np.log(np.abs(dft_shift_back)), axis=2)), cmap='gray')
                axs[1, 2].set_title('DFT After Filtered Image IDFT')
                axs[1, 2].title.set_size(8)
                fig.suptitle(f'Radius:{kernel_size}')
                plt.show()
                break
            break