import pickle
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

from svgpathtools import svg2paths
from svgpath2mpl import parse_path

carrot_path, attributes = svg2paths('/home/janek/Downloads/Screenshot-from-2023-01-05-12-35-20.svg')
carrot_marker = parse_path(attributes[0]['d'])
carrot_marker.vertices -= carrot_marker.vertices.mean(axis=0)

# carrot_marker = carrot_marker.transformed(mpl.transforms.Affine2D().rotate_deg(125))

if __name__ == "__main__":
    with open("/home/janek/software/L2CS-Net/pitch_steps_l2cs.pkl", 'rb') as f:
        pitch_steps_l2cs = pickle.load(f)
    with open("/home/janek/software/L2CS-Net/pitch_support_l2cs.pkl", 'rb') as f:
        pitch_support_l2cs = pickle.load(f)

    with open("/home/janek/software/L2CS-Net/yaw_steps_l2cs.pkl", 'rb') as f:
        yaw_steps_l2cs = pickle.load(f)

    with open("/home/janek/software/L2CS-Net/yaw_support_l2cs.pkl", 'rb') as f:
        yaw_support_l2cs = pickle.load(f)

    with open("/home/janek/software/L2CS-Net/pitch_steps_meta.pkl", 'rb') as f:
        pitch_steps_meta = pickle.load(f)

    with open("/home/janek/software/L2CS-Net/pitch_support_meta.pkl", 'rb') as f:
        pitch_support_meta = pickle.load(f)

    with open("/home/janek/software/L2CS-Net/yaw_steps_meta.pkl", 'rb') as f:
        yaw_steps_meta = pickle.load(f)

    with open("/home/janek/software/L2CS-Net/yaw_support_meta.pkl", 'rb') as f:
        yaw_support_meta = pickle.load(f)

    with open("/home/janek/software/L2CS-Net/pitch_steps_meta_gc.pkl", 'rb') as f:
        pitch_steps_meta_gc = pickle.load(f)

    with open("/home/janek/software/L2CS-Net/yaw_steps_meta_gc.pkl", 'rb') as f:
        yaw_steps_meta_gc = pickle.load(f)

    with open("/home/janek/software/L2CS-Net/pitch_steps_legacy_gc.pkl", 'rb') as f:
        pitch_steps_l2cs_gc = pickle.load(f)

    with open("/home/janek/software/L2CS-Net/yaw_steps_legacy_gc.pkl", 'rb') as f:
        yaw_steps_l2cs_gc = pickle.load(f)

    with open("/home/janek/software/L2CS-Net/pitch_support_legacy_gc.pkl", 'rb') as f:
        pitch_support_l2cs_gc = pickle.load(f)

    with open("/home/janek/software/L2CS-Net/yaw_support_legacy_gc.pkl", 'rb') as f:
        yaw_support_l2cs_gc = pickle.load(f)

    with open("/home/janek/software/L2CS-Net/pitch_support_meta_gc.pkl", 'rb') as f:
        pitch_support_meta_gc = pickle.load(f)

    with open("/home/janek/software/L2CS-Net/yaw_support_meta_gc.pkl", 'rb') as f:
        yaw_support_meta_gc = pickle.load(f)

    with open("/home/janek/software/L2CS-Net/pitch_hfc_l2cs.pkl", 'rb') as f:
        pitch_hfc_l2cs = pickle.load(f)

    with open("/home/janek/software/L2CS-Net/pitch_hfc_part2_l2cs.pkl", 'rb') as f:
        pitch_hfc_part2_l2cs = pickle.load(f)

    with open("/home/janek/software/L2CS-Net/yaw_hfc_l2cs.pkl", 'rb') as f:
        yaw_hfc_l2cs = pickle.load(f)

    with open("/home/janek/software/L2CS-Net/yaw_hfc_part2_l2cs.pkl", 'rb') as f:
        yaw_hfc_part2_l2cs = pickle.load(f)

    pitch_hfc_l2cs.extend(pitch_hfc_part2_l2cs)
    yaw_hfc_l2cs.extend(yaw_hfc_part2_l2cs)

    steps_gc = [2, 3, 5, 10, 15, 20, 30]
    steps = [2, 3, 5, 10, 15, 20]
    kernel_sizes = [2, 3, 5, 10, 15, 20, 30, 40, 50, 70, 90, 120, 150, 180]

    plt.plot(kernel_sizes, pitch_hfc_l2cs, '--o', label='Pitch')
    plt.plot(kernel_sizes, yaw_hfc_l2cs, '--o', label='Yaw')
    plt.legend()
    plt.title("Validation Error vs Lowpass Cutoff Frequency")
    plt.xlabel("Lowpass Cutoff Frequency")
    plt.ylabel("MAE [°]")
    plt.show()


    plt.plot(steps_gc, pitch_support_meta_gc, '--o', label='Pitch MAML')
    plt.plot(steps_gc, yaw_support_meta_gc, '--o', label='Yaw MAML')
    plt.plot(steps_gc, pitch_support_l2cs_gc, '--o', label='Pitch Legacy')
    plt.plot(steps_gc, yaw_support_l2cs_gc, '--o', label='Yaw Legacy')
    plt.legend()
    plt.title("Validation Error vs Support Set Size: Gaze Capture")
    plt.xlabel("Support Set Size")
    plt.ylabel("MAE [°]")
    plt.show()

    plt.plot(pitch_steps_meta_gc, '--o', label='Pitch MAML')
    plt.plot(yaw_steps_meta_gc, '--o', label='Yaw MAML')
    plt.plot(pitch_steps_l2cs_gc, '--o', label='Pitch Legacy')
    plt.plot(yaw_steps_l2cs_gc, '--o', label='Yaw Legacy')
    plt.legend()
    plt.title("Validation Error vs Number of Gradient Steps: Gaze Capture")
    plt.xlabel("Number of Gradient Steps")
    plt.ylabel("MAE [°]")
    plt.show()

    plt.plot(pitch_steps_meta, '--o', label='Pitch')
    plt.plot(yaw_steps_meta, '--o', label='Yaw')
    plt.legend()
    plt.title("Validation Error vs Number of Gradient Steps")
    plt.xlabel("Number of Gradient Steps")
    plt.ylabel("MAE [°]")
    plt.show()

    plt.plot(steps, pitch_support_meta, '--o', label='Pitch')
    plt.plot(steps, yaw_support_meta, '--o', label='Yaw')
    plt.legend()
    plt.title("Validation Error vs Support Set Size")
    plt.xlabel("Support Set Size")
    plt.ylabel("MAE [°]")
    plt.show()

    plt.plot(pitch_steps_l2cs, '--o', label='L2CS-Net')
    plt.plot(pitch_steps_meta, '--o', label='MAML')
    plt.legend()
    plt.title("Gradient steps impact on Pitch, L2CS-Net vs MAML")
    plt.ylabel("MAE [°]")
    plt.xlabel("Number of gradient steps")
    plt.show()

    plt.plot(yaw_steps_l2cs, '--o', label='L2CS-Net')
    plt.plot(yaw_steps_meta, '--o', label='MAML')
    plt.legend()
    plt.title("Gradient steps impact on Yaw, L2CS-Net vs MAML")
    plt.ylabel("MAE [°]")
    plt.xlabel("Number of gradient steps")
    plt.show()

    plt.plot(steps, pitch_support_l2cs, '--o', label='L2CS-Net')
    plt.plot(steps, pitch_support_meta, '--o', label='MAML')
    plt.legend()
    plt.title("Calibration set size impact on Pitch, L2CS-Net vs MAML")
    plt.ylabel("MAE [°]")
    plt.xlabel("Calibration set size")
    plt.show()

    plt.plot(steps, yaw_support_l2cs, '--o', label='L2CS-Net')
    plt.plot(steps, yaw_support_meta, '--o', label='MAML')
    plt.legend()
    plt.title("Calibration set size impact on Yaw, L2CS-Net vs MAML")
    plt.ylabel("MAE [°]")
    plt.xlabel("Calibration set size")
    plt.show()

    l2cs = [1.3151, 1.9901]
    l2cs_gradient_steps = [1.3107, 1.9812]
    meta = [1.1295, 1.736]
    labels = ["Yaw", "Pitch"]
    x_axis = np.arange(len(labels))

    plt.bar(x_axis-0.1, l2cs, 0.1, label='Legacy')
    plt.bar(x_axis, l2cs_gradient_steps, 0.1, label='Legacy with Gradient Steps')
    plt.bar(x_axis+0.1, meta, 0.1, label='Meta-Learning')

    plt.xticks(x_axis, labels)
    plt.xlabel("Angles")
    plt.ylabel("MAE [°]")
    plt.title("Cross-Dataset Validation: Gaze Capture")
    plt.legend()
    plt.show()
