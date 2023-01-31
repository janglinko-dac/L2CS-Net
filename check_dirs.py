import os


if __name__ == "__main__":
    dirs = os.listdir("/home/janek/software/L2CS-Net/meta_dataset_normalized")
    dirs.remove("annotations.txt")
    for dir in dirs:
        if len(os.listdir(os.path.join("/home/janek/software/L2CS-Net/meta_dataset_normalized", dir))) < 50:
            print(dir)