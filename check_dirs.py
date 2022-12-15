import os


if __name__ == "__main__":
    dirs = os.listdir("/home/jan/meta_dataset_normalized")
    dirs.remove("annotations.txt")
    for dir in dirs:
        if len(os.listdir(os.path.join("/home/jan/meta_dataset_normalized", dir))) <= 40:
            print(dir)