import os
import shutil


if __name__ == "__main__":
    dirs = os.listdir("/home/janek/GazeCaptureNormalizedTestAllModes")
    dirs.remove("annotations.txt")
    print(len(dirs))
    for dir in dirs:
        if len(os.listdir(os.path.join("/home/janek/GazeCaptureNormalizedTestAllModes", dir))) < 520:
            print(dir)
            shutil.rmtree(os.path.join("/home/janek/GazeCaptureNormalizedTestAllModes", dir))