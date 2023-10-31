import os
import random
import shutil
import sys


def create_directories(dataset_home: str) -> None:
    # Split the training data into directory structure optimized
    # for Keras.
    print("Creating directories...")
    subdirs = ["train/", "test/"]
    for subdir in subdirs:
        labeldirs = ["dogs/", "cats/"]
        for labeldir in labeldirs:
            newdir = dataset_home + subdir + labeldir
            os.makedirs(newdir, exist_ok=True)


def copy_images(dataset_home: str) -> None:
    # Copy images to the new training directory.
    # Use a random seed to withhold 25% of the pictures into the test
    # directory. Use the same random seed to always choose the test
    # images consistently.
    print("Copying images...")
    random.seed(1)
    test_ratio = 0.25
    src_directory = "dogs-vs-cats/train/"
    for file in os.listdir(src_directory):
        src_image = src_directory + "/" + file
        dst_dir = "test/" if random.random() < test_ratio else "train/"
        if file.startswith("cat"):
            dst_image = dataset_home + dst_dir + "cats/" + file
            shutil.copyfile(src_image, dst_image)
        elif file.startswith("dog"):
            dst_image = dataset_home + dst_dir + "dogs/" + file
            shutil.copyfile(src_image, dst_image)

    total_initial_images = len(os.listdir(src_directory))
    cat_train_count = len(os.listdir(dataset_home + "train/cats/"))
    cat_test_count = len(os.listdir(dataset_home + "test/cats/"))
    dog_train_count = len(os.listdir(dataset_home + "train/dogs/"))
    dog_test_count = len(os.listdir(dataset_home + "test/dogs/"))
    print(f"{total_initial_images} total images")
    print(f"{cat_train_count} cat train images")
    print(f"{cat_test_count} cat test images")
    print(f"{dog_train_count} dog train images")
    print(f"{dog_test_count} dog test images")


def main() -> int:
    print("Preparing data...")
    dataset_home = "dataset_dogs_vs_cats/"
    create_directories(dataset_home)
    copy_images(dataset_home)
    return 0


if __name__ == "__main__":
    sys.exit(main())
