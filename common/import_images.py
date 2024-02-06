import random
from glob import glob
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from common.constants import WAIT_KEY

DB_PATH = r'/home/barbara/Documents/pracka2/dataset/UTK_Face/'


def get_dataset_from_path(path):
    return glob(path)


def read_images():
    part10_images = get_dataset_from_path('/home/barbara/Documents/pracka2/dataset/UTK_Face/part1_0_199/*.jpg')
    part11_images = get_dataset_from_path('/home/barbara/Documents/pracka2/dataset/UTK_Face/part1_200_399/*.jpg')
    part12_images = get_dataset_from_path('/home/barbara/Documents/pracka2/dataset/UTK_Face/part1_400_499/*.jpg')
    return [part10_images, part11_images, part12_images]


def read_all_utk_images_part1():
    return get_dataset_from_path('/home/barbara/Documents/pracka2/dataset/UTK_Face/part1/*.jpg')


def read_utk_images_partx(part_number):
    return get_dataset_from_path(f'/home/barbara/Documents/pracka2/dataset/UTK_Face/part{part_number}/*.jpg')


def read_first_200():
    return get_dataset_from_path(DB_PATH+'part1_0_199/*.jpg')


def display_examples(dataset, number):
    size_of_dataset = len(dataset)
    for i in range(number):
        rand = random.randrange(size_of_dataset)
        cv2.imshow("title", dataset[rand])
        cv2.waitKey(WAIT_KEY)


def get_age_labels(dataset):
    labels = []
    for img in dataset:
        labels.append(get_age_label(img))
    return np.array(labels)


def get_gender_labels(dataset):
    labels = []
    for img in dataset:
        labels.append(get_sex_label(img))
    return np.array(labels)


def get_age_label(name):
    return int(name.split("_")[-4].split("/")[-1])


def get_sex_label(name):
    return int(name.split("_")[-3])


def get_race_label(name):
    return int(name.split("_")[-2])


def split_data(images, labels, test_size):
    train_img, test_img, train_lbl, test_lbl = train_test_split(images, labels, test_size=test_size, shuffle=True)
    return np.asarray(train_img), np.asarray(test_img), train_lbl, test_lbl

