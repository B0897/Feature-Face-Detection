import matplotlib.pyplot as plt

from common.constants import NUMBER_OF_CLASSES
from common.import_images import get_age_labels, get_gender_labels, DB_PATH, get_dataset_from_path


def describe_distribution(train_labels, test_labels):
    print('--------DATA DISTRIBUTION----------')
    print('TRAIN DATA')
    for group in range(NUMBER_OF_CLASSES):
        filtered = list(filter(lambda age: age == group, train_labels))
        print(f'class {group}: {len(filtered)} images ')

    print('------------------')

    print('TEST DATA')
    for group in range(NUMBER_OF_CLASSES):
        filtered = list(filter(lambda age: age == group, test_labels))
        print(f'class {group}: {len(filtered)} images ')


def get_dataset_statistics():
    part1 = get_dataset_from_path(DB_PATH+'part1/*.jpg')
    part2 = get_dataset_from_path(DB_PATH+'part2/*.jpg')
    part3 = get_dataset_from_path(DB_PATH+'part3/*.jpg')
    images = part1 + part2 + part3

    age_labels = get_age_labels(images)
    sex_labels = get_gender_labels(images)
    print(f'Total number of images = {len(images)}')

    plt.hist(age_labels, bins='auto')
    plt.title("Histogram wieku")
    plt.show()

    plt.hist(sex_labels, bins=([-0.5, 0.5, 1.5]))
    plt.title("Hsitogram płci")
    plt.show()

def create_age_groups():
    part1 = get_dataset_from_path(DB_PATH+'part1/*.jpg')
    part2 = get_dataset_from_path(DB_PATH+'part2/*.jpg')
    part3 = get_dataset_from_path(DB_PATH+'part3/*.jpg')
    images = part1 + part2 + part3

    age_labels = get_age_labels(images)
    sorted_ages = sorted(age_labels)

    examples_per_range = len(age_labels) // NUMBER_OF_CLASSES

    age_ranges = []
    start_index = 0

    for i in range(NUMBER_OF_CLASSES - 1):
        end_index = start_index + examples_per_range
        age_ranges.append((sorted_ages[start_index], sorted_ages[end_index - 1]))
        start_index = end_index

    age_ranges.append((sorted_ages[start_index], sorted_ages[-1]))
    return age_ranges


age_ranges = create_age_groups()
print(age_ranges)

# get_dataset_statistics()
