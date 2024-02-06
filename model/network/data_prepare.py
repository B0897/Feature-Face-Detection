import numpy as np
from sklearn.model_selection import train_test_split
from common.constants import NUMBER_OF_CLASSES
from sklearn.preprocessing import MinMaxScaler


def split_data(images, labels, test_size):
    train_img, test_img, train_lbl, test_lbl = train_test_split(images, labels, test_size=test_size, shuffle=True)
    return np.asarray(train_img), np.asarray(test_img), train_lbl, test_lbl


def group_age(labels):
    if NUMBER_OF_CLASSES == 2:
        return group_age2(labels)
    elif NUMBER_OF_CLASSES == 3:
        return group_age3(labels)
    elif NUMBER_OF_CLASSES == 4:
        return group_age4(labels)
    elif NUMBER_OF_CLASSES == 8:
        return group_age8(labels)


def group_age_equally(labels):
    if NUMBER_OF_CLASSES == 2:
        return group_age22(labels)
    elif NUMBER_OF_CLASSES == 3:
        return group_age33(labels)
    elif NUMBER_OF_CLASSES == 4:
        return group_age44(labels)
    elif NUMBER_OF_CLASSES == 8:
        return group_age88(labels)


def group_adults(age_labels, sex_labels):
    class_labels = []

    print('input lables')
    print(age_labels)
    print(sex_labels)

    if NUMBER_OF_CLASSES == 3:
        for i in range(len(age_labels)):
            age = age_labels[i]
            sex = sex_labels[i]
            if age < 16:
                class_labels.append(0)
            elif 16 <= age:
                if sex == 0:
                    class_labels.append(1)
                else:
                    class_labels.append(2)
        return class_labels
    if NUMBER_OF_CLASSES == 5:
        for i in range(len(age_labels)):
            age = age_labels[i]
            sex = sex_labels[i]
            if age < 16:
                class_labels.append(0)
            elif 16 <= age < 35:
                if sex == 0:
                    class_labels.append(1)
                else:
                    class_labels.append(2)
            elif 35 <= age:
                if sex == 0:
                    class_labels.append(3)
                else:
                    class_labels.append(4)
        return class_labels


# [0–19, 20+].
def group_age2(labels):
    class_labels = []
    for age in labels:
        if age < 20:
            class_labels.append(0)
        elif 20 <= age:
            class_labels.append(1)
    return class_labels


# [0–3, 4-24, 25+].
def group_age3(labels):
    class_labels = []
    for age in labels:
        if age < 4:
            class_labels.append(0)
        elif 4 <= age < 25:
            class_labels.append(1)
        elif 25 <= age:
            class_labels.append(2)
    return class_labels


# [0–14, 15–29, 30–44, 45+].
def group_age4(labels):
    class_labels = []
    for age in labels:
        if age < 15:
            class_labels.append(0)
        elif 15 <= age < 30:
            class_labels.append(1)
        elif 30 <= age < 45:
            class_labels.append(2)
        elif 45 <= age:
            class_labels.append(3)
    return class_labels


# [0–3, 4–7, 8–11, 12-17, 18-24, 25-39, 40-59, 60+]
def group_age8(labels):
    class_labels = []
    for age in labels:
        if age < 4:
            class_labels.append(0)
        elif 4 <= age < 8:
            class_labels.append(1)
        elif 8 <= age < 12:
            class_labels.append(2)
        elif 12 <= age < 18:
            class_labels.append(3)
        elif 18 <= age < 25:
            class_labels.append(4)
        elif 25 <= age < 40:
            class_labels.append(5)
        elif 40 <= age < 60:
            class_labels.append(6)
        elif 60 <= age:
            class_labels.append(7)
    return class_labels


# [0–19, 20+].
def group_age22(labels):
    class_labels = []
    for age in labels:
        if age < 29:
            class_labels.append(0)
        elif 29 <= age:
            class_labels.append(1)
    return class_labels


# [0–3, 4-24, 25+].
def group_age33(labels):
    class_labels = []
    for age in labels:
        if age < 26:
            class_labels.append(0)
        elif 26 <= age < 37:
            class_labels.append(1)
        elif 37 <= age:
            class_labels.append(2)
    return class_labels


# [0–14, 15–29, 30–44, 45+].
def group_age44(labels):
    class_labels = []
    for age in labels:
        if age < 23:
            class_labels.append(0)
        elif 23 <= age < 29:
            class_labels.append(1)
        elif 29 <= age < 45:
            class_labels.append(2)
        elif 45 <= age:
            class_labels.append(3)
    return class_labels


# [0–3, 4–7, 8–11, 12-17, 18-24, 25-39, 40-59, 60+]
def group_age88(labels):
    class_labels = []
    for age in labels:
        if age < 8:
            class_labels.append(0)
        elif 8 <= age < 23:
            class_labels.append(1)
        elif 23 <= age < 26:
            class_labels.append(2)
        elif 26 <= age < 29:
            class_labels.append(3)
        elif 29 <= age < 35:
            class_labels.append(4)
        elif 35 <= age < 45:
            class_labels.append(5)
        elif 45 <= age < 58:
            class_labels.append(6)
        elif 58 <= age:
            class_labels.append(7)
    return class_labels


def numbers_to_input_vectors(inputs):
    outputs = []
    for num in inputs:
        if num < 0 or num >= NUMBER_OF_CLASSES:
            raise ValueError("Input number is out of range for one-hot encoding.")
        input_vector = [0] * NUMBER_OF_CLASSES
        input_vector[num] = 1
        outputs.append(input_vector)
    return outputs


def normalize_list_of_vectors(list_of_tuples):
    transposed_tuples = list(zip(*list_of_tuples))
    normalized_tuples = []

    for dimension_values in transposed_tuples:
        scaler = MinMaxScaler()
        normalized_values = scaler.fit_transform([[x] for x in dimension_values]).flatten()
        normalized_tuples.append(tuple(normalized_values))

    result = list(zip(*normalized_tuples))
    return result

