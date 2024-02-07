import datetime

import face_recognition
import numpy as np

from common.constants import IMAGE_CROP_MARGIN_RATE
from common.import_images import read_first_200, get_race_label
from common.save_results import save_images, save_image_hardcoded
from model.distances.facial_features import find_facial_features
from model.distances.proportions import get_all_proportions, get_all_distances, get_all_sizes


def get_cropped_face(image_name, **kwargs):
    print_image_location = kwargs.get('print_image_location', False)
    show_cropped_image = kwargs.get('show_cropped_image', False)

    image = face_recognition.load_image_file(image_name)

    face_locations = face_recognition.face_locations(image, number_of_times_to_upsample=0, model="cnn")

    for face_location in face_locations:
        top, right, bottom, left = face_location

        if print_image_location:
            print(
                "A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom,
                                                                                                      right))

        vert_mar = round((bottom - top) * IMAGE_CROP_MARGIN_RATE)
        hor_mar = round((bottom - top) * IMAGE_CROP_MARGIN_RATE)

        face_image = image[max(0, top - vert_mar):min(bottom + vert_mar, image.shape[0]),
                     max(0, left - hor_mar):min(right + hor_mar, image.shape[1])]

        # if show_cropped_image:
        #     pil_image = Image.fromarray(face_image)
        #     pil_image.show()

        if image_name is not None and face_image is not None:
            # save_image_hardcoded(image_name, face_image)
            return (image_name, face_image)
        else:
            return None


def crop_first_200():
    images = read_first_200()
    cropped_list = list(map(get_cropped_face, images))
    return filter_none_objects(cropped_list)


def crop_first_5():
    images = read_first_200()
    images = images[:5]
    cropped_list = list(map(get_cropped_face, images))
    return filter_none_objects(cropped_list)


def crop_and_save_images(images):
    not_found = 0
    for img in images:
        cropped = get_cropped_face(img)
        if cropped is None:
            not_found += 1
        else:
            name, cropped_img = cropped
            if name is not None and cropped_img is not None:
                save_image_hardcoded(name, cropped_img)

    print(f'face not found in {not_found} of {len(images)} images')
    return


def get_and_save_face_features(image_names, input_type):
    return get_features_for_images(image_names, input_type)


def get_and_save_face_proportions(image_names):
    return get_features_for_images(image_names, 'proportions')


def get_and_save_face_proportions_extended(image_names, ):
    return get_features_for_images(image_names, 'proportions_extended')


def get_features_for_images(image_names, feature_type):
    # image_names = image_names[4:5]

    no_landmarks = 0
    proportions_list = []

    for img_n in image_names:

        img = face_recognition.load_image_file(img_n)
        landmarks = find_facial_features(img)
        if len(landmarks) == 0:
            no_landmarks += 1
        else:
            if feature_type == 'proportions':
                result = {
                    "filename": img_n,
                    "proportions": get_all_proportions(landmarks)
                }
            elif feature_type == 'proportions_extended':
                proportions = get_all_proportions(landmarks)
                distances = get_all_distances(landmarks)
                result = {
                    "filename": img_n,
                    "proportions": proportions + distances
                }
            elif feature_type == 'proportions_sizes':
                proportions = get_all_proportions(landmarks)
                distances = get_all_distances(landmarks)
                sizes = get_all_sizes(landmarks)
                race = get_race_label(img_n),
                result = {
                    "filename": img_n,
                    "proportions": proportions + distances + sizes + race
                }
            proportions_list.append(result)

    # save_proportions(proportions_list, feature_type)
    # print(proportions_list)
    print(f'Did not find landmarks in {no_landmarks} of {len(image_names)} photos')
    return proportions_list


def get_and_save_face_encodings(image_names):
    no_encodings = 0
    encodings_list = []

    for img_n in image_names:

        img = face_recognition.load_image_file(img_n)
        encodings = face_recognition.face_encodings(img)

        if len(encodings) == 0:
            no_encodings += 1
        else:
            result = {
                "filename": img_n,
                "encodings": encodings
            }
            encodings_list.append(result)

    save_encodings(encodings_list)
    print(encodings_list)
    print(f'Did not find encodings in {no_encodings}')
    return encodings_list


def save_face_proportions_from_original_image(images):
    no_face_found = 0
    no_landmarks = 0
    proportions_list = []
    for img in images:
        cropped = get_cropped_face(img)
        if cropped is None:
            no_face_found += 1
        else:
            cropped_name, cropped_img = cropped
            landmarks = find_facial_features(cropped_img)

            if len(landmarks) == 0:
                no_landmarks += 1
            else:
                result = {
                    "filename": cropped_name,
                    "proportions": get_all_proportions(landmarks)
                }
                proportions_list.append(result)

    print(proportions_list)
    save_proportions(proportions_list, '')

    print(f'Did not find faces in {no_face_found} out of {len(images)} images')
    print(f'Did not find landmarks in {no_landmarks}')
    return proportions_list


def save_proportions(proportions_list, feature_type):
    arr = np.array(proportions_list)
    np.save(f'face_{feature_type}_{datetime.datetime.now()}.npy', arr)


def save_encodings(proportions_list):
    arr = np.array(proportions_list)
    np.save(f'face_encodings_{datetime.datetime.now()}.npy', arr)


def read_proportions(filename):
    return np.load(filename, allow_pickle=True)


def crop_images(images):
    cropped_list = list(map(get_cropped_face, images))
    cropped_filtered_list = filter_none_objects(cropped_list)
    save_images(cropped_filtered_list, 'part1_cropped')
    return cropped_filtered_list


def filter_none_objects(images_tuples):
    images_tuples = list(filter(lambda t: t is not None, images_tuples))
    images_tuples = list(filter(lambda t: t[0] is not None or t[1] is not None, images_tuples))
    return images_tuples
