import numpy as np
import tensorflow as tf

from common.constants import TEST_SIZE, PROGRAM_RESULTS_PATH
from common.import_images import get_age_labels, get_gender_labels, split_data, read_all_utk_images_part1, \
    read_utk_images_partx
from common.save_results import save_images
from model.distances.facial_features import find_facial_features
from model.distances.proportions import get_all_proportions
from model.network.data_prepare import group_age, numbers_to_input_vectors, normalize_list_of_vectors, group_adults
from model.network.model import create_model, train_model, train_binary_tree
from model.preprocessing.crop_face import crop_and_save_images, crop_images, save_face_proportions_from_original_image, \
    get_and_save_face_proportions, read_proportions, get_and_save_face_encodings, \
    get_and_save_face_proportions_extended, get_and_save_face_features
from model.preprocessing.preprocessing import describe_distribution


def crop_and_save_part1():
    # resource_images = read_all_utk_images_part1()
    resource_images = read_utk_images_partx('1_9000_9999')
    # images_tuples = crop_images(resource_images)
    # crop_and_save_images(resource_images[:99])
    crop_and_save_images(resource_images[:500])

    # save_face_proportions_from_original_image(resource_images)

    # save_images(images_tuples, 'part1_cropped')
    # return images_tuples
    return


def save_face_features(input_type):
    resource_images = read_utk_images_partx('3')
    get_and_save_face_features(resource_images, input_type)
    return


def save_part1_face_proportions():
    resource_images = read_utk_images_partx('1_cropped2')
    get_and_save_face_proportions(resource_images[8000:])
    return


def read_face_proportions():
    prop1 = read_proportions(f'{PROGRAM_RESULTS_PATH}face_proportions_2023-08-30 13:38:18.167087.npy')
    prop2 = read_proportions(f'{PROGRAM_RESULTS_PATH}face_proportions_2023-08-30 13:45:25.549048.npy')
    prop3 = read_proportions(f'{PROGRAM_RESULTS_PATH}face_proportions_2023-08-30 13:53:44.762856.npy')
    prop4 = read_proportions(f'{PROGRAM_RESULTS_PATH}face_proportions_2023-08-30 13:59:17.679682.npy')
    prop5 = read_proportions(f'{PROGRAM_RESULTS_PATH}face_proportions_2023-08-30 14:21:25.537905.npy')
    return np.concatenate((prop1, prop2, prop3, prop4, prop5))


def read_face_proportions_extended():
    # part 1
    prop1 = read_proportions(f'{PROGRAM_RESULTS_PATH}face_proportions_extended_2023-10-29 12:15:17.776167.npy')
    prop2 = read_proportions(f'{PROGRAM_RESULTS_PATH}face_proportions_extended_2023-10-29 12:19:18.995124.npy')
    prop3 = read_proportions(f'{PROGRAM_RESULTS_PATH}face_proportions_extended_2023-10-29 12:31:58.671347.npy')
    # part 2
    prop4 = read_proportions(f'{PROGRAM_RESULTS_PATH}face_proportions_extended_2023-11-18 16:35:46.672751.npy')
    # part 3
    prop5 = read_proportions(f'{PROGRAM_RESULTS_PATH}face_proportions_extended_2023-11-20 16:20:32.002725.npy')
    return np.concatenate((prop1, prop2, prop3, prop4, prop5))


def read_face_proportions_sizes():
    part1 = read_proportions(f'{PROGRAM_RESULTS_PATH}face_proportions_sizes_2023-11-26 20:22:45.031237.npy')
    part2 = read_proportions(f'{PROGRAM_RESULTS_PATH}face_proportions_sizes_2023-11-26 21:07:38.772824.npy')
    part3 = read_proportions(f'{PROGRAM_RESULTS_PATH}face_proportions_sizes_2023-11-26 21:11:44.337660.npy')
    return np.concatenate((part1, part2, part3))


def read_face_encodings():
    prop1 = read_proportions(f'{PROGRAM_RESULTS_PATH}face_encodings_2023-10-16 17:16:24.876474.npy')
    prop2 = read_proportions(f'{PROGRAM_RESULTS_PATH}face_encodings_2023-10-16 17:24:13.034898.npy')
    prop3 = read_proportions(f'{PROGRAM_RESULTS_PATH}face_encodings_2023-10-16 17:35:00.556556.npy')
    prop4 = read_proportions(f'{PROGRAM_RESULTS_PATH}face_encodings_2023-10-16 17:42:43.362347.npy')
    return np.concatenate((prop1, prop2, prop3, prop4))


def prepare_network_inputs_outputs():
    resource_images = read_all_utk_images_part1()
    images_tuples = crop_images(resource_images)
    save_images(images_tuples, 'part1_cropped')

    img_names = list(map(lambda img: img[0], images_tuples))
    age_labels = get_age_labels(img_names)
    sex_labels = get_gender_labels(img_names)
    images = list(map(lambda img: img[1], images_tuples))

    net_inputs = []
    net_outputs = []
    for i, img in enumerate(images):
        features = find_facial_features(img)
        proportions = get_all_proportions(features)
        if proportions is None:
            break
        net_inputs.append(proportions)
        net_outputs.append([age_labels[i], sex_labels[i]])
    return net_inputs, net_outputs


def prepare_network_inputs_outputs_from_proportions():
    image_tuples = read_face_proportions()
    return get_net_inputs_outputs_from_tuples(image_tuples)


def prepare_network_inputs_outputs_from_proportions_extended():
    image_tuples = read_face_proportions_extended()
    return get_net_inputs_outputs_from_tuples_extended(image_tuples)


def prepare_network_inputs_outputs_from_proportions_sizes():
    image_tuples = read_face_proportions_sizes()
    return get_net_inputs_outputs_from_tuples_extended(image_tuples)


def prepare_network_inputs_outputs_from_proportions_sizes_children_excluded():
    image_tuples = read_face_proportions_sizes()
    return get_net_inputs_outputs_from_tuples_extended_children_excluded(image_tuples)


def prepare_network_inputs_outputs_from_encodings():
    image_tuples_encodings = read_face_encodings()
    net_inputs2 = list(map(lambda img: np.asarray(img['encodings'][0]), image_tuples_encodings))
    img_names = list(map(lambda img: img['filename'], image_tuples_encodings))
    age_labels = group_age(get_age_labels(img_names))
    sex_labels = get_gender_labels(img_names)

    net_outputs = []
    for i in range(len(age_labels)):
        net_outputs.append([age_labels[i], sex_labels[i]])

    return net_inputs2, net_outputs


def get_net_inputs_outputs_from_tuples(image_tuples):
    net_inputs = list(map(lambda img: img['proportions'], image_tuples))
    img_names = list(map(lambda img: img['filename'], image_tuples))
    age_labels = group_age(get_age_labels(img_names))
    sex_labels = get_gender_labels(img_names)

    net_outputs = []
    for i in range(len(age_labels)):
        net_outputs.append([age_labels[i], sex_labels[i]])

    return net_inputs, net_outputs


def get_net_inputs_outputs_from_tuples_extended(image_tuples):
    net_inputs = list(map(lambda img: img['proportions'][:-1], image_tuples))
    img_names = list(map(lambda img: img['filename'], image_tuples))


    age_labels = group_age(get_age_labels(img_names))
    sex_labels = get_gender_labels(img_names)

    net_inputs = normalize_list_of_vectors(net_inputs)
    net_outputs = []

    for i in range(len(age_labels)):
        net_outputs.append([age_labels[i], sex_labels[i]])

    return net_inputs, net_outputs


def get_net_inputs_outputs_from_tuples_extended_children_excluded(image_tuples):
    net_inputs = list(map(lambda img: img['proportions'][:-1], image_tuples))
    img_names = list(map(lambda img: img['filename'], image_tuples))

    age_labels = get_age_labels(img_names)
    sex_labels = get_gender_labels(img_names)

    net_outputs = group_adults(age_labels, sex_labels)
    net_inputs = normalize_list_of_vectors(net_inputs)

    return net_inputs, net_outputs


def train_net(input_type, optimizer):
    if input_type == 'encodings':
        inputs, labels = prepare_network_inputs_outputs_from_encodings()
    elif input_type == 'proportions':
        inputs, labels = prepare_network_inputs_outputs_from_proportions()
    elif input_type == 'proportions_extended':
        inputs, labels = prepare_network_inputs_outputs_from_proportions_extended()
    elif input_type == 'proportions_sizes':
        inputs, labels = prepare_network_inputs_outputs_from_proportions_sizes()
    elif input_type == 'proportions_sizes_children_excluded':
        inputs, labels = prepare_network_inputs_outputs_from_proportions_sizes_children_excluded()

    train_img, test_img, train_lbl, test_lbl = split_data(inputs, labels, TEST_SIZE)
    print(f'input size: {len(inputs)}')

    # trains only age (to train both comment next 2 lines)
    # test_lbl = list(map(lambda lbl: lbl[0], test_lbl))
    # train_lbl = list(map(lambda lbl: lbl[0], train_lbl))

    # trains only sex (to train both comment next 2 lines)
    # test_lbl = list(map(lambda lbl: lbl[1], test_lbl))
    # train_lbl = list(map(lambda lbl: lbl[1], train_lbl))

    # describe_distribution(train_lbl, test_lbl)

    test_lbl = numbers_to_input_vectors(test_lbl)
    train_lbl = numbers_to_input_vectors(train_lbl)

    if optimizer != 'binary_tree':
        model = create_model(input_type, optimizer)
        train_model(optimizer, model, train_img, test_img, train_lbl, test_lbl)
    else:
        train_binary_tree(train_img, test_img, train_lbl, test_lbl)
    return


def reset_tensorflow_keras_backend():
    tf.keras.backend.clear_session()
    tf.reset_default_graph()
    # _ = gc.collect()


if __name__ == '__main__':

    # pick one
    # feature_type = 'encodings'
    # feature_type = 'proportions'
    # feature_type = 'proportions_extended'
    # feature_type = 'proportions_sizes'
    feature_type = 'proportions_sizes_children_excluded'

    # PROGRAM STAGES
    # crop_and_save_part1() #1
    # save_face_features(input_type=feature_type)
    train_net(input_type=feature_type, optimizer='adam')
    # train_net(input_type=feature_type, optimizer='binary_tree')

