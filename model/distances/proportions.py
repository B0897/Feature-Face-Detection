import numpy as np

from model.distances.face_parts import FaceParts
from model.distances.face_points import get_points_of_feature, get_nose_point, get_part_center, get_chin_tip, \
    get_chin_quarters, get_all_features_sizes
from model.distances.measure_distance import get_total_length, get_center, get_x_length


def get_all_proportions(features):
    if type(features) != list:
        print(f'features type {type(features)} is not supported')
        return

    nose = get_nose_proportions(features)
    ey_no = get_eyes_nose_proportions(features)
    ey_mo = get_eyes_mouth_proportions(features)
    eb_ch = get_eyebrows_chin_proportion(features)
    ey_fa = get_eyes_face_proportions(features)
    return nose, ey_no, ey_mo, eb_ch, ey_fa


def get_all_sizes(features):
    if type(features) != list:
        print(f'features type {type(features)} is not supported')
        return

    return get_all_features_sizes(features)


def get_all_distances(features):
    if type(features) != list:
        print(f'features type {type(features)} is not supported')
        return

    chin = get_part_center(features, FaceParts.chin.name)
    left_eyebrow = get_part_center(features, FaceParts.left_eyebrow.name)
    right_eyebrow = get_part_center(features, FaceParts.right_eyebrow.name)
    nose_bridge = get_part_center(features, FaceParts.nose_bridge.name)
    nose_tip = get_part_center(features, FaceParts.nose_tip.name)
    left_eye = get_part_center(features, FaceParts.left_eye.name)
    right_eye = get_part_center(features, FaceParts.right_eye.name)
    top_lip = get_part_center(features, FaceParts.top_lip.name)
    bottom_lip = get_part_center(features, FaceParts.bottom_lip.name)

    return get_all_distances_between_points(
        [chin, left_eyebrow, right_eyebrow, nose_bridge, nose_tip, left_eye, right_eye, top_lip, bottom_lip])


def get_nose_proportions(features):
    nose_length_points = get_points_of_feature(features, FaceParts.nose_bridge.name)
    nose_width_points = get_points_of_feature(features, FaceParts.nose_tip.name)

    nose_length = get_total_length(nose_length_points)
    nose_width = get_total_length(nose_width_points)
    return nose_length / nose_width if nose_width != 0 and nose_length != 0 else 0


def get_eyes_nose_proportions(features):
    left_eye = get_part_center(features, FaceParts.left_eye.name)
    right_eye = get_part_center(features, FaceParts.right_eye.name)
    nose_point = get_nose_point(features)

    return (get_total_length([right_eye, left_eye])
            / get_total_length([get_center([left_eye, right_eye]), nose_point]))


def get_eyes_mouth_proportions(features):
    left_eye = get_part_center(features, FaceParts.left_eye.name)
    right_eye = get_part_center(features, FaceParts.right_eye.name)
    mouth_point = get_center([get_part_center(features, FaceParts.top_lip.name),
                              get_part_center(features, FaceParts.bottom_lip.name)])

    return (get_total_length([right_eye, left_eye])
            / get_total_length([get_center([left_eye, right_eye]), mouth_point]))


def get_eyebrows_chin_proportion(features):
    left_eyebrow = get_part_center(features, FaceParts.left_eyebrow.name)
    right_eyebrow = get_part_center(features, FaceParts.right_eyebrow.name)
    between = get_center([left_eyebrow, right_eyebrow])

    chin_tip = get_chin_tip(features)

    right_chin, left_chin = get_chin_quarters(features)

    right_dist = get_total_length([between, right_chin])
    left_dist = get_total_length([between, left_chin])

    return get_total_length([between, chin_tip]) / np.average([right_dist, left_dist])


def get_eyes_face_proportions(features):
    left_eye = get_points_of_feature(features, FaceParts.left_eye.name)
    right_eye = get_points_of_feature(features, FaceParts.right_eye.name)

    eye_size = get_total_length(left_eye) + get_total_length(right_eye)
    face_width = get_x_length(get_points_of_feature(features, FaceParts.chin.name))

    return eye_size / face_width


def get_all_distances_between_points(points,):
    distances = []
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            result = get_total_length([points[i], points[j]])
            distances.append(result)
    return (*distances,)



