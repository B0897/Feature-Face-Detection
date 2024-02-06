from model.distances.face_parts import FaceParts, Direction
from model.distances.measure_distance import get_centroid, get_center, get_bottom, get_total_length, \
    get_closest_to_x_distance


def get_all_features_sizes(features):
    chin = get_total_length(get_points_of_feature(features, FaceParts.chin.name))
    left_eyebrow = get_total_length(get_points_of_feature(features, FaceParts.left_eyebrow.name))
    right_eyebrow = get_total_length(get_points_of_feature(features, FaceParts.right_eyebrow.name))
    nose_bridge = get_total_length(get_points_of_feature(features, FaceParts.nose_bridge.name))
    nose_tip = get_total_length(get_points_of_feature(features, FaceParts.nose_tip.name))
    left_eye = get_total_length(get_points_of_feature(features, FaceParts.left_eye.name))
    right_eye = get_total_length(get_points_of_feature(features, FaceParts.right_eye.name))
    top_lip = get_total_length(get_points_of_feature(features, FaceParts.top_lip.name))
    bottom_lip = get_total_length(get_points_of_feature(features, FaceParts.bottom_lip.name))
    return chin, left_eyebrow, right_eyebrow, nose_bridge, nose_tip, left_eye, right_eye, top_lip, bottom_lip


def get_chin_quarters(features):
    chin_points = get_points_of_feature(features, FaceParts.chin.name)
    chin_tip = get_bottom(chin_points)

    lip_points = get_points_of_feature(features, FaceParts.bottom_lip.name)
    mouth_width = get_total_length(lip_points)

    right_quarter = get_closest_to_x_distance(chin_points, chin_tip, 0.75*mouth_width, Direction.right)
    left_quarter = get_closest_to_x_distance(chin_points, chin_tip, 0.75*mouth_width, Direction.left)

    return right_quarter, left_quarter


def get_nose_point(features):
    length_points = get_points_of_feature(features, FaceParts.nose_bridge.name)
    width_points = get_points_of_feature(features, FaceParts.nose_tip.name)

    sum_points = lambda p: p[0] + p[1]
    sums = sum_points(length_points)
    max_index = sums.index(max(sums))

    lowest_point = length_points[max_index]
    center_bridge = get_centroid(width_points)
    return get_centroid([lowest_point, center_bridge])


def get_chin_tip(features):
    chin_points = get_points_of_feature(features, FaceParts.chin.name)
    return get_bottom(chin_points)


def get_part_center(features, part_name):
    points = get_points_of_feature(features, part_name)
    return get_center(points)


def get_points_of_feature(features, part_name):
    for fl in features:
        for ff in fl.keys():
            if ff == part_name:
                return fl[ff]
