import math
import numpy as np

from model.distances.face_parts import Direction

def get_center(points):
    return get_centroid(points)


def get_centroid(points):
    x = [p[0] for p in points]
    y = [p[1] for p in points]
    centroid = (sum(x) / len(points), sum(y) / len(points))
    return centroid


def get_closest_to_x_distance(points, ref_point, distance, direction):
    img_point = (0, 0)

    if direction == Direction.up:
        img_point = (ref_point[0], max(ref_point[1] - distance, 0))
    elif direction == Direction.down:
        img_point = (ref_point[0], ref_point[1] + distance)
    elif direction == Direction.right:
        img_point = (ref_point[0] + distance, ref_point[1])
    elif direction == Direction.left:
        img_point = (max(ref_point[0] - distance, 0), ref_point[1])
    return find_closest_point(img_point, points)


def get_top(points):
    ys = get_ys_from_points(points)
    max_value = min(ys) # bo y rosnie w dol
    max_points = list(filter(lambda p: p[1] == max_value, points))
    if len(max_points) > 1:
        median = np.median(get_xs_from_points(max_points))
        return find_closest_point((median, max_value), points)
    else:
        return max_points[0]


def get_bottom(points):
    ys = get_ys_from_points(points)
    min_value = max(ys) # bo y rosnie w dol
    min_points = list(filter(lambda p: p[1] == min_value, points))
    if len(min_points) > 1:
        median = np.median(get_xs_from_points(min_points))
        return find_closest_point((median, min_value), points)
    else:
        return min_points[0]


def get_rightest(points):
    xs = get_xs_from_points(points)
    max_value = max(xs)
    max_points = list(filter(lambda p: p[0] == max_value, points))
    if len(max_points) > 1:
        median = np.median(get_ys_from_points(max_points))
        return find_closest_point((max_value, median), points)
    else:
        return max_points[0]


def get_leftest(points):
    xs = get_xs_from_points(points)
    min_value = min(xs)
    min_points = list(filter(lambda p: p[0] == min_value, points))
    if len(min_points) > 1:
        median = np.median(get_ys_from_points(min_points))
        return find_closest_point((min_value, median), points)
    else:
        return min_points[0]


def find_closest_point(ref_point, points):
    dist = list(map(lambda p: (ref_point[0] - p[0])**2 + (ref_point[1] - p[1])**2, points))
    closest_index = dist.index(min(dist))
    return points[closest_index]


def get_total_length(points):
    return math.sqrt(get_x_length(points)**2 + get_y_length(points)**2)


def get_x_length(points):
    xs = get_xs_from_points(points)
    return max(xs) - min(xs)


def get_xs_from_points(points):
    return list(map(get_x_coord, points))


def get_y_length(points):
    ys = get_ys_from_points(points)
    return max(ys) - min(ys)


def get_ys_from_points(points):
    return list(map(get_y_coord, points))


def get_x_coord(point):
    return point[0]


def get_y_coord(point):
    return point[1]
