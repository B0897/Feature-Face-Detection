from keras.callbacks import LearningRateScheduler
import numpy as np


def get_all_distances_between_points(points,):
    distances = []
    x = 0
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            print("ip = ", points[i], ", jp = ", points[j])
            x += 1
            # result = get_total_length([points[i], points[j]])
            # distances.append(result)

    print(x)
    print(*distances,)
    return (*distances,)


# get_all_distances_between_points(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'])


# LEARNING_RATE1 = 0.01
# LEARNING_RATE2 = LearningRateScheduler(lambda epoch: 1e-4 * (0.75 ** np.floor(epoch / 2)))



# print(type(LEARNING_RATE1))
# print(isinstance(LEARNING_RATE1, float))
# print(type(LEARNING_RATE2))


x = [1, 2, 3, 4, 5 ,6]

print(x[:-1])
