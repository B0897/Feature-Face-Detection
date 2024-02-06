from keras.callbacks import LearningRateScheduler
import numpy as np

WAIT_KEY = 1000
INPUT_SIZE = 5
TEST_SIZE = 0.2
LEARNING_RATE = 0.00001
# LEARNING_RATE = LearningRateScheduler(lambda epoch: 1e-4 * (0.75 ** np.floor(epoch / 2)))
IMAGE_CROP_MARGIN_RATE = 0.3

# model parameters
OPTIMIZER_ADAM = 'adam'
OPTIMIZER_SGD = 'sgd'
LOSS = ['mse', 'binary_crossentropy']
LOSS2 = ['mse', 'categorical_crossentropy', 'mean_squared_error', 'mae']
METRICS = ['accuracy']


# model parameters

DB_PATH = r'/home/barbara/Documents/pracka2/dataset/UTK_Face/'
SAVE_PATH = r'/home/barbara/Documents/pracka2/dataset/UTK_Face/'
SAVE_PATH2 = r'/home/barbara/Documents/pracka2/dataset/UTK_Face/part1_cropped'
SAVE_PATH3 = r'/home/barbara/Documents/pracka2/dataset/UTK_Face/part1_cropped2'
CROPPED_DATA_RESULTS_PATH = r'/home/barbara/Documents/pracka2/SDM@-main_2023_04_18_edited/cvs/'

IMAGES_PATH = r'/home/barbara/Documents/pracka2/SDM@-main_2023_08_02_edited/wyniki/modele/'

PROGRAM_RESULTS_PATH = r'/home/barbara/PycharmProjects/featureFaceDetection/results/'

# network properties
EPOCHS = 200
# BATCH_SIZE = 128 #firstly
BATCH_SIZE = 64
SHUFFLE = True

# 2, 3, 4 or 8
NUMBER_OF_CLASSES = 8


# NUMBER_OF_CLASSES = 5

