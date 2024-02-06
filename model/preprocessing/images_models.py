import keras.utils
from common.constants import IMAGES_PATH, OPTIMIZER_SGD, OPTIMIZER_ADAM, LOSS2, METRICS
from keras.layers import *
from keras.models import *

from tensorflow.keras import layers, models
from tensorflow import keras


EPOCHS = 15
# BATCH_SIZE = 128 #firstly
BATCH_SIZE = 64
SHUFFLE = True

IMAGE_SIZE = 300


def build_flat_model(input_size):
    inputs = Input(shape=(input_size, input_size, 1))
    conv1 = Conv2D(32, kernel_size=(3, 3), activation='relu')(inputs)
    conv2 = Conv2D(64, kernel_size=(3, 3), activation='relu')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(128, kernel_size=(3, 3), activation='relu')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv3)
    x = Dropout(0.25)(pool2)
    return Flatten()(x), inputs


def build_age_model(flat_model):
    dropout = Dropout(0.5)
    age_model = Dense(128, activation='relu')(flat_model)
    age_model = dropout(age_model)
    age_model = Dense(64, activation='relu')(age_model)
    age_model = dropout(age_model)
    age_model = Dense(32, activation='relu')(age_model)
    age_model = dropout(age_model)
    return Dense(1, activation='relu')(age_model)


def build_sex_model(flat_model):
    dropout = Dropout(0.5)
    gender_model = Dense(128, activation='relu')(flat_model)
    gender_model = dropout(gender_model)
    gender_model = Dense(64, activation='relu')(gender_model)
    gender_model = dropout(gender_model)
    gender_model = Dense(32, activation='relu')(gender_model)
    gender_model = dropout(gender_model)
    gender_model = Dense(16, activation='relu')(gender_model)
    gender_model = dropout(gender_model)
    gender_model = Dense(8, activation='relu')(gender_model)
    gender_model = dropout(gender_model)
    return Dense(1, activation='sigmoid')(gender_model)


def create_model(optimizer):
    flat_model, inputs = build_flat_model(IMAGE_SIZE)
    age_model = build_age_model(flat_model)
    sex_model = build_sex_model(flat_model)

    # OPTIMIZER = (SGD(learning_rate=LEARNING_RATE) if optimizer == OPTIMIZER_SGD else OPTIMIZER_ADAM)
    OPTIMIZER = optimizer

    agemodel = Model(inputs=inputs, outputs=age_model)
    agemodel.compile(optimizer=OPTIMIZER, loss=LOSS2, metrics=METRICS)
    agemodel.summary()

    sexmodel = Model(inputs=inputs, outputs=sex_model)
    sexmodel.compile(optimizer=OPTIMIZER, loss=LOSS2, metrics=METRICS, )
    sexmodel.summary()

    return agemodel, sexmodel


def build_age_model2():
    model = models.Sequential()
    model.add(layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 1)))
    model.add(layers.Conv2D(32, kernel_size=(3, 3), activation='relu'))
    model.add(layers.Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.25))

    model.add(layers.Flatten())
    model.add(layers.Dense(units=128, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(units=64, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(units=32, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(units=1, activation='relu'))

    model.compile(optimizer=OPTIMIZER_SGD, loss=LOSS2, metrics=METRICS)
    model.summary()
    return model


def build_sex_model2():
    model = models.Sequential()
    model.add(layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 1)))
    model.add(layers.Conv2D(32, kernel_size=(3, 3), activation='relu'))
    model.add(layers.Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.25))

    model.add(layers.Flatten())
    model.add(layers.Dense(units=128, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(units=64, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(units=32, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(units=16, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(units=8, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(units=1, activation='sigmoid'))

    model.compile(optimizer=OPTIMIZER_SGD, loss=LOSS2, metrics=METRICS)
    model.summary()
    return model


def draw_model(model, file_name):
    keras.utils.plot_model(
        model,
        to_file=file_name,
        show_shapes=True,
        show_dtype=False,
        show_layer_names=True,
        rankdir="TB",
        expand_nested=True,
        dpi=200,
        show_layer_activations=True,
        show_trainable=False
    )


file_name = IMAGES_PATH + 'sex_model2'

agemodel2 = build_sex_model2()
draw_model(agemodel2, file_name)