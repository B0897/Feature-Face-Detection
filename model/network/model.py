import datetime
from sklearn.preprocessing import LabelEncoder
from keras.layers import *
from keras.models import *
from keras.callbacks import LearningRateScheduler
import matplotlib.pyplot as plt
import numpy as np
from keras.optimizers import SGD
from sklearn.tree import DecisionTreeClassifier
from tensorflow.keras import layers, models
from tensorflow import keras
# from tensorflow.python.estimator import keras
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score

from common.constants import INPUT_SIZE, LEARNING_RATE, OPTIMIZER_SGD, LOSS, METRICS, EPOCHS, BATCH_SIZE, SHUFFLE, \
    OPTIMIZER_ADAM, LOSS2, NUMBER_OF_CLASSES


def create_model(input_type, optimizer):
    if input_type == 'encodings':
        mlp_model = build_mlp_encodings_model(128, optimizer)
    elif input_type == 'proportions':
        mlp_model = build_mlp_encodings_model(5, optimizer)
    elif input_type == 'proportions_extended':
        mlp_model = build_mlp_encodings_model(41, optimizer)
    elif input_type == 'proportions_sizes':
        mlp_model = build_mlp_encodings_model(51, optimizer)
    elif input_type == 'proportions_sizes_children_excluded':
        mlp_model = build_mlp_encodings_model(50, optimizer)
    return mlp_model


def train_model(optimizer, model, train_img, test_img, train_lbl, test_lbl):
    lr_sched = LearningRateScheduler(lambda epoch: 1e-4 * (0.75 ** np.floor(epoch / 2)))

    train_lbl = np.asarray(train_lbl)
    test_lbl = np.asarray(test_lbl)

    if optimizer == OPTIMIZER_SGD:
        h = model.fit(train_img, train_lbl, validation_data=(test_img, test_lbl),
                      epochs=EPOCHS, batch_size=BATCH_SIZE, shuffle=SHUFFLE)
    else:
        # h = model.fit(train_img, train_lbl, validation_data=(test_img, test_lbl),
        #               epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[lr_sched], shuffle=SHUFFLE)
        h = model.fit(train_img, train_lbl, validation_data=(test_img, test_lbl),
                      epochs=EPOCHS, batch_size=BATCH_SIZE, shuffle=SHUFFLE)

    # model.save(f'model{model_label}_{datetime.datetime.now()}.h5')
    # print_results(h)
    plot_results(h)
    plot_confusion_matrix(model, test_img, test_lbl)
    return model


def plot_results(history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history['accuracy'], label='Dane treningowe')
    axes[0].plot(history.history['val_accuracy'], label='Dane walidacyjne')
    axes[0].set_title('Dokładność modelu')
    axes[0].set_xlabel('Epoki')
    axes[0].set_ylabel('Dokładność')
    axes[0].legend()
    axes[0].grid()

    axes[1].plot(history.history['loss'], label='Dane treningowe')
    axes[1].plot(history.history['val_loss'], label='Dane walidacyjne')
    axes[1].set_title('Funkcja straty')
    axes[1].set_xlabel('Epoki')
    axes[1].set_ylabel('Strata')
    axes[1].legend()
    axes[1].grid()

    plt.tight_layout()
    plt.show()
    return


def plot_confusion_matrix(model, img, labels):
    predict_x = model.predict(img)
    pred = np.argmax(predict_x, axis=1)
    labels = np.argmax(labels, axis=1)

    label_encoder = LabelEncoder()
    y_test_encoded = label_encoder.fit_transform(labels)
    y_pred_encoded = label_encoder.transform(pred)

    cm = confusion_matrix(y_test_encoded, y_pred_encoded)

    classes = label_encoder.classes_
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap='viridis', values_format='d')

    plt.title('Confusion Matrix')
    plt.show()


def build_mlp_model(optimizer):
    # OPTIMIZER = (SGD(learning_rate=LEARNING_RATE) if optimizer == OPTIMIZER_SGD else OPTIMIZER_ADAM)
    model = Sequential()
    model.add(Dense(INPUT_SIZE, input_shape=(INPUT_SIZE,), activation='relu'))

    model.add(Dense(16, activation='relu'))
    model.add(Dense(16, activation='relu'))

    model.add(Dense(2, activation='softmax'))

    model.compile(loss=LOSS2, optimizer=optimizer, metrics=METRICS)
    model.summary()
    return model


def build_mlp_model2(optimizer):
    model = models.Sequential()

    model.add(layers.Input(shape=(5,)))

    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(2, activation='softmax'))

    model.compile(optimizer=optimizer, loss=LOSS2, metrics=METRICS)

    model.summary()
    return model


def build_mlp_model_age(optimizer):
    dropout = 0.2

    model = models.Sequential()

    model.add(layers.Input(shape=(5,)))

    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(dropout))

    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(dropout))

    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(dropout))

    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(dropout))

    model.add(layers.Dense(1, activation='linear'))

    model.compile(optimizer=optimizer, loss=LOSS2, metrics=METRICS)

    model.summary()
    return model


def build_mlp_encodings_model(input_size, optimizer):
    dropout = 0.1
    model = models.Sequential()

    model.add(layers.Input(shape=(input_size,)))

    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(dropout))

    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(dropout))

    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(dropout))

    model.add(layers.Dense(NUMBER_OF_CLASSES, activation='softmax'))

    if optimizer == 'adam':
        optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    elif optimizer == 'sgd':
        optimizer = keras.optimizers.SGD(learning_rate=LEARNING_RATE)

    model.compile(optimizer=optimizer, loss=LOSS2, metrics=METRICS)
    model.summary()
    return model


def train_binary_tree(train_img, test_img, train_lbl, test_lbl):
    tree_classifier = DecisionTreeClassifier()
    tree_classifier.fit(train_img, train_lbl)
    test_predicted = tree_classifier.predict(test_img)
    accuracy = accuracy_score(test_lbl, test_predicted)
    print(f'Accuracy: {accuracy}')
