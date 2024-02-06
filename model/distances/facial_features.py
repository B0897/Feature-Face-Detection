import face_recognition
import numpy
from PIL import ImageDraw, Image


def find_facial_features(image):
    if type(image) != numpy.ndarray:
        print(f'image of type {type(image)} is not supported')
        return

    face_landmarks_list = face_recognition.face_landmarks(image)
    print("{} faces found in image.".format(len(face_landmarks_list)))
    pil_image = Image.fromarray(image)
    d = ImageDraw.Draw(pil_image)
    for face_landmarks in face_landmarks_list:
        for facial_feature in face_landmarks.keys():
            print("The {} has points: {}".format(facial_feature,
                                                 face_landmarks[facial_feature]))
        for facial_feature in face_landmarks.keys():
            d.line(face_landmarks[facial_feature], width=2)

    pil_image.show()
    return face_landmarks_list


def find_facial_feature_encodings(image):
    if type(image) != numpy.ndarray:
        print(f'image of type {type(image)} is not supported')
        return
    return face_recognition.face_encodings(image)
