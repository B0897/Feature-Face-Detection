import os
from PIL import Image as pimg
import matplotlib.image as mpimg
import cv2

from common.constants import SAVE_PATH, SAVE_PATH2, SAVE_PATH3


def save_images(images_tuples, folder_name):
    img_names = list(map(lambda img: img[0], images_tuples))
    images = list(map(lambda img: img[1], images_tuples))

    os.mkdir(f'{SAVE_PATH}/{folder_name}')
    os.chdir(f'{SAVE_PATH}/{folder_name}')

    for i, img in enumerate(images):
        cv2.imwrite(f'img{img_names[i]}.jpg', img)

    return


def save_image_hardcoded(img_name, img):
    # os.mkdir(f'{SAVE_PATH2}')
    os.chdir(f'{SAVE_PATH3}')

    # img.save(f'cropped_{img_name}', "JPG")

    img_name = img_name.split("/")[-1]
    mpimg.imsave(f'{SAVE_PATH3}/{img_name}', img)
    # cv2.imwrite(f'cropped_{img_name}', img)
    return
