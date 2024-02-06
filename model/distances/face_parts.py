from enum import Enum


class FaceParts(Enum):
    chin = 'chin'
    left_eyebrow = 'left_eyebrow'
    right_eyebrow = 'right_eyebrow'
    nose_bridge = 'nose_bridge'
    nose_tip = 'nose_tip'
    left_eye = 'left_eye'
    right_eye = 'right_eye'
    top_lip = 'top_lip'
    bottom_lip = 'bottom_lip'


class Direction(Enum):
    up = 'up'
    down = 'bottom'
    right = 'right'
    left = 'left'
