import math
import os
from numpy import array

import cv2

# Path to recognizer
EIGEN_RECOGNIZER_PATH = '../rsc/recognition_files/eigen_recognizer.yml'
FISHER_RECOGNIZER_PATH = '../rsc/recognition_files/fisher_recognizer.yml'
LBHP_RECOGNIZER_PATH = '../rsc/recognition_files/lbhp_recognizer.yml'

# Path to cascades
FRONTAL_FACE_CASCADE_PATH = "../rsc/cascades/haarcascade_frontalface_default.xml"
RIGHT_EYE_CASCADE_PATH = "../rsc/cascades/haarcascade_righteye_2splits.xml"
LEFT_EYE_CASCADE_PATH = "../rsc/cascades/haarcascade_lefteye_2splits.xml"


