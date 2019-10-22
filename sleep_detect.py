from keras import backend as K
import imutils
from keras.models import load_model
import numpy as np
import keras
import requests
from scipy.spatial import distance as dist
from imutils import face_utils
import time
import dlib
import cv2, os, sys
import collections
import random
import face_recognition
import pickle
import math
import threading
import tensorflow as tf

################## PHAN DINH NGHIA CLASS, FUNCTION #######################


# Class dinh nghia vi tri 2 mat va long may con nguoi
class FacialLandMarksPosition:
    left_eye_start_index, left_eye_end_index = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    right_eye_start_index, right_eye_end_index = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    right_eyebrow_start_index, right_eyebrow_end_index = face_utils.FACIAL_LANDMARKS_IDXS["right_eyebrow"]
    left_eyebrow_start_index, left_eyebrow_end_index = face_utils.FACIAL_LANDMARKS_IDXS["left_eyebrow"]
    
# Ham du doan mat dong hay mo
def predict_eye_state(model, image):
    # Resize anh ve 100x100
    image = cv2.resize(image, (100, 100))
    image = image.astype(dtype=np.float32)

    # Chuyen thanh tensor
    image_batch = np.reshape(image, (1, 100, 100, 1))

    return np.argmax(model.predict(image_batch)[0])


################ CHUONG TRINH CHINH ##############################

# Load model dlib de phat hien cac diem tren mat nguoi - landmark
facial_landmarks_predictor = '68_face_landmarks_predictor.dat'
predictor = dlib.shape_predictor(facial_landmarks_predictor)

# Load model predict xem mat nguoi dang dong hay mo
model = load_model('pretrain3.hdf5')

# Lay anh tu Webcam
cap = cv2.VideoCapture(0)
scale = 0.5
countClose = 0
currState = 0
alarmThreshold = 5


while (True):
    c = time.time()
    # Doc anh tu webcam va chuyen thanh RGB
    ret, frame = cap.read()
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Resize anh con 50% kich thuoc goc
    original_height, original_width = image.shape[:2]
    resized_image = cv2.resize(image, (0, 0), fx=scale, fy=scale)

    # Chuyen sang he mau LAB de lay thanh lan Lightness
    lab = cv2.cvtColor(resized_image, cv2.COLOR_BGR2LAB)
    l, _, _ = cv2.split(lab)
    resized_height, resized_width = l.shape[:2]
    height_ratio, width_ratio = original_height / resized_height, original_width / resized_width

    # Tim kiem khuon mat bang HOG
    face_locations = face_recognition.face_locations(l, model='hog')

    # Neu tim thay it nhat 1 khuon mat
    if len(face_locations):

        # Lay vi tri khuon mat
        top, right, bottom, left = face_locations[0]
        x1, y1, x2, y2 = left, top, right, bottom
        x1 = int(x1 * width_ratio)
        y1 = int(y1 * height_ratio)
        x2 = int(x2 * width_ratio)
        y2 = int(y2 * height_ratio)

        # Trich xuat vi tri 2 mat

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        shape = predictor(gray, dlib.rectangle(x1, y1, x2, y2))
        face_landmarks = face_utils.shape_to_np(shape)

        left_eye_indices = face_landmarks[FacialLandMarksPosition.left_eye_start_index:
                                          FacialLandMarksPosition.left_eye_end_index]

        left_eyebrow_indices = face_landmarks[FacialLandMarksPosition.left_eyebrow_start_index:
                                          FacialLandMarksPosition.left_eyebrow_end_index]
        add_left_eye_eyebrow = np.array([])
        add_left_eye_eyebrow = np.row_stack([left_eyebrow_indices[:,:],left_eye_indices[:,:]])
        (x, y, w, h) = cv2.boundingRect(add_left_eye_eyebrow)
        left_eye = gray[y:y + h + 20, x - 10:x + w ]

        right_eye_indices = face_landmarks[FacialLandMarksPosition.right_eye_start_index:
                                           FacialLandMarksPosition.right_eye_end_index]
        right_eyebrow_indices = face_landmarks[FacialLandMarksPosition.right_eyebrow_start_index:
                                          FacialLandMarksPosition.right_eyebrow_end_index]
        (x, y, w, h) = cv2.boundingRect(np.array([right_eye_indices]))
        
        add_right_eye_eyebrow = np.array([])
        add_right_eye_eyebrow = np.row_stack([right_eyebrow_indices[:,:],right_eye_indices[:,:]])
        (x, y, w, h) = cv2.boundingRect(add_right_eye_eyebrow)
        right_eye = gray[y:y + h + 20, x:x + w]

        left_eye_open = 'yes' if predict_eye_state(model=model, image=left_eye) else 'no'
        right_eye_open = 'yes' if predict_eye_state(model=model, image=right_eye) else 'no'

        print('left eye open: {0}    right eye open: {1}'.format(left_eye_open, right_eye_open))

        # Neu 2 mat mo thi hien thi mau xanh con khong thi mau do
        if left_eye_open == 'yes' and right_eye_open == 'yes':
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            currState = 0
            countClose = 0

        else:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            currState = 1
            countClose +=1


    frame = cv2.flip(frame, 1)
    if countClose > alarmThreshold:
        cv2.putText(frame, "Canh bao! Phat hien ngu gat!", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255),
                    lineType=cv2.LINE_AA)
    cv2.imshow('Sleep Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
 