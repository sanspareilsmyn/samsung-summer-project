# -*- coding: utf-8 -*-
import cv2
from sklearn.svm import LinearSVC
from skimage.feature import hog
from skimage.io import imread
from sklearn.externals import joblib
import glob
import os
from config import *
import numpy as np


def predict_display():
    des_type = 'HOG'
    clf_type = 'LIN_SVM'
    os.system('sudo modprobe bcm2835-v4l2')

    a, b, c = None, None, None
    cap = cv2.VideoCapture(-1)
    cap.set(3, 640)
    cap.set(4, 480)

    if cap.isOpened():
        ret, a = cap.read()
        ret, b = cap.read()

        while ret:
            ret, c = cap.read()
            draw = c.copy()

            if not ret:
                break

            a_gray = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
            b_gray = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
            c_gray = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)

            absdiff1_temp = cv2.absdiff(a_gray, b_gray)
            absdiff2_temp = cv2.absdiff(b_gray, c_gray)

            ret, absdiff1 = cv2.threshold(absdiff1_temp, 50, 255, cv2.THRESH_BINARY)
            ret, absdiff2 = cv2.threshold(absdiff2_temp, 50, 255, cv2.THRESH_BINARY)

            diff_temp = cv2.bitwise_and(absdiff1, absdiff2)

            kernel = np.ones((5, 5), np.uint8)
            diff = cv2.morphologyEx(diff_temp, cv2.MORPH_OPEN, kernel)
            diff_cnt = cv2.countNonZero(diff)

            if diff_cnt > max_diff:
                nzero = np.nonzero(diff)
                roi = cv2.rectangle(draw, (min(nzero[1]), min(nzero[0])), (max(nzero[1]), max(nzero[0])), (0, 255, 0),
                                    2)
                cv2.imwrite("roi.jpg", roi)
                im = imread("roi.jpg", as_grey=True)
                if des_type == "HOG":
                    fd = hog(im, orientations, pixels_per_cell, cells_per_block)
                    clf = joblib.load('../data/models/hogsvm.model')
                    fd = np.array(fd)
                    fd = np.squeeze(fd)
                    fd = np.reshape(fd, (-1, 657072))

                    if str(clf.predict(fd)) == '[0]':
                        cv2.putText(draw, 'coin', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                    if str(clf.predict(fd)) == '[1]':
                        cv2.putText(draw, 'scissors', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2,
                                    cv2.LINE_AA)
                    if str(clf.predict(fd)) == '[2]':
                        cv2.putText(draw, 'coke', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                    if str(clf.predict(fd)) == '[3]':
                        cv2.putText(draw, 'toothpaste', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2,
                                    cv2.LINE_AA)

            cv2.imshow('window', draw)
            cv2.imshow('diff', diff)

            a = b
            b = c

            if cv2.waitKey(1) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()


predict_display()