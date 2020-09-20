import cv2
import os
import numpy as np
import matplotlib

str = 'Motion detected'

def rotate(src):
    dst = cv2.flip(src, -1)
    return dst

def show_video_difference():
    os.system('sudo modprobe bcm2835-v4l2')
    cap = cv2.VideoCapture(-1)
    cap.set(3, 640)
    cap.set(4, 480)
    fgbg = cv2.createBackgroundSubtractorMOG2(varThreshold=100)

    while True:
        ret, frame = cap.read()
        frame_rotate = rotate(frame)
        fgmask_temp = fgbg.apply(frame_rotate)
        fgmask = rotate(fgmask_temp)

        if not ret:
            print('Not Found Devices')
            break

        nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(fgmask)

        for index, centroid in enumerate(centroids):
            if stats[index][0] == 0 and stats[index][1] == 0:
                continue
            if np.any(np.isnan(centroid)):
                continue

            x, y, width, height, area = stats[index]
            centerX, centerY = int(centroid[0]), int(centroid[1])

            if area > 100:
                cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0))
                cv2.putText(frame, str, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        fgmask_3_channel = cv2.cvtColor(fgmask, cv2.COLOR_GRAY2BGR)
        combined_video = np.vstack((frame_rotate, fgmask_3_channel))
        cv2.imshow('motion sensor', combined_video)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


show_video_difference()