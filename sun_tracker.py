import math

import numpy as np
import cv2


def sun_yaw(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    (lh, ls, lv) = 0,0,250
    (hh, hs, hv) = 0,0,255
    mask = cv2.inRange(hsv, (lh, ls, lv), (hh, hs, hv))
    connectivity = 4
    output = cv2.connectedComponentsWithStats(mask, connectivity, cv2.CV_32S)
    num_labels = output[0]

    labels = output[1]

    stats = output[2]

    filtered = np.zeros_like(mask)
    print(cv2.CC_STAT_AREA)
    sun_x = 0
    sun_y = 0
    max_a = 0
    for i in range(1, num_labels):
        a = stats[i, cv2.CC_STAT_AREA]
        t = stats[i, cv2.CC_STAT_TOP]
        l = stats[i, cv2.CC_STAT_LEFT]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        if (a >= 500):
            filtered[np.where(labels == i)] = 255
            if a > max_a:
                max_a = a
                sun_x = int(l + w / 2)
                sun_y = int(t + h / 2)
    if sun_x != 0 and sun_y != 0:
        cam_width_angle = math.pi
        print(frame.shape)
        yaw_sun = round(math.degrees(math.atan(cam_width_angle) *  (sun_x - frame.shape[0]/2) / frame.shape[0]),1)
        return (sun_x, sun_y,yaw_sun,max_a)
    else:
        return None