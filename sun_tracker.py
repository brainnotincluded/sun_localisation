import math

import numpy as np
import cv2


def sun_yaw(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    (lh, ls, lv) = 0,0,250
    (hh, hs, hv) = 0,0,255
    mask = cv2.inRange(hsv, (lh, ls, lv), (hh, hs, hv))
    connectivity = 4
    # Perform the operation
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


'''import math

import numpy as np
import cv2

cv2.namedWindow("mask")

def nothing(x):
    pass

cv2.createTrackbar("lh", "mask", 0, 255, nothing)
cv2.createTrackbar("ls", "mask", 0, 255, nothing)
cv2.createTrackbar("lv", "mask", 250, 255, nothing)
cv2.createTrackbar("hh", "mask", 0, 255, nothing)
cv2.createTrackbar("hs", "mask", 0, 255, nothing)
cv2.createTrackbar("hv", "mask", 255, 255, nothing)

cam = cv2.VideoCapture("2.mkv")

while (True):
    success, frame = cam.read()
    if frame is None:
        break
    #frame = cv2.imread("sun_1.jpg")
    #frame[100 : 550, 100 : 550, 0] = 240
    #frame[:, :, 2] += 50

    # print(frame.shape)
    coef_resize = 0.25
    frame = cv2.resize(frame, (512, 512))#(int(frame.shape[1] * coef_resize),int(frame.shape[0] * coef_resize)))
    #frame = 255 - frame

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lh = cv2.getTrackbarPos("lh", "mask")
    ls = cv2.getTrackbarPos("ls", "mask")
    lv = cv2.getTrackbarPos("lv", "mask")
    hh = cv2.getTrackbarPos("hh", "mask")
    hs = cv2.getTrackbarPos("hs", "mask")
    hv = cv2.getTrackbarPos("hv", "mask")

    mask = cv2.inRange(hsv, (lh, ls, lv), (hh, hs, hv))

    cv2.imshow("mask", mask)

    connectivity = 4
    # Perform the operation
    output = cv2.connectedComponentsWithStats(mask, connectivity, cv2.CV_32S)

    # Get the results
    # The first cell is the number of labels
    num_labels = output[0]
    # The second cell is the label matrix
    labels = output[1]
    # The third cell is the stat matrix
    stats = output[2]

    filtered = np.zeros_like(mask)
    print(stats)
    print("---------------")
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

        #print(a)

        if (a >= 500):
            filtered[np.where(labels == i)] = 255
            #print(a)
            if a > max_a:
                max_a = a
                sun_x = int(l + w / 2)
                sun_y = int(t + h / 2)

            cv2.putText(frame, str(a), (l + w, t + h), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.rectangle(frame, (l, t), (l + w, t + h), (0, 255, 0), 2)
    if sun_x != 0 and sun_y != 0:
        cam_width_angle = math.pi
        print(frame.shape)
        yaw_sun = round(math.degrees(math.atan(cam_width_angle) *  (sun_x - frame.shape[0]/2) / frame.shape[0]),1)
        cv2.circle(frame, (sun_x, sun_y), 5, (0, 0, 0), 2)
        cv2.putText(frame, f"yaw {yaw_sun}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    #print("=====================")
    #break

    cv2.imshow("frame", frame)
    # cv2.imshow("hsv", hsv[:, :, 0])
    cv2.imshow("filtered", filtered)

    key = cv2.waitKey(5) & 0xFF

    if (key == ord('q')):
        break

time_now = 12.5
angle_12 = (12.5 - 12) * 15 # Угол между 12 и биссектриссой = углу между текущим направлением и биссектриссой. Показывает на юг


cam.release()
cv2.destroyAllWindows()
cv2.waitKey(0)
'''
