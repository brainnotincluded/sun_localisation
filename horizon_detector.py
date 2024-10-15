import numpy as np
import cv2, time
import math
from sun_tracker import sun_yaw
import statistics
def launch_horizon_detector():
    v1 = "assets//2.mkv"
    v3 = "flying_turn.avi"
    v4 = "y1.mp4"
    v5 = "3.mp4"
    with open("assets//frame_data2.txt", "r") as file:
      frame_data = {}
      for line in file:
        n, date, time_ = line.strip().split()
        v2 = f"assets/frame2/{time_}.jpg"
        cap = cv2.VideoCapture(v2)

        lower = np.array([207, 115, 103])
        upper = np.array([210, 60, 110])
        responses = []
        def houghline(edges):

            lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)
            if lines is None:
                return None
            r = []
            theta = []
            print('lines', lines)
            for r_theta in lines:
                arr = np.array(r_theta[0], dtype=np.float64)
                r.append(arr[0])
                theta.append(arr[1])

            r = statistics.mean(r)
            theta = statistics.mean(theta)
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * r
            y0 = b * r
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            y1 += 8
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            y2 += 8
            if x1 == x2:
                x2 = x1 + 1
            k = (y2 - y1) / (x2 - x1)
            b = y1 - k * x1
            pitch = round((1 - (k * edges.shape[0] / 2 + b) / (edges.shape[1])) * 200 - 100, 1)
            roll = round(math.degrees(math.atan(k)), 1)

            return ((x1, y1), (x2, y2), (pitch, roll),)


        while True:
            ret, frame = cap.read()
            if frame is None:
                break
            frame = cv2.resize(frame, (512, 512))
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hsv_mask = cv2.inRange(hsv, lower, upper)
            bgr2gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            bgr2gray = cv2.add(bgr2gray, hsv_mask)
            image_blurred = cv2.GaussianBlur(bgr2gray, ksize=(3, 3), sigmaX=0)
            # image_blurred = bgr2gray
            _, image_thresholded = cv2.threshold(
                image_blurred, thresh=0, maxval=1,
                type=cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            image_thresholded = image_thresholded - 1
            image_closed = image_thresholded

            cv2.imshow("img_closed", image_closed)

            horizon_x1 = 0
            horizon_x2 = bgr2gray.shape[1] - 1
            try:
                horizon_y1 = max(np.where(image_closed[:, horizon_x1] == 0)[0])
            except:
                continue
            try:
                horizon_y2 = max(np.where(image_closed[:, horizon_x2] == 0)[0])
            except:
                horizon_y2 = horizon_y1
            kernel = np.ones((16, 16), dtype="uint8")
            image_closed = cv2.dilate(image_closed, kernel)
            cv2.imshow("img_closed_2_______", image_closed)
            edges = cv2.Canny(image_closed, 70, 135)
            line = houghline(edges)
            from ransac import ransac as ransac2
            print('line', line)
            if not line is None:
                cv2.line(frame, line[0], line[1], (0, 0, 255), 2)
                cv2.putText(frame, f"roll {line[2][1]}, pitch {line[2][0]}%", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 255, 0), 2)
            _sun_yaw = sun_yaw(frame)
            if not _sun_yaw is None:
                sun_x, sun_y, yaw, area_sun = _sun_yaw
                cv2.circle(frame, (sun_x, sun_y), 5, (0, 0, 0), 2)
                cv2.putText(frame, f"a {area_sun}", (sun_x + 20, sun_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2,
                            cv2.LINE_AA)
                cv2.putText(frame, f"yaw {yaw}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.imshow("frame", frame)
            cv2.imshow("edges", edges)

            key = cv2.waitKey(1)
            if key == ord(" "):
                break
            time.sleep(0.0005)

if __name__ == '__main__':
    launch_horizon_detector()