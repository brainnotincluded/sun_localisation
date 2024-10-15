import cv2
import numpy as np

def launch_calib1():
  def undistort_image(image_path, camera_matrix, dist_coeffs):
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w,h), 1, (w,h))
    # Undistort
    dst = cv2.undistort(img, camera_matrix, dist_coeffs, None, newcameramtx)
    # Crop the image
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    cv2.imwrite(image_path, dst)
  camera_matrix = np.array([[593.3836415767648, 0.0, 362.9188616787984],
         [0.0, 530.5741645380468, 219.96420993282118],
         [0.0, 0.0, 1.0]])

  dist_coeffs = np.array([[-0.420768330261403, 0.08916544139545872, -0.0020588116149551482, 0.002024447002549296, 0.27692536721285477]])

  with open("assets//frame_data2.txt", "r") as file:
    frame_data = {}
    for line in file:
      n, date, time_ = line.strip().split()
      v2 = f"assets/frame2/{time_}.jpg"
      undistort_image(v2, camera_matrix, dist_coeffs)
if __name__ == "__main__":
  launch_calib1()