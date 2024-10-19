import os

import cv2
def launch():
 def video_to_frames(video_path, output_dir="frames"):
  vidcap = cv2.VideoCapture(video_path)
  length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
  fps = vidcap.get(cv2.CAP_PROP_FPS)
  success, image = vidcap.read()
  count = 0
  _ = []
  if not vidcap.isOpened():
   raise IOError("Cannot open webcam")

  if not os.path.exists(output_dir):
   os.makedirs(output_dir)
  print(video_path)
  i = float(video_path.split("/")[-1].removesuffix(".mp4").removesuffix(".avi"))
  while success:
   i += 1/fps
   _ += [f"{i} {i}.png\n"]
   cv2.imwrite(os.path.join(output_dir, f"{i}.png"), image)
   success, image = vidcap.read()
   count += 1

  vidcap.release()
  print(f"Successfully extracted {count} frames from {video_path} to {output_dir}")
  with open("data.txt", "w") as file:
   file.writelines(_)

 video_directory = "video/"
 for filename in os.listdir(video_directory):
  if filename.endswith((".mp4", ".avi")):
   video_to_frames(video_directory + filename)
   break

 with open("data.txt", "r") as f:
  with open("data_.txt", "w") as file:
   _ = []
   for line in f:
    _ += line.split()[0] + "\n"
   file.writelines(_)
   
if __name__ == "__main__":
 launch()
