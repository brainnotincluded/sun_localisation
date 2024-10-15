import os
def launch_refact_images():
  images_path = 'assets/frame2'

  with open("assets//frame_data2.txt", "r") as file:
    frame_data = {}
    for line in file:
      n, date, time = line.strip().split()
      frame_data[n+".jpg"] = time


  for filename in os.listdir(images_path):
    if filename.endswith((".jpg", ".jpeg", ".png")):
      old_path = os.path.join(images_path, filename)
      new_filename = frame_data.get(filename, filename)+".jpg"
      new_path = os.path.join(images_path, new_filename)
      print(old_path, new_path, new_filename, filename)
      os.rename(old_path, new_path)

  print("Файлы переименованы!")

if __name__ == "__main__":
  launch_refact_images()