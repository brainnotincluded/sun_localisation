import os
images_path = 'assets//'
for filename in os.listdir(images_path):
    if filename.endswith('.csv'):
        os.rename(images_path+filename, images_path+filename[:-4]+"_" + ".txt")

        with open(images_path+filename[:-4]+"_" + ".txt", "r") as f:
            _ = f.readlines()
            for i, line in enumerate(_):
                line = line.replace('"', '')
                line = line.replace(',', ' ')
                _[i] = line
                print(line)
            with open(images_path+filename[:-4] + ".txt", "w") as f_:
                f_.writelines(_)
