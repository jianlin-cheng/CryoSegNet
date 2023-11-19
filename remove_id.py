import os
import glob
import config
import shutil


files = glob.glob(config.my_dataset_path + '/*.mrc')
for file in files:
    try:
        f = file.split("/")[-1][22:]
        directory = os.path.join(*[p for p in file.split("/")[:-1]])
        destination_filename = f"{directory}/{f}"
        shutil.move(file, destination_filename)
    except:
        print("Error renaming file")