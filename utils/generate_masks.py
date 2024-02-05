import numpy as np
import os
import mrcfile
import cv2
import glob
import pandas as pd

MRC_FILE_LOCATION = "finetune_dataset/mrc_files/"
CSV_FILE_LOCATION = "finetune_dataset/csv_files/"
MASK_FILE_LOCATION = "finetune_dataset/mask_files/"
STAR_FILE_LOCATION = "finetune_dataset/sample.star"

coordinates_ = pd.read_csv(f'{STAR_FILE_LOCATION}', skiprows = 6)
records = coordinates_["_rlnCoordinateY #3"]
columns_names = ['Micrographs Filename', 'X-Coordinate', 'Y-Coordinate']
df = pd.DataFrame()
micrograph_filename = []
x_coordinate = []
y_coordinate = []
for i in range(len(records)):
    try:
        values = records[i].split("\t")
    except:
        values = records[i].split(" ")
    micrograph_filename.append(values[0])
    x_coordinate.append(int(float(values[1])))
    y_coordinate.append(int(float(values[2])))
df.insert(0, columns_names[0], micrograph_filename)
df.insert(1, columns_names[1], x_coordinate)
df.insert(2, columns_names[2], y_coordinate)

try:
    os.makedirs(CSV_FILE_LOCATION)
except:
    pass

diameter = int(input("Please, enter diameter of protein in pixels \n"))
files = df['Micrographs Filename'].unique()
for f in files:
    f_name = f[:-4]
    df_box = df[df['Micrographs Filename'] == f]
    df_coord = df_box.loc[:, ['X-Coordinate', 'Y-Coordinate']]
    df_new = pd.DataFrame()
    col_names = ['X-Coordinate', 'Y-Coordinate', 'Diameter']
    x_coordinates = []
    y_coordinates = []
    diameters = []
    for index, row in df_coord.iterrows():
        x, y = row["X-Coordinate"], row["Y-Coordinate"]
        x_coordinates.append(x)
        y_coordinates.append(y)
        diameters.append(diameter)
    df_new.insert(0, col_names[0], x_coordinates)
    df_new.insert(1, col_names[1], y_coordinates)
    df_new.insert(2, col_names[2], diameters)
    df_new.to_csv(f"{CSV_FILE_LOCATION}{f_name}.csv", index=False)
    
coordinate_files = glob.glob(f"{CSV_FILE_LOCATION}*.csv")
for cf in coordinate_files:
    f_name = cf.split("/")[-1][:-4]
    micrograph_filename = f"{MRC_FILE_LOCATION}{f_name}.mrc"
    image = mrcfile.read(micrograph_filename)
    image = image.T
    image = np.rot90(image)

    mask = np.zeros_like(image)
    try:
        coordinates = pd.read_csv(cf, usecols=[0,1,2])
        for i, c in coordinates.iterrows():
            x = c['X-Coordinate']
            y = c['Y-Coordinate']
            r = int(c['Diameter']/2)
            coords = cv2.circle(mask, (x, y), r, (255, 255, 255), -1)
        # cv2.imwrite(f"{MASK_FILE_LOCATION}{f_name}_mask.jpg", coords)
        cv2.imwrite(f"finetune_dataset/sample_mask.jpg", coords)
        print('Success')
    except:
        print('Error Creating Mask')

