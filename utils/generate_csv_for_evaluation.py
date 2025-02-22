import glob
import pandas as pd
import numpy as np
import cv2
import os
import mrcfile

empiar_ids = [10028, 10081, 10345, 11056, 10532, 10093, 10017]
diameters = [224, 154, 149, 164, 174, 172, 108]

empiar_diameter_dict = dict(zip(empiar_ids, diameters))

        
        
def generate_csv_for_evaluation(empiar_id, star_file):
    image = cv2.imread(glob.glob(f'test_dataset/{empiar_id}/images/*.jpg')[0])
    try:
        a, b, _ = image.shape
    except:
        a, b = image.shape
    original_df = pd.read_csv(f'{star_file}', skiprows = 7)
    destination_dir = f'Evaluation/CryoSegNet/{empiar_id}/coordinates/'
    try:
        os.makedirs(destination_dir)
    except:
        pass

    records = original_df['_rlnDiameter #4']
    columns_names = ['Micrographs Filename', 'X-Coordinate', 'Y-Coordinate']
    df = pd.DataFrame()
    micrograph_filename = []
    x_coordinate = []
    y_coordinate = []
    for i in range(len(records)):
        values = records[i].split(" ")
        micrograph_filename.append(values[0])
        x_coordinate.append(int(float(values[1])))
        y_coordinate.append(int(float(values[2])))
    df.insert(0, columns_names[0], micrograph_filename)
    df.insert(1, columns_names[1], x_coordinate)
    df.insert(2, columns_names[2], y_coordinate)
    df.to_csv(f"Evaluation/CryoSegNet/{empiar_id}/{empiar_id}.csv", index = False)
    
    df['Diameter'] = np.array([empiar_diameter_dict[empiar_id] for _ in range(len(df['X-Coordinate']))])
    files = df['Micrographs Filename'].unique()
    for f in files:
        f_name = f[:-4]
        df_box = df[df['Micrographs Filename'] == f]
        df_coord = df_box.loc[:, ['X-Coordinate', 'Y-Coordinate', 'Diameter']]
        df_new = pd.DataFrame()
        col_names = ['X-Coordinate', 'Y-Coordinate', 'Diameter']
        x_coordinates = []
        y_coordinates = []
        diameter = []
        for index, row in df_coord.iterrows():
            temp_img = np.zeros((b, a))
            x, y = row["X-Coordinate"], row["Y-Coordinate"]
            temp_img[int(x), int(y)] = 1
    
            temp_img = np.rot90(temp_img, k = 1)
            temp_img = temp_img.T

            max_index = np.argmax(temp_img)
            max_index_2d = np.unravel_index(max_index, temp_img.shape)
            x_coordinates.append(max_index_2d[0])
            y_coordinates.append(max_index_2d[1])
            diameter.append(empiar_diameter_dict[empiar_id])
        df_new.insert(0, col_names[0], x_coordinates)
        df_new.insert(1, col_names[1], y_coordinates)
        df_new.insert(2, col_names[2], diameter)
        df_new.to_csv(f"{destination_dir}{f_name}.csv", index=False)
