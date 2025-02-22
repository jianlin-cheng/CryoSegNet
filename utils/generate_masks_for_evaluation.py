import os
import numpy as np
import pandas as pd
import mrcfile
import cv2
import glob
from matplotlib import pyplot as plt
import math
import shutil


def generate_masks_for_evaluation(empiar_id):
    files = glob.glob(f"test_dataset/{empiar_id}/masks/*.jpg")
    
    destination_dir = f'Evaluation/CryoSegNet/{empiar_id}/masks/'
    try:
        os.makedirs(destination_dir)
    except:
        pass
    
    for n, file in enumerate(files):
        
        f = file.split('/')[-1][:-9]
        micrograph_filename = f"test_dataset/{empiar_id}/images/{f}.jpg"
        coordinate_filename = f"Evaluation/CryoSegNet/{empiar_id}/coordinates/{f}.csv"

        # image = mrcfile.read(micrograph_filename)
        
        # image = image.T
        # image = np.rot90(image)
        image = cv2.imread(micrograph_filename)

        mask = np.zeros_like(image)
        try:
            coordinates = pd.read_csv(coordinate_filename, usecols=[0,1,2])
            for i, c in coordinates.iterrows():
                x = c['X-Coordinate']
                y = c['Y-Coordinate']
                r = int(c['Diameter']/2)
                coords = cv2.circle(mask, (x, y), r, (255, 255, 255), -1)
            cv2.imwrite(f"{destination_dir}{f}_mask.jpg", coords)
        except:
            print('Error Creating Mask')


        
        

    

