# Code for generating star file

from denoise import denoise
import config
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
import csv
import glob
import random
from dataset import transform, min_max
from model_5_layers import UNET
import config
from tqdm import tqdm
import mrcfile
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import statistics as st

DEVICE = 'cuda:0'

sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=DEVICE)

mask_generator = SamAutomaticMaskGenerator(sam)

def get_annotations(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    
    return img

def generate_output(model, image_path, star_writer):
    # set model to evaluation mode
    model.eval()
    # turn off gradient tracking
    with torch.no_grad():
        image = cv2.imread(image_path, 0)
        # image = denoise(image_path)
        height, width = image.shape
        image = cv2.resize(image, (config.INPUT_IMAGE_WIDTH, config.INPUT_IMAGE_HEIGHT))
        
        image = torch.from_numpy(image).unsqueeze(0).float()
        image = image / 255.0
        image = image.to(DEVICE).unsqueeze(0)
        
        predicted_mask = model(image)       
        predicted_mask = torch.sigmoid(predicted_mask)
        predicted_mask = predicted_mask.cpu().numpy().reshape(config.INPUT_IMAGE_WIDTH, config.INPUT_IMAGE_HEIGHT)
        predicted_mask = np.rot90(predicted_mask, k=3)
        predicted_mask = predicted_mask.T
                
        sam_output = np.repeat(transform(predicted_mask)[:,:,None], 3, axis=-1)
        predicted_mask = cv2.resize(predicted_mask, (width, height))
        predicted_mask = min_max(predicted_mask) 
        
        masks = mask_generator.generate(sam_output)
        sam_mask = get_annotations(masks)
        sam_mask = cv2.resize(sam_mask, (width, height) )
        
        
        bboxes = {"bbox": [], "iou": []}
        for i in range(0, len(masks)):
            # if masks[i]["predicted_iou"] > 0.94:
                bboxes["bbox"].append(masks[i]["bbox"])
                bboxes["iou"].append(masks[i]["predicted_iou"])
        
        if len(bboxes) > 1:
            x_ = st.mode([box[2] for box in bboxes["bbox"]])
            y_ = st.mode([box[3] for box in bboxes["bbox"]])
            d_ = np.sqrt((x_ * width / 1024)**2 + (y_ * height / 1024)**2)
            r_ = int(d_//2)
            th = r_ * 0.2     

            filename = image_path.split("/")[-1][:-4] + '.mrc'
            for i in range(len(bboxes["bbox"])):
                box, iou = bboxes["bbox"][i], bboxes["iou"][i] 
                if box[2] < x_ + th and box[2] > x_ - th/3 and box[3] < y_ + th and box[3] > y_ - th/3:                 
                    x_new, y_new = int((box[0] + box[2]/2) / 1024 * width) , int((box[1] + box[3]/2) / 1024 * height)
                    star_writer.writerow([filename, x_new, y_new, 2*r_]) 
                    if iou > 0.99:
                        star_writer.writerow([filename, x_new + random.randint(0, int(th/2)), y_new + random.randint(0, int(th/2)), 2*r_])   
                        star_writer.writerow([filename, x_new + random.randint(-int(th/2), 0), y_new + random.randint(-int(th/2), 0), 2*r_])            
                                                                                                                                                                     
        else:
            pass
        
print("[INFO] Loading up Test Micrographs ...")
images_path = list(glob.glob("/bml/Rajan_CryoEM/Processed_Datasets/New_Val/N10345/images/*.jpg"))
# images_path = list(glob.glob("/bml/Rajan_CryoEM/Final_Validation_Full_Micrographs/10025/micrographs/*.mrc"))

print(f"[INFO] Number of Micrographs = {len(images_path)}\n")
print("[INFO] Loading the model...")
file_name = input("\nProvide the file name for storing output in .star format | Example: output.star\n")
print("[INFO] Generating star file for input Cryo-EM Micrographs...")
print("[INFO] Generation may take more time depending upon the number of micrographs...\n")

model = UNET().to(DEVICE)
state_dict = torch.load('output/results_final_model_5_layers_july_29/models/CryoPPP denoised with BCE & Dice Loss Att-UNET 5 layers Batchsize: 6,  InputShape: 1024, LR 0.0001, Server: Daisy Epochs: 120, Date: 2023-07-28.pth')
model.load_state_dict(state_dict)

with open(file_name, "w") as star_file:
    star_writer = csv.writer(star_file, delimiter=' ')
    star_writer.writerow([])
    star_writer.writerow(["data_"])
    star_writer.writerow([])
    star_writer.writerow(["loop_"])
    star_writer.writerow(["_rlnMicrographName", "#1"])
    star_writer.writerow(["_rlnCoordinateX", "#2"])
    star_writer.writerow(["_rlnCoordinateY", "#3"])
    star_writer.writerow(["_rlnDiameter", "#4"])

    for i in tqdm(range(0, len(images_path), 1)):
        generate_output(model, images_path[i], star_writer)
    
