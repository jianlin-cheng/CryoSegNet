# Code for calculating precision recall curve
import copy
import config
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
import glob
import os
from model_5_layers import UNET
import config
import mrcfile

print("[INFO] Loading up model...")
model = UNET().to(device=config.device)
state_dict = torch.load(config.cryosegnet_checkpoint)
model.load_state_dict(state_dict)

def plot_precision_recall(recalls, precisions, empiar_ids):
    title_ = f'Precision-Recall Curve'
    plt.figure(figsize=(8, 6), dpi=600)
    for j in range(len(empiar_ids)): 
        plt.plot(recalls[j], precisions[j], label=f'EMPIAR {empiar_ids[j]}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.title(title_)
    plt.grid(True)    
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.savefig(os.path.join(config.output_path, title_ + '_.png'))


def calculate_precision_recall(model, image_path):
    # set model to evaluation mode
    model.eval()
    with torch.no_grad():
        mask_path = image_path.replace("images", "masks")
        mask_path = mask_path.replace(".jpg", "_mask.jpg")   
        # image = mrcfile.read(image_path)
        # image = image.T
        # image = np.rot90(image)
        image = cv2.imread(image_path, 0)
        mask = cv2.imread(mask_path, 0)
        
        image = cv2.resize(image, (config.input_image_width, config.input_image_height))        
        mask = cv2.resize(mask, (config.input_image_width, config.input_image_height))
        mask = mask/255
        
        image = torch.from_numpy(image).unsqueeze(0).float()
        image = image / 255.0
        image = image.to(config.device).unsqueeze(0)
        
        predicted_mask = model(image)         
        predicted_mask = torch.sigmoid(predicted_mask)
        predicted_mask = predicted_mask.cpu().numpy()
        
        precisions = []
        recalls = []
        
        thresholds = np.linspace(0, 1, 100)
        smooth = 1
        for th in thresholds:
            pred_mask = copy.deepcopy(predicted_mask)
            pred_mask[pred_mask >= th] = 1.0
            pred_mask[pred_mask < th ] = 0 
            

            true_positive = np.sum(np.logical_and(mask, pred_mask))
            false_positive = np.sum(np.logical_and(np.logical_not(mask), pred_mask))
            false_negative = np.sum(np.logical_and(mask, np.logical_not(pred_mask)))

            # Calculate precision and recall
            precision = (true_positive  + smooth)/ (true_positive + false_positive + smooth)
            recall = (true_positive  + smooth) / (true_positive + false_negative + smooth)
            precisions.append(precision)
            recalls.append(recall)
             
        return precisions, recalls

empiar_ids = [10028, 10081, 10345, 11056, 10532, 10093, 10017]
precisions_ = []
recalls_ = []
for empiar_id in empiar_ids:
    print("[INFO] Loading up test images path ...")
    images_path = list(glob.glob(f"{config.test_dataset_path}/{empiar_id}/images/*.jpg"))

    
    total_precisions = []
    total_recalls = []
    for i in range(0, len(images_path), 1):
        precisions, recalls = calculate_precision_recall(model, images_path[i])
        total_precisions.append(precisions)
        total_recalls.append(recalls)
        
    precision = np.max(total_precisions, axis = 0)
    recall = np.max(total_recalls, axis = 0)  
    precisions_.append(precision)
    recalls_.append(recall)
    
plot_precision_recall(recalls_, precisions_, empiar_ids)
