# Code for calculating different evaluation metrics like Precision, Recall, F1-Score and Dice Score.

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
import copy
import config
import numpy as np
import torch
import cv2
import glob
from models.model_5_layers import UNET
import config
import mrcfile

print("[INFO] Loading up model...")
model = UNET().to(device=config.device)
state_dict = torch.load(config.cryosegnet_checkpoint)
model.load_state_dict(state_dict)


def evaluation_metrics(model, image_path, threshold):
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
        
        smooth = 0.001  
        pred_mask = copy.deepcopy(predicted_mask)
        pred_mask[pred_mask >= threshold] = 1.0
        pred_mask[pred_mask < threshold] = 0         

        true_positive = np.sum(np.logical_and(mask, pred_mask))
        false_positive = np.sum(np.logical_and(np.logical_not(mask), pred_mask))
        false_negative = np.sum(np.logical_and(mask, np.logical_not(pred_mask)))

        # Calculate precision, recall, f1, dice score
        precision = (true_positive  + smooth) / (true_positive + false_positive + smooth)
        recall = (true_positive  + smooth) / (true_positive + false_negative + smooth)
        
        inputs = mask.reshape(-1)
        targets = pred_mask.reshape(-1)

        intersection = (inputs * targets).sum()
        dice_score = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth) 
             
        return precision, recall, dice_score

def evaluation(model, images_path, empiar_id, threshold=0.1):
    total_precisions = []
    total_recalls = []
    total_dice_scores = []

    for i in range(0, len(images_path), 1):
        precision, recall, dice_score = evaluation_metrics(model, images_path[i], threshold)
        total_precisions.append(precision)
        total_recalls.append(recall)
        total_dice_scores.append(dice_score)
        
    precision = np.round(np.max(total_precisions, axis = 0), 3)
    recall = np.round(np.max(total_recalls, axis = 0), 3)  
    f1_score = np.round((2 * precision * recall) / (precision + recall), 3)
    dice_score = np.round(np.max(total_dice_scores, axis = 0), 3) 
    
    print("\tPrecision", precision)
    print("\tRecall", recall)
    print("\tF1-Score", f1_score)
    print("\tDice Score", dice_score)
    

empiar_ids = [10028, 10081, 10345, 11056, 10532, 10093, 10017]
print("[INFO] Loading up test images path ...")
for empiar_id in empiar_ids:
    images_path = list(glob.glob(f"{config.test_dataset_path}/{empiar_id}/images/*.jpg"))
    print("\n")
    print(f"Evaluation Results for EMPIAR ID {empiar_id}")
    evaluation(model, images_path, empiar_id)
    print(f"--------------------------------------")
