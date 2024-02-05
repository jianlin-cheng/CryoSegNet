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
import config


def evaluation_metrics(gt_path, comparison_mask_path):
    mask = cv2.imread(gt_path, 0)    
    comparison_mask = cv2.imread(comparison_mask_path, 0)
    
    mask = mask/255    
    comparison_mask = comparison_mask / 255
    
    smooth = 0.001  
    pred_mask = copy.deepcopy(comparison_mask) 

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

def evaluation(gt_path, model):
    total_precisions = []
    total_recalls = []
    total_dice_scores = []
    
    for i in range(0, len(gt_path), 1):
        g_path = gt_path[i]
        c_path = g_path.replace("Groundtruth", f"General/{model}")
        
        try:
            precision, recall, dice_score = evaluation_metrics(g_path, c_path)
        except:
            pass
        
        total_precisions.append(precision)
        total_recalls.append(recall)
        total_dice_scores.append(dice_score)
       
    precision = np.round(np.max(total_precisions, axis = 0), 3)
    recall = np.round(np.max(total_recalls, axis = 0), 3)
    f1_score = np.round((2 * precision * recall) / (precision + recall), 3)
    dice_score = np.round(np.max(total_dice_scores, axis = 0), 3)


    print(f"For {model}:")
    print("\tPrecision", precision)
    print("\tRecall", recall)
    print("\tF1-Score", f1_score)
    print("\tDice Score", dice_score)
    

empiar_ids = [10028, 10081, 10345, 11056, 10532, 10093, 10017]
print("[INFO] Loading up test images path ...")
for empiar_id in empiar_ids:
    gt_path = list(glob.glob(f"{config.test_dataset_path}/{empiar_id}/masks/*.jpg"))[20:]
    print("\n")
    print(f"Evaluation Results for EMPIAR ID {empiar_id}")
    evaluation(gt_path, model = "CrYOLO")
    evaluation(gt_path, model = "Topaz")
    evaluation(gt_path, model = "CryoSegNet")
    print(f"--------------------------------------")