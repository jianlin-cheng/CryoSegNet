# Code for calculating different evaluation metrics like Precision, Recall, F1-Score and Dice Score.

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
import copy
import config
import numpy as np
import cv2
import glob
from utils.predict import predict
from utils.generate_csv_for_evaluation import generate_csv_for_evaluation
from utils.generate_masks_for_evaluation import generate_masks_for_evaluation


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
    print("[INFO] Calculating evaluation metrics ...")
    total_precisions = []
    total_recalls = []
    total_dice_scores = []
    
    for i in range(0, len(gt_path), 1):
        g_path = gt_path[i]
        c_path = g_path.replace("test_dataset", f"Evaluation/{model}")  
        try:
            precision, recall, dice_score = evaluation_metrics(g_path, c_path)
            total_precisions.append(precision)
            total_recalls.append(recall)
            total_dice_scores.append(dice_score)
        except:
            pass
       
    precision = np.round(np.max(total_precisions, axis = 0), 3)
    recall = np.round(np.max(total_recalls, axis = 0), 3)
    f1_score = np.round((2 * precision * recall) / (precision + recall), 3)
    dice_score = np.round(np.max(total_dice_scores, axis = 0), 3)


    print(f"For {model}:")
    print("\tPrecision", precision)
    print("\tRecall", recall)
    print("\tF1-Score", f1_score)
    print("\tDice Score", dice_score)
    

def precision_recall():
    star_file = predict(config.test_dataset_path, config.empiar_id)
    generate_csv_for_evaluation(config.empiar_id, star_file)
    generate_masks_for_evaluation(config.empiar_id)
    gt_path = list(glob.glob(f"{config.test_dataset_path}/{config.empiar_id}/masks/*.jpg"))[20:]
    print("\n")
    print(f"Evaluation Results for EMPIAR ID {config.empiar_id}")
    evaluation(gt_path, model = "CryoSegNet")
    print(f"--------------------------------------")
    
precision_recall()
