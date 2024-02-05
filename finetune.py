# Code for training

from utils.accuracy import dice_score, jaccard_score
from dataset.dataset import CryoEMDataset, CryoEMFineTuneDataset
from models.model_5_layers import UNET
import numpy as np
import config
import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch.nn.functional as F
from utils.loss import DiceLoss
import glob
from tqdm import tqdm
import time
from datetime import datetime, date
import os

mask_path = list(glob.glob(config.train_dataset_path + 'masks/*.jpg'))
num_masks = len(mask_path)
train_val_split = round(0.8*num_masks)
train_mask_path = mask_path[:train_val_split]
val_mask_path = mask_path[train_val_split:]

train_ds = CryoEMFineTuneDataset(mask_dir=train_mask_path, transform=None)
val_ds = CryoEMFineTuneDataset(mask_dir=val_mask_path, transform=None)

print(f"[INFO] Found {len(train_ds)} examples in the training set...")
print(f"[INFO] Found {len(val_ds)} examples in the validation set...")

train_loader = DataLoader(train_ds, shuffle=True, batch_size=config.batch_size, pin_memory=config.pin_memory, num_workers=config.num_workers)
val_loader = DataLoader(val_ds, shuffle=True, batch_size=config.batch_size, pin_memory=config.pin_memory, num_workers=config.num_workers)
print(f"[INFO] Train Loader Length {len(train_loader)}...")

# initialize our UNet model
model = UNET().to(device=config.device)
state_dict = torch.load(config.cryosegnet_checkpoint)
model.load_state_dict(state_dict)


# initialize loss function and optimizer
criterion1 = BCEWithLogitsLoss()
criterion2 = DiceLoss()
optimizer = Adam(model.parameters(), lr=config.learning_rate)


# calculate steps per epoch for training and test set
train_steps = len(train_ds) // config.batch_size
val_steps = len(val_ds) // config.batch_size
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"[INFO] Number of Training Steps : {train_steps}")
print(f"[INFO] Number of Validation Steps : {val_steps}")
#print(f"[INFO] Total Number of Parameters : {total_params}")

# initialize a dictionary to store training history
H = {"train_loss": [], "val_loss": [], "train_dice_score": [], "val_dice_score": [], "train_jaccard_score": [], "val_jaccard_score": [], "epochs": []}

best_val_loss = float("inf")
# loop over epochs
print("[INFO] Training the network...")
start_time = time.time()
for e in tqdm(range(config.num_epochs)):
    model.train()
    
    train_loss = 0
    train_dice_scores = []
    train_jaccard_scores = []
    # loop over the training set

    for i, data in enumerate(train_loader):
        x, y = data
        x, y = x.to(config.device), y.to(config.device)

        optimizer.zero_grad()
        
        pred = model(x)
        loss1 = criterion1(pred, y) 
        loss2 = criterion2(nn.Sigmoid()(pred), y)
        loss = (loss1 + loss2)/2
        loss.backward()
        optimizer.step()
        
        # Accumulate the train loss
        train_loss += loss.item() * 1.0
        
        pred = nn.Sigmoid()(pred)
        train_dice_scores.append(dice_score(y, pred).item())
        train_jaccard_scores.append(jaccard_score(y, pred).item())
        
    # Calculate train loss
    train_loss /= len(train_loader)
    train_dice_score = np.mean(train_dice_scores)
    train_jaccard_score = np.mean(train_jaccard_scores)
    
    val_loss = 0    
    val_dice_scores = [] 
    val_jaccard_scores = []
    
    model.eval()
    with torch.no_grad(): 
        for i, data in enumerate(val_loader):
            x, y = data
            x, y = x.to(config.device), y.to(config.device)
            
            
            pred = model(x)
            loss = criterion2(nn.Sigmoid()(pred), y)
            
            # Accumulate the validation loss
            val_loss += loss.item() * 1
            
            pred = nn.Sigmoid()(pred)
            
            # Accumulate the val dice scores and jaccard scores
            val_dice_scores.append(dice_score(y, pred).item())
            val_jaccard_scores.append(jaccard_score(y, pred).item())

    # Calculate validation loss
    val_loss /= len(val_loader)
    val_dice_score = np.mean(val_dice_scores)
    val_jaccard_score = np.mean(val_jaccard_scores)
    
    # update our training history
    H["train_loss"].append(train_loss)
    H["val_loss"].append(val_loss)
    H["train_dice_score"].append(train_dice_score)
    H["train_jaccard_score"].append(train_jaccard_score)
    H["val_dice_score"].append(val_dice_score)
    H["val_jaccard_score"].append(val_jaccard_score)
    H["epochs"].append(e + 1)
    
    # print the model training and validation information
    print("[INFO] EPOCH: {}/{}".format(e + 1, config.num_epochs))
    print("Train Loss: {:.4f}, Validation Loss: {:.4f}, Train Dice Score: {:.4f}. Validation Dice Score: {:.4f}".format(
    train_loss, val_loss, train_dice_score, val_dice_score))
    
    
    # serialize the model to disk
    if e %2 == 0:
        MODEL_PATH = "Finetuned model Date: {}.pth".format(date.today())
        torch.save(model.state_dict(), os.path.join(f"{config.output_path}/models/", MODEL_PATH))
        
    if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(f"{config.output_path}/models/", "finetuned_cryosegnet_best_val_loss.pth"))

# display the total time needed to perform the training
end_time = time.time()
print("[INFO] total time taken to train the model: {:.2f}s".format(
    end_time - start_time))

