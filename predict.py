# Code for making predictions on individual micrographs

import copy
from denoise import denoise
import config
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
import glob
import os
from dataset import transform
from model_5_layers import UNET
import config
import mrcfile
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import statistics as st

DEVICE="cuda:0"
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

def prepare_plot(image, mask, predicted_mask, sam_mask, coords, image_path):
    plt.figure(figsize=(40, 30))
    plt.subplot(231)
    plt.title('Testing Image', fontsize=14)
    plt.imshow(image, cmap='gray')
    plt.subplot(232)
    plt.title('Original Mask', fontsize=14)
    plt.imshow(mask, cmap='gray')
    plt.subplot(234)
    plt.title('Attention-UNET Mask', fontsize=14)
    plt.imshow(predicted_mask, cmap='gray')
    plt.subplot(235)
    plt.title('SAM Mask', fontsize=14)
    plt.imshow(sam_mask, cmap='gray')
    plt.subplot(236)
    plt.title('Final Picked Particles', fontsize=14)
    plt.imshow(coords, cmap='gray')
    path = image_path.split("/")[-1]
    path = path.replace(".jpg", "_result.jpg")
    plt.savefig(os.path.join("output/validation/results/", path))
    final_path = os.path.join("output/validation/results/", f'final3__{path}')
    cv2.imwrite(final_path, coords)




def make_predictions(model, image_path):
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
        height, width = image.shape
        orig_image = copy.deepcopy(image)
        orig_mask = copy.deepcopy(mask)
        image = cv2.resize(image, (config.INPUT_IMAGE_WIDTH, config.INPUT_IMAGE_HEIGHT))        
        mask = cv2.resize(mask, (config.INPUT_IMAGE_WIDTH, config.INPUT_IMAGE_HEIGHT))
        segment_mask = copy.deepcopy(orig_image)
        
        image = torch.from_numpy(image).unsqueeze(0).float()
        image = image / 255.0
        image = image.to(DEVICE).unsqueeze(0)
        
        mask = cv2.imread(mask_path, 0)
        mask = cv2.resize(mask, (config.INPUT_IMAGE_WIDTH, config.INPUT_IMAGE_HEIGHT))
        
        predicted_mask = model(image)    
     
        predicted_mask = torch.sigmoid(predicted_mask)
        predicted_mask = predicted_mask.cpu().numpy().reshape(config.INPUT_IMAGE_WIDTH, config.INPUT_IMAGE_HEIGHT)
        
        print("Path | Min Max Mean", predicted_mask.min(), predicted_mask.max(), predicted_mask.mean())

        sam_output = np.repeat(transform(predicted_mask)[:,:,None], 3, axis=-1)
        predicted_mask = cv2.resize(predicted_mask, (width, height))
        
        masks = mask_generator.generate(sam_output)
        sam_mask = get_annotations(masks)
        sam_mask = cv2.resize(sam_mask, (width, height) )
        
        
        
        bboxes = []
        for i in range(0, len(masks)):
            if masks[i]["predicted_iou"] > 0.94:
                box = masks[i]["bbox"]
                bboxes.append(box)
        
        if len(bboxes) > 1:
        
            x_ = st.mode([box[2] for box in bboxes])
            y_ = st.mode([box[3] for box in bboxes])
            d_ = np.sqrt((x_ * width / 1024)**2 + (y_ * height / 1024)**2)
            r_ = int(d_//2)
            th = r_ * 0.20
            segment_mask = cv2.cvtColor(segment_mask, cv2.COLOR_GRAY2BGR)
            for b in bboxes:
                if b[2] < x_ + th and b[2] > x_ - th/3 and b[3] < y_ + th and b[3] > y_ - th/3: 
                    x_new, y_new = int((b[0] + b[2]/2) / 1024 * width) , int((b[1] + b[3]/2) / 1024 * height)
                    coords = cv2.circle(segment_mask, (x_new, y_new),  r_, (0, 0, 255), 8)
            try:
                prepare_plot(orig_image, orig_mask, predicted_mask, sam_mask, coords, image_path)
            except:
                pass
        else:
            pass
        
print("[INFO] Loading up test images path ...")
images_path = list(glob.glob("/bml/Rajan_CryoEM/Processed_Datasets/New_Train/10406/test/images/FoilHole_3165074_Data_3165619_3165620_20190104_1807-61111.jpg"))
print("[INFO] Loading up model...")
print(images_path)

model = UNET().to(DEVICE)
state_dict = torch.load('output/results_final_model_5_layers_july_29/models/CryoPPP denoised with BCE & Dice Loss Att-UNET 5 layers Batchsize: 6,  InputShape: 1024, LR 0.0001, Server: Daisy Epochs: 120, Date: 2023-07-28.pth')
model.load_state_dict(state_dict)

for i in range(0, len(images_path), 1):
	make_predictions(model, images_path[i])