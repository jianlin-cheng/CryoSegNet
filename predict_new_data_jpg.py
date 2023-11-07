# Code for making predictions on individual micrographs

import copy
from utils.denoise import denoise, denoise_jpg_image
import config
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
import glob
import os
from dataset.dataset import transform
from models.model_5_layers import UNET
import config
import mrcfile
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import statistics as st

print("[INFO] Loading up model...")
model = UNET().to(device=config.device)
state_dict = torch.load(config.cryosegnet_checkpoint)
model.load_state_dict(state_dict)

sam_model = sam_model_registry[config.model_type](checkpoint=config.sam_checkpoint)
sam_model.to(device=config.device)

mask_generator = SamAutomaticMaskGenerator(sam_model)

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

def prepare_plot(image, predicted_mask, sam_mask, coords, image_path):
    plt.figure(figsize=(40, 30))
    plt.subplot(221)
    plt.title('Testing Image', fontsize=14)
    plt.imshow(image, cmap='gray')
    plt.subplot(222)
    plt.title('Attention-UNET Mask', fontsize=14)
    plt.imshow(predicted_mask, cmap='gray')
    plt.subplot(223)
    plt.title('SAM Mask', fontsize=14)
    plt.imshow(sam_mask, cmap='gray')
    plt.subplot(224)
    plt.title('Final Picked Particles', fontsize=14)
    plt.imshow(coords, cmap='gray')
    path = image_path.split("/")[-1]
    path = path.replace(".jpg", "_result.jpg")
    #plt.savefig(os.path.join(f"{config.output_path}/results/", path))
    final_path = os.path.join(f"{config.output_path}/results/", f'predicted_{path}')
    cv2.imwrite(final_path, coords)




def make_predictions(model, image_path):
    # set model to evaluation mode
    model.eval()
    with torch.no_grad():
        # image = mrcfile.read(image_path)
        # image = image.T
        # image = np.rot90(image)
        image = cv2.imread(image_path, 0)
        image = denoise_jpg_image(image)
        height, width = image.shape
        orig_image = copy.deepcopy(image)
        image = cv2.resize(image, (config.input_image_width, config.input_image_height))    
        segment_mask = copy.deepcopy(orig_image)
        
        image = torch.from_numpy(image).unsqueeze(0).float()
        image = image / 255.0
        image = image.to(config.device).unsqueeze(0)
        
        
        predicted_mask = model(image)    
     
        predicted_mask = torch.sigmoid(predicted_mask)
        predicted_mask = predicted_mask.cpu().numpy().reshape(config.input_image_width, config.input_image_height)

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
            d_ = np.sqrt((x_ * width / config.input_image_width)**2 + (y_ * height / config.input_image_height)**2)
            r_ = int(d_//2)
            th = r_ * 0.20
            segment_mask = cv2.cvtColor(segment_mask, cv2.COLOR_GRAY2BGR)
            for b in bboxes:
                if b[2] < x_ + th and b[2] > x_ - th/3 and b[3] < y_ + th and b[3] > y_ - th/3: 
                    x_new, y_new = int((b[0] + b[2]/2) / config.input_image_width * width) , int((b[1] + b[3]/2) / config.input_image_height * height)
                    coords = cv2.circle(segment_mask, (x_new, y_new),  r_, (0, 0, 255), 8)
            try:
                prepare_plot(orig_image, predicted_mask, sam_mask, coords, image_path)
            except:
                pass
        else:
            pass
        
print("[INFO] Loading up test images path ...")
images_path = list(glob.glob(f"{config.my_dataset_path}/*.jpg"))

for i in range(0, len(images_path), 1):
	make_predictions(model, images_path[i])