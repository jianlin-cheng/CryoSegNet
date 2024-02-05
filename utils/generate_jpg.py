from denoise import denoise, denoise_jpg_image
import glob
import cv2

files = glob.glob("finetune_dataset/mrc_files/*.mrc")

for f in files:
    image = denoise(f)
    cv2.imwrite(f"finetune_dataset/images/{f.split('/')[-1][:-4]}.jpg", image)

