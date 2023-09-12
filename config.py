# Configuration file
# Specify parameters as per your need

import torch

# please provide path of the training dataset
DATASET_PATH = ""
BASE_OUTPUT = "output"

VALIDATION_SPLIT = 0.2

# specify cuda
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
PIN_MEMORY = True if DEVICE == "cuda:0" else False
NUM_WORKERS = 8

NUM_CHANNELS = 1
NUM_CLASSES = 1
NUM_LEVELS = 3

LR = 0.0001
NUM_EPOCHS = 200
BATCH_SIZE = 6

INPUT_IMAGE_WIDTH = 1024
INPUT_IMAGE_HEIGHT = INPUT_IMAGE_WIDTH
INPUT_SHAPE = INPUT_IMAGE_WIDTH


LOG = False

ARCHITECTURE_NAME = "CryoSegNet Training Batchsize: {},  InputShape: {}, LR {}".format(BATCH_SIZE, INPUT_SHAPE, LR)
