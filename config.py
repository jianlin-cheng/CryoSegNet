# Configuration file

import argparse
import torch

# Create an argument parser
parser = argparse.ArgumentParser(description="CryoSegNet Training")

# Add arguments
parser.add_argument("--train_dataset_path", type=str, default="train_dataset/*/", help="Path to the training dataset")
parser.add_argument("--test_dataset_path", type=str, default="test_dataset", help="Path to the test dataset")
parser.add_argument("--output_path", type=str, default="output", help="Output directory")

# Device-related arguments
parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="Device (cuda:0 or cpu)")
parser.add_argument("--pin_memory", action="store_true", help="Enable pin_memory for data loading if using CUDA")
parser.add_argument("--num_workers", type=int, default=8, help="Number of data loading workers")

# Model-related arguments
parser.add_argument("--num_channels", type=int, default=1, help="Number of input channels")
parser.add_argument("--num_classes", type=int, default=1, help="Number of classes")
parser.add_argument("--num_levels", type=int, default=3, help="Number of levels in the model")

# Training-related arguments
parser.add_argument("--learning_rate", type=float, default=0.0001, help="Learning rate")
parser.add_argument("--num_epochs", type=int, default=200, help="Number of training epochs")
parser.add_argument("--batch_size", type=int, default=6, help="Batch size")

# Input image size
parser.add_argument("--input_image_width", type=int, default=1024, help="Input image width")
parser.add_argument("--input_image_height", type=int, default=1024, help="Input image height")
parser.add_argument("--input_shape", type=int, default=1024, help="Input image shape")

# Logging-related arguments
parser.add_argument("--logging", action="store_true", help="Enable logging")

# Model related arguments
parser.add_argument("--cryosegnet_checkpoint", type=str, default="pretrained_models/cryosegnet.pth", help="Path to CryoSegNet checkpoint")
parser.add_argument("--sam_checkpoint", type=str, default="pretrained_models/sam_vit_h_4b8939.pth", help="Path to SAM checkpoint")
parser.add_argument("--model_type", type=str, default="vit_h", help="SAM Model type")

# Additional Arguments for prediction
parser.add_argument("--empiar_id", type=int, default=10081, help="EMPIAR ID for prediction")
parser.add_argument("--file_name", type=str, default="10081.star", help="Filename for picked proteins coordinates")

# Additional info in architecture name
architecture_name = "CryoSegNet Model with Batchsize: {}, InputShape: {}, LR {}".format(
    parser.parse_args().batch_size,
    parser.parse_args().input_shape,
    parser.parse_args().learning_rate
)
parser.add_argument("--architecture_name", type=str, default=architecture_name, help="Model architecture name")

# Parse the command-line arguments
args = parser.parse_args()

# Access the parsed arguments
train_dataset_path = args.train_dataset_path
test_dataset_path = args.test_dataset_path
output_path = args.output_path
device = args.device
pin_memory = args.pin_memory
num_workers = args.num_workers
num_channels = args.num_channels
num_classes = args.num_classes
num_levels = args.num_levels
learning_rate = args.learning_rate
num_epochs = args.num_epochs
batch_size = args.batch_size
input_image_width = args.input_image_width
input_image_height = args.input_image_height
input_shape = args.input_shape
logging = args.logging
architecture_name = args.architecture_name
cryosegnet_checkpoint = args.cryosegnet_checkpoint
sam_checkpoint = args.sam_checkpoint
model_type = args.model_type
empiar_id = args.empiar_id
file_name = args.file_name
