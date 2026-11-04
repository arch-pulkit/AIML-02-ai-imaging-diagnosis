from pathlib import Path

import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset

from src.data.make_splits import OUT_PATH as SPLITS_CSV
from src.data.nih import LABELS
from src.data.resize_images import OUT_DIR as IMAGE_DIR

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


"""
returns a torchvision compose
pil 'L image to float tensor (3, size, size), normalized
Transforms are sequence of steps applied to transform raw image into the exact format a PyTorch model expects.
"""
def build_transforms(train: bool, size: int = 224):
    if train:
        return T.Compose([
            T.Resize((size, size)),
            T.RandomAffine(degrees= 7, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            T.ColorJitter(brightness=0.1, contrast=0.1),
            T.Grayscale(num_output_channels=3),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    else:
        return T.Compose([
            T.Resize((size, size)),
            T.Grayscale(num_output_channels=3),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    
#In PyTorch, any custom dataset needs to inherit Dataset and must follow 3 core parts: __init__, __len__, __getitem__
class NIHDataset(Dataset):
    def __init__(self, splits_csv=SPLITS_CSV, image_dir=IMAGE_DIR, split: str = "train", transform=None):
        #Read and filter splits SCV by splits
        df = pd.read_csv(splits_csv)
        df = df[df["split"] == split].reset_index(drop=True)

        if len(df) == 0:
            raise ValueError(f"No rows for split={split!r}")

        #storing only file path and labels in memory, not load all images on ram at once
        #in float as PyTorch's loss function rejects integer labels
        image_dir_path = Path(image_dir)
        self.image_paths = [image_dir_path / name for name in df["image_index"]]

        #Store labels as float32 Numpy arrays
        self.labels = df[LABELS].to_numpy(dtype="float32")

        #setting up transforms, defaults to train/eval transforms based on split in None provided
        self.transform = transform if transform is not None else build_transforms(train=(split == "train"))

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:

        img_path = self.image_paths[idx]

        with Image.open(img_path) as img:
            img = img.convert("L")
            img_tensor = self.transform(img)

        label_tensor = torch.tensor(self.labels[idx], dtype=torch.float32)

        return img_tensor, label_tensor




 


