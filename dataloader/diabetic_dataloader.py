import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torchvision import transforms
import random
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import AutoImageProcessor
from collections import Counter

DEFAULT_DIABETIC_IMAGE_FOLDER = os.getenv("DIABETIC_IMAGE_FOLDER", "data/drd")

def _resolve_path(value, default):
    return default if value in (None, "", "None") else value

class CustomImageDataset(Dataset):
    def __init__(self, csv_file, image_folder, image_processor, transform=None, num_classes=5):

        self.data = pd.read_csv(csv_file)
        self.image_folder = image_folder
        self.image_processor = image_processor
        self.transform = transform
        self.num_classes = num_classes

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        img_path = self.data.loc[idx, 'image']
        label = self.data.loc[idx, 'level']


        full_path = os.path.join(self.image_folder, img_path + ".jpeg")
        image = Image.open(full_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        processed = self.image_processor(images=image, return_tensors="pt")
        image_tensor = processed["pixel_values"].squeeze(0)  

        if label == 0:
            label = 0
        elif label in [1,2,3]:
            label = 1
        elif label in [4]:
            label = 2

        label_tensor = torch.tensor(label, dtype=torch.long)

        return image_tensor, label_tensor


def diabetic_create_dataloader(args, is_train="train"):


    image_processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
    diabetic_image_folder = _resolve_path(getattr(args, "diabetic_image_folder", None), DEFAULT_DIABETIC_IMAGE_FOLDER)

    if is_train == "train":
        select_num = random.randint(0, 3)
        augmentations = []

        if select_num == 1:
            augmentations.append(transforms.RandomHorizontalFlip(p=1.0))
        elif select_num == 2:
            augmentations.append(transforms.RandomRotation(degrees=(-8, 8)))
        elif select_num == 3:
            augmentations.append(transforms.RandomVerticalFlip(p=1.0))
        transform = transforms.Compose(augmentations) if augmentations else None
    else:
        transform = None

    if is_train == "train":
        dataset = CustomImageDataset(
            csv_file=args.diabetic_train_csv_file,
            image_folder=diabetic_image_folder,
            image_processor=image_processor,
            transform=transform,
            num_classes=args.num_classes
        )
        
    elif is_train == "val":
        dataset = CustomImageDataset(
            csv_file=args.diabetic_val_csv_file,
            image_folder=diabetic_image_folder,
            image_processor=image_processor,
            transform=transform,
            num_classes=args.diabetic_classes
        )
    elif is_train == "test":
        dataset = CustomImageDataset(
            csv_file=args.diabetic_test_csv_file,
            image_folder=diabetic_image_folder,
            image_processor=image_processor,
            transform=transform,
            num_classes=args.diabetic_classes
        )
        return DataLoader(
            dataset,
            batch_size=1,
            shuffle=True,
            num_workers=24,
            pin_memory=True,
            drop_last=True
        )
    elif is_train == "total":
        dataset = CustomImageDataset(
            csv_file=args.diabetic_test_csv_file,
            image_folder=diabetic_image_folder,
            image_processor=image_processor,
            transform=transform,
            num_classes=args.diabetic_classes
        )
        return dataset

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=24,
        pin_memory=True,
        drop_last=True
    )

    return dataloader