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

DEFAULT_APTOS_TRAIN_IMAGE_FOLDER = os.getenv("APTOS_TRAIN_IMAGE_FOLDER", "data/aptos_train")
DEFAULT_APTOS_VAL_IMAGE_FOLDER = os.getenv("APTOS_VAL_IMAGE_FOLDER", "data/aptos_val")
DEFAULT_APTOS_TEST_IMAGE_FOLDER = os.getenv("APTOS_TEST_IMAGE_FOLDER", "data/aptos_test")

def _resolve_path(value, default):
    return default if value in (None, "", "None") else value
class CustomImageDataset(Dataset):
    def __init__(self, csv_file, image_folder, image_processor, transform=None, num_classes=5):

        self.data = pd.read_csv(csv_file)
        self.image_processor = image_processor
        self.num_classes = num_classes
        self.transform = transform
        self.image_folder = image_folder

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        img_path = self.data.loc[idx, 'id_code']
        label = self.data.loc[idx, 'diagnosis']

        image = Image.open(os.path.join(self.image_folder, img_path + ".png")).convert('RGB')

        if self.transform:
            image = self.transform(image)

        processed = self.image_processor(image, return_tensors="pt")
        image = processed["pixel_values"].squeeze(0)

        if label == 0:
            label = 0
        elif label in [1,2,3]:
            label = 1
        elif label in [4]:
            label = 2
        
        label = torch.tensor(label, dtype=torch.long)

        return image, label

def atop_create_dataloader(args, is_train="train"):

    image_processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
    aptos_train_image_folder = _resolve_path(getattr(args, "aptos_train_image_folder", None), DEFAULT_APTOS_TRAIN_IMAGE_FOLDER)
    aptos_val_image_folder = _resolve_path(getattr(args, "aptos_val_image_folder", None), DEFAULT_APTOS_VAL_IMAGE_FOLDER)
    aptos_test_image_folder = _resolve_path(getattr(args, "aptos_test_image_folder", None), DEFAULT_APTOS_TEST_IMAGE_FOLDER)
    if is_train == "train":
        select_num = random.randint(0, 3)
        aug_list = []
        if select_num == 1:
            aug_list.append(transforms.RandomHorizontalFlip(p=1.0))
        elif select_num == 2:
            aug_list.append(transforms.RandomRotation(degrees=(-8, 8)))
        elif select_num == 3:
            aug_list.append(transforms.RandomVerticalFlip(p=1.0))
        transform = transforms.Compose(aug_list) if aug_list else None
    else:
        transform = None
    
    if is_train == "train":
        dataset = CustomImageDataset(
            csv_file=r"./dataloader/atop/train_1.csv",
            image_folder=aptos_train_image_folder,
            image_processor=image_processor,
            transform = transform,
            num_classes=args.num_classes
        )

    elif is_train == "val":
        dataset = CustomImageDataset(
            csv_file=r"./dataloader/atop/valid.csv",
            image_folder=aptos_val_image_folder,
            image_processor=image_processor,
            transform = transform,
            num_classes=args.num_classes
        )
    elif is_train == "test":
        dataset = CustomImageDataset(
            csv_file=r"./dataloader/atop/test.csv",
            image_folder=aptos_test_image_folder,
            image_processor=image_processor,
            transform = transform,
            num_classes=args.num_classes
        )
        return DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    elif is_train == "total":
        dataset = CustomImageDataset(
            csv_file=r"./dataloader/atop/test.csv",
            image_folder=aptos_test_image_folder,
            image_processor=image_processor,
            transform = transform,
            num_classes=args.num_classes
        )
        return dataset

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4 * 10,  # 필요에 따라 조정하세요.
        pin_memory=True,
        drop_last=True
    )

    return dataloader
