import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np



class CarvanaDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index].replace(".jpg", "_mask.gif"))
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask[mask == 255.0] = 1.0

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask


img_dir_train = "data/train_images"
img_dir_train_mask = "data/train_masks"
img_dir_val = "data/val_images"
img_dir_val_mask = "data/val_masks"

def CarvanaDataset_practice():
    CarvanaDataset(img_dir_train, img_dir_train_mask, transform=None)
    print(len(img_dir_train))

if __name__ == "main":
    CarvanaDataset_practice()