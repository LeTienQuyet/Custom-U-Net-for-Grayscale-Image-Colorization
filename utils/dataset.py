import os

import kornia.color as K

from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

class ColorizationDataset(Dataset):
    def __init__(self, root="./data", split="train", img_size=128, color_mode="lab"):
        self.root = root
        self.split = split
        self.color_mode = color_mode
        self.filenames = os.listdir(os.path.join(root, split))

        if split == "train":
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor()
            ])


    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        file_path = os.path.join(self.root, self.split, filename)

        img = Image.open(file_path).convert("RGB")
        rgb_tensor = self.transform(img) # [3, H, W] in range [0, 1]

        lab_tensor = K.rgb_to_lab(rgb_tensor.unsqueeze(0)).squeeze(0) # [1, 3, H, W] for kornia -> [3, H, W]
        L_tensor = lab_tensor[0:1, :, :] / 100

        if self.color_mode == "lab":
            # ab for target
            ab_tensor = (lab_tensor[1:3, :, :] + 128.0) / 255.0
            return L_tensor, ab_tensor
        else:
            # RGB for target
            return L_tensor, rgb_tensor

def prepare_data(root="./data", split="train", batch_size=64, img_size=128, color_mode="lab"):
    dataset = ColorizationDataset(root, split, img_size, color_mode)
    shuffle = True if split == "train" else False
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2)
    return dataloader