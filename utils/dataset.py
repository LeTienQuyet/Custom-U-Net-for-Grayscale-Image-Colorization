import os

from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

class ColorizationDataset(Dataset):
    def __init__(self, root="./data", split="train", img_size=256):
        self.root = root
        self.split = split
        self.filenames = os.listdir(os.path.join(root, split))

        if split == "train":
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip()
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
            ])

        self.gray_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        self.color_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        file_path = os.path.join(self.root, self.split, filename)

        img = Image.open(file_path).convert("RGB")
        img = self.transform(img)

        gray_img = transforms.Grayscale(num_output_channels=1)(img)

        gray_tensor = self.gray_transform(gray_img)
        color_tensor = self.color_transform(img)

        return gray_tensor, color_tensor

def prepare_data(root="./data", split="train", batch_size=64, img_size=256):
    dataset = ColorizationDataset(root, split, img_size)
    shuffle = True if split == "train" else False
    dataloader = DataLoader(dataset, batch_size, shuffle)
    return dataloader