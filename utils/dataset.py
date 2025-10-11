import os

from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

class ColorizationDataset(Dataset):
    def __init__(self, root="./data", split="train", img_size=256):
        self.root = root
        self.split = split
        self.filenames = os.listdir(os.path.join(root, f"{split}_black"))

        self.transform_black = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, ), std=(0.5, ))
        ])

        self.transform_color = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]

        file_black = os.path.join(self.root, f"{self.split}_black", filename)
        file_color = os.path.join(self.root, f"{self.split}_color", filename)

        img_black = Image.open(file_black)
        img_color = Image.open(file_color)

        img_black = self.transform_black(img_black)
        img_color = self.transform_color(img_color)
        return img_black, img_color

def prepare_data(root="./data", split="train", batch_size=64, img_size=256):
    dataset = ColorizationDataset(root, split, img_size)
    shuffle = True if split == "train" else False
    dataloader = DataLoader(dataset, batch_size, shuffle)
    return dataloader