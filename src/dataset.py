import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class FrameDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, label_map=None):
        """
        Args:
            csv_file (str): Path to metadata.csv
            root_dir (str): Root directory with all extracted frames
            transform (callable, optional): Torchvision transforms
            label_map (dict, optional): Maps string labels -> numeric class IDs
        """
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

        # If not provided, build mapping from dataset
        if label_map is None:
            labels = sorted(self.data['label'].unique())
            self.label_map = {lbl: idx for idx, lbl in enumerate(labels)}
        else:
            self.label_map = label_map

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.root_dir, row['filepath'])
        image = Image.open(img_path).convert("RGB")
        label = self.label_map[row['label']]

        if self.transform:
            image = self.transform(image)

        return image, label
