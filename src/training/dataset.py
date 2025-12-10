import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class FrameDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir              # required for portable paths
        self.transform = transform

        labels = sorted(self.data['label'].unique())
        self.label_map = {lbl: idx for idx, lbl in enumerate(labels)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # CSV stores relative path: <label>/<filename>.jpg
        rel = row['rel_path'].replace("\\", "/")  # windows-proof
        img_path = os.path.join(self.root_dir, rel)

        image = Image.open(img_path).convert("RGB")
        label = self.label_map[row['label']]

        if self.transform:
            image = self.transform(image)

        return image, label
