import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class FrameDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

        labels = sorted(self.data['label'].unique())
        self.label_map = {lbl: idx for idx, lbl in enumerate(labels)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
    
        # Make path robust: handle Windows absolute paths and backslashes
        rel_path = str(row['filepath'])
        rel_path = rel_path.replace('\\', '/')              # win → posix
        low = rel_path.lower()
        anchor = '/frames_global/'
        i = low.find(anchor)
        if i != -1:
            rel_path = rel_path[i + len(anchor):]           # strip everything up to frames_global/
        # Else: assume it's already relative
    
        img_path = os.path.join(self.root_dir, rel_path)
    
        image = Image.open(img_path).convert("RGB")
        label = self.label_map[row['label']]
    
        if self.transform:
            image = self.transform(image)
        return image, label
