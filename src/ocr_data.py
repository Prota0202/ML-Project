import struct
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from . import config


class CharIDXDataset(Dataset):
    def __init__(self, images_path, labels_path, mapping_path=None):
        images_path = Path(images_path)
        labels_path = Path(labels_path)
        mapping_path = Path(mapping_path or (config.IMAGE_DIR / "mapping.txt"))

        with open(images_path, "rb") as f:
            magic, n, h, w = struct.unpack(">IIII", f.read(16))
            buf = np.frombuffer(f.read(), dtype=np.uint8).reshape(n, h, w)
        with open(labels_path, "rb") as f:
            magic, n2 = struct.unpack(">II", f.read(8))
            labels = np.frombuffer(f.read(), dtype=np.uint8)

        self.label_to_char = {}
        with open(mapping_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                idx_s, asc_s = line.split()
                self.label_to_char[int(idx_s)] = chr(int(asc_s))

        # Sans cette transposition, les caractères ressortent retournés.
        # Je l'ai vu au premier plot, c'est dans le script fourni.
        buf = np.transpose(buf, (0, 2, 1)).copy()
        self.images = torch.tensor(buf, dtype=torch.float32) / 255.0
        self.labels = torch.tensor(labels.astype(np.int64))

    def __len__(self):
        return int(self.labels.shape[0])

    def __getitem__(self, idx):
        return self.images[idx].unsqueeze(0), int(self.labels[idx].item())

    @property
    def num_classes(self):
        return int(self.labels.max().item()) + 1

    def class_labels(self):
        n = self.num_classes
        return [self.label_to_char.get(i, "?") for i in range(n)]
