import torch
import os
import cv2
import pandas as pd
from torch.utils.data import Dataset
import numpy as np
from enum import Enum
from PIL import Image


class SampleType(Enum):
    DICT = "dict"
    TUPLE = "tuple"
    LIST = "list"


def _get_typed_sample(input, target, sample_type):
    if sample_type == SampleType.DICT:
        return {"input": input, "target": target}
    elif sample_type == SampleType.TUPLE:
        return (input, target)
    elif sample_type == SampleType.LIST:
        return [input, target]
    else:
        raise TypeError("Provided sample_type is not dict, list, tuple")


class BeautyDataset(Dataset):
    def __init__(self, root, crop_size, sample_type=SampleType.DICT):
        self.root = root
        data = pd.read_csv("/opt/userhome/yangqingpu/workspace/my-project/data.csv")
        self.images = data["name"].tolist()
        self.labels = data["label"].tolist()
        self.sample_type = sample_type
        self.crop_size = crop_size

    def __getitem__(self, index):
        img_path = os.path.join(self.root, "Images", self.images[index])
        target = self.labels[index]
        img = cv2.imread(img_path).astype(np.uint8)
        img = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
        # input_ = torch.from_numpy(img)
        return _get_typed_sample(img, target, self.sample_type)

    def __len__(self):
        return len(self.images)

