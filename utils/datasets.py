from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
import glob
import random
import os
import warnings
import numpy as np
from PIL import Image
from PIL import ImageFile
from utils.augmentations import AUGMENTATION_TRANSFORMS

ImageFile.LOAD_TRUNCATED_IMAGES = True

def pad_to_square(img, pad_value):
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)

    return img, pad


def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image

def parse_label_data(img_list):
    label = []
    for file_path in img_list:
        b = file_path.split('/')
        file_name = b[-1].replace('.jpg', '.txt').replace(".png", ".txt")
        new = b[:-1]
        new.append('Label')
        new.append(file_name)
        label_path = '/'.join(new)
        label.append(label_path)
    return label

class ImageFolder(Dataset):
    def __init__(self, folder_path, transform=None):
        self.files = sorted(glob.glob("%s/*.*" % folder_path))
        self.transform = transform

    def __getitem__(self, index):

        img_path = self.files[index % len(self.files)]
        img = np.array(
            Image.open(img_path).convert('RGB'),
            dtype=np.uint8)

        # Label Placeholder
        boxes = np.zeros((1, 5))

        # Apply transforms
        if self.transform:
            img, _ = self.transform((img, boxes))

        return img_path, img

    def __len__(self):
        return len(self.files)


# ListDataset 이라는 클래스
# train.py에서 ListDataset 클래스를 호출할 떄 input 값으로 train.txt (입력이미지 파일 경로들이 적힌 txt)
class ListDataset(Dataset):
    def __init__(self, custom, file_path, img_size=416, multiscale=True, transform=None):
        if custom:
            with open(file_path, "r") as file:
                self.img_files = file.readlines()  # 이미지 파일의 path들을 읽어들임
            self.label_files = parse_label_data(self.img_files)
        else:
            with open(file_path, "r") as file:
                self.img_files = file.readlines()  # 이미지 파일의 path들을 읽어드림
            self.label_files = [
                path.replace("images", "Labels").replace(".png", ".txt").replace(".jpg", ".txt")
                for path in self.img_files
            ]

        self.img_size = img_size
        self.max_objects = 100
        self.multiscale = multiscale
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0
        self.transform = transform

    def __getitem__(self, index):

        # ---------
        #  Image
        # ---------
        try:
            img_path = self.img_files[index % len(self.img_files)].rstrip()
            img = np.array(Image.open(img_path).convert('RGB'), dtype=np.uint8)
        except Exception as e:
            pass
            # print(f"Could not read image '{img_path}'.")
            return

        # ---------
        #  Label
        # ---------
        try:
            label_path = self.label_files[index % len(self.img_files)].rstrip()
            # Ignore warning if file is empty
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                boxes = np.loadtxt(label_path).reshape(-1, 5)
        except Exception as e:
            print(f"Could not read label '{label_path}'.")
            return

        # -----------
        #  Transform
        # -----------

        if self.transform:
            try:
                img, bb_targets = self.transform((img, boxes))
            except:
                print(f"Could not apply transform.")
                return

        return img_path, img, bb_targets

    def collate_fn(self, batch):
        self.batch_count += 1

        # Drop invalid images
        batch = [data for data in batch if data is not None]
        paths, imgs, bb_targets = list(zip(*batch))

        # Selects new image size every tenth batch
        if self.multiscale and self.batch_count % 10 == 0:
            self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))

        # Resize images to input shape
        imgs = torch.stack([resize(img, self.img_size) for img in imgs])

        # Add sample index to targets
        for i, boxes in enumerate(bb_targets):
            boxes[:, 0] = i
        bb_targets = torch.cat(bb_targets, 0)

        return paths, imgs, bb_targets

    def __len__(self):
        return len(self.img_files)
