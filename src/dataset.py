import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

class SegmentationDataset(Dataset):
    """
    从 train.txt / test.txt 读取原图路径，自动推导 mask 路径。
    规则：mask 在同一父目录下的 "<类别>Mask/" 里，文件名与原图相同。

    例: ../Datasets/USDatasetsClean/benign/254.png
        -> ../Datasets/USDatasetsClean/benignMask/254.png

        ../Datasets/USDatasetsClean/malignant/118.png
        -> ../Datasets/USDatasetsClean/malignantMask/118.png
    """

    def __init__(self, list_file, img_size=256, augment=False):
        with open(list_file, encoding="utf-8") as f:
            self.img_paths = [line.strip().replace("\\", "/")
                              for line in f if line.strip()]
        self.img_size = img_size
        self.augment = augment
        self.mask_paths = [self._img_to_mask(p) for p in self.img_paths]

    @staticmethod
    def _img_to_mask(img_path):
        """
        ../Datasets/USDatasetsClean/benign/254.png
        -> ../Datasets/USDatasetsClean/benignMask/254.png
        """
        d, fname = os.path.split(img_path)          # d=.../benign, fname=254.png
        parent, cls = os.path.split(d)              # parent=.../USDatasetsClean, cls=benign
        mask_dir = os.path.join(parent, cls + "Mask")
        return os.path.join(mask_dir, fname).replace("\\", "/")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]

        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"读不到图像: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size))

        cls = os.path.basename(os.path.dirname(img_path))
        if cls == "normal":
            # normal 无肿瘤，全黑 mask
            mask = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
        else:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"读不到 mask: {mask_path}")
        mask = cv2.resize(mask, (self.img_size, self.img_size),
                          interpolation=cv2.INTER_NEAREST)

        # 同步增强：随机翻转
        if self.augment:
            if np.random.rand() < 0.5:
                img = np.fliplr(img).copy()
                mask = np.fliplr(mask).copy()
            if np.random.rand() < 0.5:
                img = np.flipud(img).copy()
                mask = np.flipud(mask).copy()

        # 图像归一化 HWC -> CHW
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img)

        # mask 二值化：>127 视为前景
        mask = (mask > 127).astype(np.float32)
        mask = torch.from_numpy(mask).unsqueeze(0)   # (1, H, W)

        return img, mask