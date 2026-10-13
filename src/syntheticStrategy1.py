import os
import random
from glob import glob
import cv2
import numpy as np
import torch
from syntheticImage import improve_mix

# ---------------- 配置 ----------------
BW = 20          # 融合边缘宽度控制
DATASET_NAME = f"BW{BW}"
DATASET_ROOT = "../Datasets/SyntheticDataset1"
NORMAL_DIR = "../Datasets/USDatasetsClean/normal"
TRAIN_FILE = "train_data_list/train.txt"
SEED = 41

def seed_torch(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ["PYTHONHASHSEED"] = str(seed)


def make_dirs():
    for sub in ["benign", "benignMask", "malignant", "malignantMask"]:
        os.makedirs(os.path.join(DATASET_ROOT, DATASET_NAME, sub), exist_ok=True)


def create_extend_datasets():
    with open(TRAIN_FILE, encoding="utf-8") as f:
        sample_list = [line.strip().replace("\\", "/") for line in f if line.strip()]

    extends = []

    for sample in sample_list:
        sample_dir, sample_name = os.path.split(sample)
        sample_id = os.path.splitext(sample_name)[0]
        sample_root, sample_type = os.path.split(sample_dir)
        normal_id = 116
        normal = os.path.join(NORMAL_DIR, str(normal_id)+".png")

        gen_img = os.path.join(DATASET_ROOT, DATASET_NAME, sample_type,
                               f"{sample_id}_{normal_id}.png")
        gen_mask = os.path.join(DATASET_ROOT, DATASET_NAME, sample_type + "Mask",
                                f"{sample_id}_{normal_id}.png")

        if os.path.exists(gen_img):
            extends.append(gen_img)
            continue

        img = cv2.imread(sample)
        mask_path = os.path.join(sample_root, sample_type + "Mask", sample_id + ".png")
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)[..., None]
        normal_img = cv2.imread(normal)

        if img is None or mask is None or normal_img is None:
            print(f"[skip] 读图失败: {sample}")
            continue

        gen_image, gen_label = improve_mix(img, normal_img, mask, BW)
        # 当BW=0时，使用下面的语句
        # gen_image, gen_label = improve_mix(img, normal_img, mask, BW + 1)
        cv2.imwrite(gen_img, gen_image)
        cv2.imwrite(gen_mask, gen_label)
        extends.append(gen_img)
        print(f"生成: {gen_img}")

    # ---------------- 写入记录 txt ----------------
    out_txt = f"synthetic1_{DATASET_NAME}_list.txt"
    with open(out_txt, "w", encoding="utf-8") as f:
        for p in extends:
            f.write(p.replace("\\", "/") + "\n")
    print("-" * 40)
    print(f"共生成 {len(extends)} 张合成图")

if __name__ == "__main__":
    seed_torch(SEED)
    make_dirs()
    create_extend_datasets()