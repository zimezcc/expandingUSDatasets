# 提取肿瘤区域，随机选择normal图像位置
import os
import torch
import numpy as np
import cv2
import random
import albumentations as A

def seed_torch(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


seed_torch(41)

def bianyuan_ronghe(node_weight, row_ronghe_pixels, col_ronghe_pixels):
    row_unit = 1 / row_ronghe_pixels
    col_unit = 1 / col_ronghe_pixels
    m, n, c = node_weight.shape
    for row in range(m):
        cols = np.where(node_weight[row, :, 0] == 1)[0]
        if len(cols) > 0:
            min_col_row = np.min(cols)
            max_col_row = np.max(cols)
            for i in range(0, col_ronghe_pixels):
                node_weight[row, max(min_col_row - i, 0), :] = 1 - col_unit * (i + 1)
                node_weight[row, min(max_col_row + i, n - 1), :] = 1 - col_unit * (i + 1)
    for col in range(n):
        rows = np.where(node_weight[:, col, 0] == 1)[0]
        if len(rows) > 0:
            min_col_row = np.min(rows)
            max_col_row = np.max(rows)
            for i in range(0, row_ronghe_pixels):
                node_weight[max(min_col_row - i, 0), col, :] = max(1 - row_unit * (i + 1),
                                                                   node_weight[max(min_col_row - i, 0), col, 0])
                node_weight[min(max_col_row + i, m - 1), col, :] = max(1 - row_unit * (i + 1),
                                                                       node_weight[min(max_col_row + i, m - 1), col, 0])
    return node_weight

def improve_mix(image, normal, mask, ronghe_ration=0.1):
    m, n, c = image.shape
    mask_index = np.where(mask == 255)
    #求取肿瘤大小
    min_mask_row = np.min(mask_index[0])
    max_mask_row = np.max(mask_index[0])
    row_ronghe_pixels = int((max_mask_row - min_mask_row)*ronghe_ration)
    min_mask_col = np.min(mask_index[1])
    max_mask_col = np.max(mask_index[1])
    col_ronghe_pixels = int((max_mask_col - min_mask_col)*ronghe_ration)
    min_row = max(0, min_mask_row - row_ronghe_pixels)
    max_row = min(m - 1, max_mask_row + row_ronghe_pixels)
    min_col = max(0, min_mask_col - col_ronghe_pixels)
    max_col = min(n - 1, max_mask_col + col_ronghe_pixels)
    node = image[min_row:max_row + 1, min_col:max_col + 1, :]
    node_mask = mask[min_row:max_row + 1, min_col:max_col + 1]
    transform = A.Compose([A.RandomRotate90(p=0.5), A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5)])
    augmented = transform(image=node, mask=node_mask)
    augmented_iamge = augmented['image']
    augmented_mask = augmented['mask']
    node_weight = augmented_mask / 255
    node_weight = bianyuan_ronghe(node_weight, row_ronghe_pixels, col_ronghe_pixels)
    # 计算新的宽度和高度
    width = augmented_iamge.shape[1]
    height = augmented_iamge.shape[0]
    m, n, c = normal.shape
    if width >= n - 1:
        width = (int)((n - 1) / 2)
    if height >= m - 1:
        height = (int)((m - 1) / 2)
    # 缩放图像
    resized_node = cv2.resize(augmented_iamge, (width, height), interpolation=cv2.INTER_LINEAR)
    resized_node_mask = cv2.resize(augmented_mask, (width, height), interpolation=cv2.INTER_LINEAR)
    resized_node_weight = cv2.resize(node_weight, (width, height), interpolation=cv2.INTER_LINEAR)

    # random_x = random.randrange((int)((m-height-1)/2))
    random_x = random.randrange(m - height - 1)
    random_y = random.randrange(n - width - 1)

    weight = np.zeros((m, n, c))
    generate_mask = np.zeros((m, n, c))
    generate_mask[random_x: random_x + height, random_y:random_y + width, 0] = resized_node_mask
    generate_mask[random_x: random_x + height, random_y:random_y + width, 1] = resized_node_mask
    generate_mask[random_x: random_x + height, random_y:random_y + width, 2] = resized_node_mask
    weight[random_x: random_x + height, random_y:random_y + width, 0] = resized_node_weight
    weight[random_x: random_x + height, random_y:random_y + width, 1] = resized_node_weight
    weight[random_x: random_x + height, random_y:random_y + width, 2] = resized_node_weight
    generate_image = np.zeros((m, n, c))
    generate_image[random_x: random_x + height, random_y:random_y + width, :] = resized_node
    generate_image = generate_image * weight + normal * (1 - weight)
    return generate_image, generate_mask
