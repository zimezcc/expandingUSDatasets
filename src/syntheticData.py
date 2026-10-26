# 提取肿瘤区域，随机选择normal图像位置并柔和融合 (最终完美版)
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

def refine_edges(node_weight, BW):
    """
    按照你的逻辑：检查每行/每列的非0边界，若不是 unit，则向外补齐递减。
    """
    m, n, c = node_weight.shape
    unit = 1.0 / BW
    epsilon = 1e-5  # 用于浮点数比较

    # 1. 按行检查
    for row in range(m):
        # 提取当前行非零的索引
        nonzero_cols = np.where(node_weight[row, :, 0] > 0)[0]
        if len(nonzero_cols) > 0:
            min_idx = nonzero_cols[0]
            max_idx = nonzero_cols[-1]

            # 检查左边缘
            val_left = node_weight[row, min_idx, 0]
            if abs(val_left - unit) > epsilon:
                # 如果不是 unit，且没有贴着图像边缘，则往外依次更新
                curr_val = val_left - unit
                idx = min_idx - 1
                while idx >= 0 and curr_val > 0:
                    # 如果碰到的原有值比要更新的值大，则保留原有值（避免覆盖别的行扩展过来的）
                    node_weight[row, idx, :] = max(node_weight[row, idx, 0], curr_val)
                    curr_val -= unit
                    idx -= 1

            # 检查右边缘
            val_right = node_weight[row, max_idx, 0]
            if abs(val_right - unit) > epsilon:
                curr_val = val_right - unit
                idx = max_idx + 1
                while idx < n and curr_val > 0:
                    node_weight[row, idx, :] = max(node_weight[row, idx, 0], curr_val)
                    curr_val -= unit
                    idx += 1

    # 2. 按列检查
    for col in range(n):
        # 提取当前列非零的索引
        nonzero_rows = np.where(node_weight[:, col, 0] > 0)[0]
        if len(nonzero_rows) > 0:
            min_idx = nonzero_rows[0]
            max_idx = nonzero_rows[-1]

            # 检查上边缘
            val_top = node_weight[min_idx, col, 0]
            if abs(val_top - unit) > epsilon:
                curr_val = val_top - unit
                idx = min_idx - 1
                while idx >= 0 and curr_val > 0:
                    node_weight[idx, col, :] = max(node_weight[idx, col, 0], curr_val)
                    curr_val -= unit
                    idx -= 1

            # 检查下边缘
            val_bottom = node_weight[max_idx, col, 0]
            if abs(val_bottom - unit) > epsilon:
                curr_val = val_bottom - unit
                idx = max_idx + 1
                while idx < m and curr_val > 0:
                    node_weight[idx, col, :] = max(node_weight[idx, col, 0], curr_val)
                    curr_val -= unit
                    idx += 1

    return node_weight

def improve_mix(image, normal, mask, BW):
    # 注意：这里改为获取 normal 图片的真实尺寸作为最终输出画布尺寸
    m, n, c = normal.shape

    mask_index = np.where(mask == 255)

    # 1. 求取肿瘤本身的实际高和宽
    min_mask_row = np.min(mask_index[0])
    max_mask_row = np.max(mask_index[0])
    min_mask_col = np.min(mask_index[1])
    max_mask_col = np.max(mask_index[1])

    tumor_h = max_mask_row - min_mask_row + 1
    tumor_w = max_mask_col - min_mask_col + 1
    min_tumor_dim = min(tumor_h, tumor_w)

    # ================= 动态自适应 BW =================
    current_BW = BW  # 默认就是传入的 20

    # 兜底保护：防止原图太小导致越界
    current_BW = min(current_BW, image.shape[0] // 2, image.shape[1] // 2)
    # ===============================================

    # 2. 使用 current_BW 裁剪肿瘤块 (从 image 裁剪)
    min_row = max(0, min_mask_row - current_BW)
    max_row = min(image.shape[0] - 1, max_mask_row + current_BW)
    min_col = max(0, min_mask_col - current_BW)
    max_col = min(image.shape[1] - 1, max_mask_col + current_BW)

    node = image[min_row:max_row + 1, min_col:max_col + 1, :]
    node_mask = mask[min_row:max_row + 1, min_col:max_col + 1]

    # 3. 数据增强
    transform = A.Compose([A.RandomRotate90(p=0.5), A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5)])
    augmented = transform(image=node, mask=node_mask)
    augmented_image = augmented['image']
    augmented_mask = augmented['mask']

    node_weight = augmented_mask / 255
    node_weight = bianyuan_ronghe(node_weight, BW, BW)
    # 2. 执行你提出的查漏补缺逻辑
    node_weight = refine_edges(node_weight, BW)
    # 4. 计算宽高，寻找背景位置
    height = augmented_image.shape[0]
    width = augmented_image.shape[1]

    if width >= n - 1:
        width = (int)((n - 1) / 2)
    if height >= m - 1:
        height = (int)((m - 1) / 2)

    # random_x = random.randrange((int)((m-height-1)/2))
    random_x = random.randrange(m - height - 1)
    random_y = random.randrange(n - width - 1)

    # 缩放图像 (Mask 必须用最近邻插值)
    resized_node = cv2.resize(augmented_image, (width, height), interpolation=cv2.INTER_LINEAR)
    resized_node_mask = cv2.resize(augmented_mask, (width, height), interpolation=cv2.INTER_NEAREST)
    resized_node_weight = cv2.resize(node_weight, (width, height), interpolation=cv2.INTER_NEAREST)
    # 注意：这里使用 normal 的真实尺寸 m, n, c 来创建画布！
    weight = np.zeros((m, n, c))
    generate_mask = np.zeros((m, n, c))

    generate_mask[random_x: random_x + height, random_y: random_y + width, 0] = resized_node_mask
    generate_mask[random_x: random_x + height, random_y: random_y + width, 1] = resized_node_mask
    generate_mask[random_x: random_x + height, random_y: random_y + width, 2] = resized_node_mask

    weight[random_x: random_x + height, random_y: random_y + width, 0] = resized_node_weight
    weight[random_x: random_x + height, random_y: random_y + width, 1] = resized_node_weight
    weight[random_x: random_x + height, random_y: random_y + width, 2] = resized_node_weight

    # 7. 最终融合
    generate_image = np.zeros((m, n, c))
    generate_image[random_x: random_x + height, random_y: random_y + width, :] = resized_node

    # 此时 generate_image 和 normal 的 shape 都是 (m, n, c)，完美 broadcast
    generate_image = generate_image * weight + normal * (1.0 - weight)

    return generate_image, generate_mask