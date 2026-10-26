import os
import sys
import glob
import csv
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# 让 Python 能找到 networks 和 dataset
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dataset import SegmentationDataset
from networks.CMUNeXt import cmunext
from networks.UNeXt import UNext
from networks.CMUNet import CMUNet
from networks.MedNeXt import MedNeXt2D
from networks.ConvUNeXt import ConvUNeXt

# ---------------- 配置 ----------------
IMG_SIZE = 256
TEST_BATCH_SIZE = 64
NUM_CLASSES = 1
IN_CHANNELS = 3
NUM_WORKERS = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_LIST = os.path.join(BASE_DIR, "test.txt")

# 模型字典
MODEL_ZOO = {
    "CMUNeXt": lambda: cmunext(),
    "UNext": lambda: UNext(),
    "CMUNet": lambda: CMUNet(),
    "MedNeXt": lambda: MedNeXt2D(in_channels=IN_CHANNELS, n_classes=NUM_CLASSES, preset='B', deep_supervision=False),
    "ConvUNeXt": lambda: ConvUNeXt(in_channels=IN_CHANNELS, n_classes=NUM_CLASSES, base_channels=32),
}


# 指标计算 (与训练代码完全一致)
def batch_metrics(pred_logits, targets, threshold=0.5, eps=1e-6):
    preds = (torch.sigmoid(pred_logits) > threshold).float()
    preds_flat = preds.view(preds.size(0), -1)
    targets_flat = targets.view(targets.size(0), -1)

    tp = (preds_flat * targets_flat).sum(dim=1)
    fp = (preds_flat * (1 - targets_flat)).sum(dim=1)
    fn = ((1 - preds_flat) * targets_flat).sum(dim=1)

    iou_per = (tp + eps) / (tp + fp + fn + eps)
    f1_per = (2 * tp + eps) / (2 * tp + fp + fn + eps)

    return iou_per.mean().item(), f1_per.mean().item()


@torch.no_grad()
def evaluate(model, test_loader):
    model.eval()
    total_iou = 0.0
    total_f1 = 0.0
    n = 0

    for img, mask in tqdm(test_loader, desc="Testing", leave=False):
        img = img.to(DEVICE, non_blocking=True)
        mask = mask.to(DEVICE, non_blocking=True)

        out = model(img)
        iou_m, f1_m = batch_metrics(out, mask)
        bs = img.size(0)
        total_iou += iou_m * bs
        total_f1 += f1_m * bs
        n += bs

    return total_iou / n, total_f1 / n


def main():
    print(f"使用设备: {DEVICE}")
    test_ds = SegmentationDataset(TEST_LIST, IMG_SIZE, augment=False)
    test_loader = DataLoader(test_ds, batch_size=TEST_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
                             pin_memory=True)
    print(f"测试集大小: {len(test_ds)} 张图片")

    results = []

    # 遍历 checkpoints 下的所有实验
    # 假设你的保存路径是: checkpoints/1026/{model_name}/{train_name}/best.pth
    ckpt_base_dir = os.path.join(BASE_DIR, "checkpoints")

    if not os.path.exists(ckpt_base_dir):
        print(f"错误: 找不到权重目录 {ckpt_base_dir}")
        return

    for model_name in MODEL_ZOO.keys():
        model_dir = os.path.join(ckpt_base_dir, model_name)
        if not os.path.exists(model_dir):
            continue

        for exp_name in os.listdir(model_dir):
            exp_dir = os.path.join(model_dir, exp_name)
            if not os.path.isdir(exp_dir):
                continue

            best_ckpt_path = os.path.join(exp_dir, "best.pth")
            if not os.path.exists(best_ckpt_path):
                continue

            print(f"\n>>> 正在验证: {model_name} | {exp_name}")

            # 1. 加载模型
            try:
                model = MODEL_ZOO[model_name]().to(DEVICE)
                checkpoint = torch.load(best_ckpt_path, map_location=DEVICE)

                # 兼容不同保存格式
                if "model_state" in checkpoint:
                    model.load_state_dict(checkpoint["model_state"])
                else:
                    model.load_state_dict(checkpoint)
            except Exception as e:
                print(f"  [错误] 加载模型失败: {e}")
                continue

            # 2. 验证
            val_iou, val_f1 = evaluate(model, test_loader)

            # 3. 记录原始训练时的指标 (做对比)
            saved_f1 = checkpoint.get("best_f1", -1.0)
            saved_iou = checkpoint.get("iou_at_best_f1", -1.0)
            best_epoch = checkpoint.get("epoch", -1)

            print(f"  -> 重新计算: F1={val_f1:.4f}, IoU={val_iou:.4f}")
            print(f"  -> 训练记录: F1={saved_f1:.4f}, IoU={saved_iou:.4f} (Epoch {best_epoch})")

            # 差异过大说明可能数据加载或评测逻辑有变
            if abs(val_f1 - saved_f1) > 0.01:
                print(f"  [警告] 重新计算与训练记录差异较大！")

            results.append({
                "train_list": exp_name,
                "model": model_name,
                "re_eval_f1": f"{val_f1:.4f}",
                "re_eval_iou": f"{val_iou:.4f}",
                "saved_f1": f"{saved_f1:.4f}",
                "saved_iou": f"{saved_iou:.4f}",
                "best_epoch": best_epoch
            })

            # 清理显存
            del model
            torch.cuda.empty_cache()

    # 保存结果
    if results:
        csv_path = os.path.join(BASE_DIR, "re_evaluate_summary.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\n验证完成！结果已保存至: {csv_path}")
    else:
        print("\n未找到任何可验证的权重文件。")


if __name__ == "__main__":
    main()