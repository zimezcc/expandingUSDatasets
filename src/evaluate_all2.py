import os
import sys
import csv
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

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

MODEL_ZOO = {
    "CMUNeXt": lambda: cmunext(),
    "UNext": lambda: UNext(),
    "CMUNet": lambda: CMUNet(),
    "MedNeXt": lambda: MedNeXt2D(in_channels=IN_CHANNELS, n_classes=NUM_CLASSES, preset='B', deep_supervision=False),
    "ConvUNeXt": lambda: ConvUNeXt(in_channels=IN_CHANNELS, n_classes=NUM_CLASSES, base_channels=32),
}

# 指标计算：返回【逐图】张量，保留未舍入浮点值
def batch_metrics_per_image(pred_logits, targets, threshold=0.5, eps=1e-6):
    """返回每个样本各自的 IoU 和 F1（形状 [B]）。"""
    preds = (torch.sigmoid(pred_logits) > threshold).float()
    preds_flat = preds.view(preds.size(0), -1)
    targets_flat = targets.view(targets.size(0), -1)

    tp = (preds_flat * targets_flat).sum(dim=1)
    fp = (preds_flat * (1 - targets_flat)).sum(dim=1)
    fn = ((1 - preds_flat) * targets_flat).sum(dim=1)

    iou_per = (tp + eps) / (tp + fp + fn + eps)
    f1_per = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    return iou_per, f1_per  # 均为 [B] 的 tensor

def _get_image_names(dataset, start_idx, batch_size):
    """从 dataset.img_paths 取本 batch 对应的图像标识，用 '类别目录/文件名' 形式。"""
    names = []
    paths = dataset.img_paths
    for i in range(start_idx, start_idx + batch_size):
        p = paths[i].replace("\\", "/")
        cls = os.path.basename(os.path.dirname(p))   # benign / malignant
        fname = os.path.basename(p)                  # 254.png
        names.append(f"{cls}/{fname}")
    return names


@torch.no_grad()
def evaluate(model, test_loader, dataset):
    """返回: (汇总 mean_iou, 汇总 mean_f1, 逐图记录列表)"""
    model.eval()
    per_image_records = []
    global_idx = 0

    for img, mask in tqdm(test_loader, desc="Testing", leave=False):
        img = img.to(DEVICE, non_blocking=True)
        mask = mask.to(DEVICE, non_blocking=True)

        out = model(img)
        iou_per, f1_per = batch_metrics_per_image(out, mask)  # [B], [B]

        bs = img.size(0)
        names = _get_image_names(dataset, global_idx, bs)

        for j in range(bs):
            per_image_records.append({
                "image_id": names[j],
                "index": global_idx + j,
                "iou": float(iou_per[j].item()),   # 未舍入原始值
                "f1": float(f1_per[j].item()),
            })
        global_idx += bs

    # 汇总：按图像逐一平均，与逐图记录严格一致、可反算
    n = len(per_image_records)
    mean_iou = sum(r["iou"] for r in per_image_records) / n if n else 0.0
    mean_f1 = sum(r["f1"] for r in per_image_records) / n if n else 0.0
    return mean_iou, mean_f1, per_image_records


def main():
    print(f"使用设备: {DEVICE}")
    test_ds = SegmentationDataset(TEST_LIST, IMG_SIZE, augment=False)
    test_loader = DataLoader(test_ds, batch_size=TEST_BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True)
    print(f"测试集大小: {len(test_ds)} 张图片")

    results = []
    per_image_dir = os.path.join(BASE_DIR, "per_image_scores")
    os.makedirs(per_image_dir, exist_ok=True)

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

            try:
                model = MODEL_ZOO[model_name]().to(DEVICE)
                checkpoint = torch.load(best_ckpt_path, map_location=DEVICE)
                if "model_state" in checkpoint:
                    model.load_state_dict(checkpoint["model_state"])
                else:
                    model.load_state_dict(checkpoint)
            except Exception as e:
                print(f"  [错误] 加载模型失败: {e}")
                continue

            val_iou, val_f1, per_image = evaluate(model, test_loader, test_ds)

            saved_f1 = checkpoint.get("best_f1", -1.0)
            saved_iou = checkpoint.get("iou_at_best_f1", -1.0)
            best_epoch = checkpoint.get("epoch", -1)

            print(f"  -> 重新计算: F1={val_f1:.4f}, IoU={val_iou:.4f}")
            print(f"  -> 训练记录: F1={saved_f1:.4f}, IoU={saved_iou:.4f} (Epoch {best_epoch})")
            if abs(val_f1 - saved_f1) > 0.01:
                print(f"  [警告] 重新计算与训练记录差异较大！")

            # 写出逐图像分数 CSV（未舍入，float 全精度）
            per_image_csv = os.path.join(per_image_dir, f"{model_name}__{exp_name}__per_image.csv")
            with open(per_image_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["image_id", "index", "iou", "f1"])
                writer.writeheader()
                writer.writerows(per_image)
            print(f"  -> 逐图分数已保存: {per_image_csv}")

            results.append({
                "train_list": exp_name,
                "model": model_name,
                "re_eval_f1": f"{val_f1:.6f}",   # 汇总表 6 位小数
                "re_eval_iou": f"{val_iou:.6f}",
                "saved_f1": f"{saved_f1:.6f}",
                "saved_iou": f"{saved_iou:.6f}",
                "best_epoch": best_epoch,
                "per_image_csv": os.path.relpath(per_image_csv, BASE_DIR),
            })

            del model
            torch.cuda.empty_cache()

    if results:
        csv_path = os.path.join(BASE_DIR, "re_evaluate_summary.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\n验证完成！汇总结果: {csv_path}")
        print(f"逐图分数目录: {per_image_dir}")
    else:
        print("\n未找到任何可验证的权重文件。")


if __name__ == "__main__":
    main()