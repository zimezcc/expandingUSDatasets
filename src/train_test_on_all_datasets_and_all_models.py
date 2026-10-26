import os
import sys
import glob
import csv
import time
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

# 让 Python 能找到 networks 目录
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from networks.CMUNeXt import cmunext, cmunext_s, cmunext_l
from dataset import SegmentationDataset
from networks.UNeXt import UNext
from networks.CMUNet import CMUNet
from networks.MedNeXt import MedNeXt2D
from networks.ConvUNeXt import ConvUNeXt

# ---------------- 随机种子（尽量可复现） ----------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

# 开启确定性会牺牲少量速度；如果更在意速度，可以注释掉下面两行
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ---------------- 分布式相关 ----------------
USE_DDP = "LOCAL_RANK" in os.environ
if USE_DDP:
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP

    dist.init_process_group(backend="nccl")
    LOCAL_RANK = int(os.environ["LOCAL_RANK"])
    WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))
    torch.cuda.set_device(LOCAL_RANK)
    DEVICE = torch.device("cuda", LOCAL_RANK)
    IS_MAIN = (LOCAL_RANK == 0)

    # 每个 rank 用不同但确定的种子，避免 4 个进程随机序列完全相同
    rank_seed = SEED + LOCAL_RANK
    random.seed(rank_seed)
    np.random.seed(rank_seed)
    torch.manual_seed(rank_seed)
    torch.cuda.manual_seed_all(rank_seed)
else:
    LOCAL_RANK = 0
    WORLD_SIZE = 1
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    IS_MAIN = True

# ---------------- 配置 ----------------
IMG_SIZE = 256
BATCH_SIZE = 32
EPOCHS = 300
LR = 1e-3
WEIGHT_DECAY = 1e-4
NUM_CLASSES = 1
NUM_WORKERS = max(1, 8 // WORLD_SIZE) if USE_DDP else 8
IN_CHANNELS = 3
TEST_BATCH_SIZE = BATCH_SIZE * 2

USE_AMP = True
print(f"使用设备: {DEVICE}, 分布式: {USE_DDP}, 进程数: {WORLD_SIZE}")
print(f"GPU 数量: {torch.cuda.device_count()}")
print(f"随机种子: {SEED}, 当前 rank 种子: {SEED + LOCAL_RANK}")

# ---------------- 损失 ----------------
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1)
        inter = (probs * targets).sum(dim=1)
        union = probs.sum(dim=1) + targets.sum(dim=1)
        dice = (2 * inter + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.bce_weight = bce_weight

    def forward(self, logits, targets):
        return self.bce_weight * self.bce(logits, targets) + \
               (1 - self.bce_weight) * self.dice(logits, targets)

# ---------------- 指标 ----------------
def batch_metrics(pred_logits, targets, threshold=0.5, eps=1e-6):
    """返回 batch 内按图平均的 (IoU, F1)"""
    preds = (torch.sigmoid(pred_logits) > threshold).float()
    preds_flat = preds.view(preds.size(0), -1)
    targets_flat = targets.view(targets.size(0), -1)

    tp = (preds_flat * targets_flat).sum(dim=1)
    fp = (preds_flat * (1 - targets_flat)).sum(dim=1)
    fn = ((1 - preds_flat) * targets_flat).sum(dim=1)

    iou_per = (tp + eps) / (tp + fp + fn + eps)
    f1_per = (2 * tp + eps) / (2 * tp + fp + fn + eps)

    return iou_per.mean().item(), f1_per.mean().item()

# ---------------- 训练 ----------------
def train_one_epoch(model, optimizer, criterion, train_loader, scaler, use_amp):
    model.train()
    total_loss = 0.0
    total_iou = 0.0
    total_f1 = 0.0
    n = 0

    for img, mask in train_loader:
        img = img.to(DEVICE, non_blocking=True)
        mask = mask.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            with torch.amp.autocast("cuda"):
                out = model(img)
                loss = criterion(out, mask)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(img)
            loss = criterion(out, mask)
            loss.backward()
            optimizer.step()

        iou_m, f1_m = batch_metrics(out.detach(), mask)
        bs = img.size(0)
        total_loss += loss.item() * bs
        total_iou += iou_m * bs
        total_f1 += f1_m * bs
        n += bs

    return total_loss / n, total_iou / n, total_f1 / n

# ---------------- 验证 ----------------
@torch.no_grad()
def evaluate(model, criterion, test_loader, scaler, use_amp):
    model.eval()
    total_loss = 0.0
    total_iou = 0.0
    total_f1 = 0.0
    n = 0

    for img, mask in test_loader:
        img = img.to(DEVICE, non_blocking=True)
        mask = mask.to(DEVICE, non_blocking=True)

        if use_amp:
            with torch.amp.autocast("cuda"):
                out = model(img)
                loss = criterion(out, mask)
        else:
            out = model(img)
            loss = criterion(out, mask)

        iou_m, f1_m = batch_metrics(out, mask)
        bs = img.size(0)
        total_loss += loss.item() * bs
        total_iou += iou_m * bs
        total_f1 += f1_m * bs
        n += bs

    return total_loss / n, total_iou / n, total_f1 / n

# ---------------- 日志 ----------------
def init_log(log_path, config=None):
    if not os.path.exists(log_path):
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("分割模型 训练日志（只记录 IoU 和 F1）\n")
            f.write("=" * 80 + "\n")
            if config:
                for k, v in config.items():
                    f.write(f"{k:20s}: {v}\n")
            f.write("-" * 80 + "\n")
            header = (
                f"{'epoch':>5s}  {'lr':>10s}  "
                f"{'tr_loss':>8s}  {'tr_iou':>8s}  {'tr_f1':>8s}  |  "
                f"{'va_loss':>8s}  {'va_iou':>8s}  {'va_f1':>8s}  {'time':>7s}\n"
            )
            f.write(header)
            f.write("-" * 80 + "\n")

def append_log(log_path, epoch, lr, tr, va, dt):
    line = (
        f"{epoch:5d}  {lr:10.3e}  "
        f"{tr[0]:8.4f}  {tr[1]:8.4f}  {tr[2]:8.4f}  |  "
        f"{va[0]:8.4f}  {va[1]:8.4f}  {va[2]:8.4f}  {dt:6.1f}s\n"
    )
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(line)

def log_best(log_path, epoch, f1, iou_at_best_f1):
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"[best] epoch={epoch}, val_f1={f1:.4f}, "
                f"iou_at_best_f1={iou_at_best_f1:.4f}\n")

def log_final(log_path, best_f1, best_epoch, iou_at_best_f1=0.0):
    with open(log_path, "a", encoding="utf-8") as f:
        f.write("-" * 80 + "\n")
        f.write(f"训练完成。最佳 F1={best_f1:.4f} (epoch {best_epoch})，"
                f"该 epoch 对应 IoU={iou_at_best_f1:.4f}\n")
        f.write("=" * 80 + "\n")

# ---------------- 主循环 ----------------
if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    TEST_LIST = os.path.join(BASE_DIR, "test.txt")

    MODEL_ZOO = {
        "CMUNeXt": lambda: cmunext(),
        "UNext":     lambda: UNext(),
        "CMUNet":    lambda: CMUNet(),
        "MedNeXt": lambda: MedNeXt2D(in_channels=IN_CHANNELS, n_classes=NUM_CLASSES,
                                       preset='B', deep_supervision=False),
        "ConvUNeXt": lambda: ConvUNeXt(in_channels=IN_CHANNELS, n_classes=NUM_CLASSES,
                                       base_channels=32),
    }

    summary_rows = []
    TRAIN_LISTS = [
       "train_data_list/train.txt",
       "train_data_list/train+normal.txt",
       "train_data_list/train+synthetic1_BW20.txt",
       "train_data_list/train+synthetic2_BW20.txt",
       "train_data_list/train+normal+synthetic2_BW0.txt",
       "train_data_list/train+normal+synthetic2_BW10.txt",
       "train_data_list/train+normal+synthetic2_BW20.txt",
       "train_data_list/train+normal+synthetic2_BW30.txt",
        "train_data_list/train+normal+synthetic2_BW40.txt"
        ]

    for TRAIN_LIST in TRAIN_LISTS:
        train_name = os.path.splitext(os.path.basename(TRAIN_LIST))[0]
        if IS_MAIN:
            print("\n" + "=" * 80)
            print(f"实验数据: {TRAIN_LIST}")
            print("=" * 80)

        train_ds = SegmentationDataset(TRAIN_LIST, IMG_SIZE, augment=True)
        test_ds = SegmentationDataset(TEST_LIST, IMG_SIZE, augment=False)

        if USE_DDP:
            train_sampler = DistributedSampler(
                train_ds, num_replicas=WORLD_SIZE, rank=LOCAL_RANK,
                shuffle=True, drop_last=True, seed=SEED,
            )
            train_loader = DataLoader(
                train_ds, batch_size=BATCH_SIZE, sampler=train_sampler,
                num_workers=NUM_WORKERS, pin_memory=True, drop_last=True,
                persistent_workers=(NUM_WORKERS > 0),
            )
        else:
            train_sampler = None
            train_loader = DataLoader(
                train_ds, batch_size=BATCH_SIZE, shuffle=True,
                num_workers=NUM_WORKERS, pin_memory=True, drop_last=True,
                persistent_workers=(NUM_WORKERS > 0),
            )

        test_loader = DataLoader(
            test_ds, batch_size=TEST_BATCH_SIZE, shuffle=False,
            num_workers=NUM_WORKERS, pin_memory=True,
            persistent_workers=(NUM_WORKERS > 0),
        )

        if IS_MAIN:
            print(f"训练集: {len(train_ds)} 张，测试集: {len(test_ds)} 张")
            print(f"每卡 batch: {BATCH_SIZE}，全局 batch: {BATCH_SIZE * WORLD_SIZE}")

        for model_name, model_fn in MODEL_ZOO.items():
            if IS_MAIN:
                print("\n" + "-" * 80)
                print(f"模型: {model_name}")
                print("-" * 80)

            SAVE_DIR = os.path.join("checkpoints", "1026", model_name, train_name)
            LOG_DIR = os.path.join("logs", "1026", model_name, train_name)

            if IS_MAIN:
                os.makedirs(SAVE_DIR, exist_ok=True)
                os.makedirs(LOG_DIR, exist_ok=True)

            if USE_DDP:
                dist.barrier()

            LOG_PATH = os.path.join(LOG_DIR, "train_log.txt")
            best_ckpt_path = os.path.join(SAVE_DIR, "best.pth")

            if os.path.exists(best_ckpt_path):
                if IS_MAIN:
                    print(f"跳过 {model_name}（已存在 {best_ckpt_path}）")
                continue

            try:
                model = model_fn().to(DEVICE)
            except Exception as e:
                if IS_MAIN:
                    print(f"[错误] 构建模型 {model_name} 失败: {e}")
                continue

            if USE_DDP:
                model = DDP(model, device_ids=[LOCAL_RANK])
            elif torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)

            criterion = BCEDiceLoss(bce_weight=0.5).to(DEVICE)
            optimizer = Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
            scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

            use_amp = USE_AMP and (DEVICE.type == "cuda")
            scaler = torch.amp.GradScaler("cuda") if use_amp else None

            if IS_MAIN:
                config = {
                    "MODEL": model_name,
                    "TRAIN_LIST": TRAIN_LIST,
                    "TEST_LIST": TEST_LIST,
                    "IMG_SIZE": IMG_SIZE,
                    "BATCH_SIZE_PER_GPU": BATCH_SIZE,
                    "GLOBAL_BATCH": BATCH_SIZE * WORLD_SIZE,
                    "EPOCHS": EPOCHS,
                    "LR": LR,
                    "WEIGHT_DECAY": WEIGHT_DECAY,
                    "NUM_CLASSES": NUM_CLASSES,
                    "IN_CHANNELS": IN_CHANNELS,
                    "DEVICE": str(DEVICE),
                    "NUM_WORKERS": NUM_WORKERS,
                    "USE_AMP": use_amp,
                    "DDP": USE_DDP,
                    "SEED": SEED,
                    "训练集数量": len(train_ds),
                    "测试集数量": len(test_ds),
                }
                init_log(LOG_PATH, config)

            best_f1 = 0.0
            best_epoch = 0
            best_iou_at_best_f1 = 0.0

            try:
                for epoch in range(1, EPOCHS + 1):
                    if USE_DDP and train_sampler is not None:
                        train_sampler.set_epoch(epoch)

                    t0 = time.time()
                    tr = train_one_epoch(model, optimizer, criterion,
                                         train_loader, scaler, use_amp)
                    va = evaluate(model, criterion, test_loader, scaler, use_amp)
                    scheduler.step()
                    dt = time.time() - t0
                    lr_now = optimizer.param_groups[0]["lr"]

                    if IS_MAIN:
                        print(f"Epoch [{epoch}/{EPOCHS}] ({dt:.1f}s) lr={lr_now:.2e} | "
                              f"tr_loss={tr[0]:.4f} IoU={tr[1]:.4f} F1={tr[2]:.4f} | "
                              f"va_loss={va[0]:.4f} IoU={va[1]:.4f} F1={va[2]:.4f}")

                        append_log(LOG_PATH, epoch, lr_now, tr, va, dt)

                    if va[2] > best_f1:
                        best_f1, best_epoch = va[2], epoch
                        best_iou_at_best_f1 = va[1]

                        if IS_MAIN:
                            model_to_save = model.module if hasattr(model, "module") else model
                            torch.save({
                                "epoch": epoch,
                                "model_name": model_name,
                                "model_state": model_to_save.state_dict(),
                                "optimizer_state": optimizer.state_dict(),
                                "best_f1": best_f1,
                                "iou_at_best_f1": best_iou_at_best_f1,
                                "val_metrics": {"loss": va[0], "iou": va[1], "f1": va[2]},
                            }, best_ckpt_path)
                            print(f"  -> 保存最佳模型 (val F1={best_f1:.4f}, "
                                  f"IoU={best_iou_at_best_f1:.4f}) 到 {best_ckpt_path}")
                            log_best(LOG_PATH, epoch, best_f1, best_iou_at_best_f1)

                if IS_MAIN:
                    print(f"[{model_name}] 完成。最佳 F1={best_f1:.4f} "
                          f"(epoch {best_epoch}, 对应 IoU={best_iou_at_best_f1:.4f})")
                    log_final(LOG_PATH, best_f1, best_epoch, best_iou_at_best_f1)

                    summary_rows.append({
                        "train_list": train_name,
                        "model": model_name,
                        "best_f1": f"{best_f1:.4f}",
                        "best_epoch": best_epoch,
                        "iou_at_best_f1": f"{best_iou_at_best_f1:.4f}",
                    })

            except Exception as e:
                if IS_MAIN:
                    print(f"[错误] 模型 {model_name} 训练失败: {e}")
                    log_final(LOG_PATH, -1.0, -1, -1.0)

            finally:
                del model, optimizer, scheduler, scaler
                if DEVICE.type == "cuda":
                    torch.cuda.empty_cache()

    if IS_MAIN and summary_rows:
        summary_path = os.path.join("logs", "1026", "summary.csv")
        os.makedirs(os.path.dirname(summary_path), exist_ok=True)
        file_exists = os.path.exists(summary_path)
        with open(summary_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["train_list", "model",
                                                   "best_f1", "best_epoch",
                                                   "iou_at_best_f1"])
            if not file_exists:
                writer.writeheader()
            writer.writerows(summary_rows)
        print(f"\n汇总结果已写入: {summary_path}")

    if IS_MAIN:
        print("\n所有实验完成。")

    if USE_DDP:
        dist.destroy_process_group()