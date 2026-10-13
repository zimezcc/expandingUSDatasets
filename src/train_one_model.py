import os
import sys
import glob
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

# 让 Python 能找到 networks 目录
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from networks.CMUNeXt import cmunext, cmunext_s, cmunext_l
from dataset import SegmentationDataset
from networks.UNeXt import  UNext
from networks.CMUNet import CMUNet
from networks.MedNeXt import MedNeXt2D
from networks.ConvUNeXt import ConvUNeXt

# ---------------- 配置 ----------------
IMG_SIZE = 256
BATCH_SIZE = 8
EPOCHS = 1
LR = 1e-3
WEIGHT_DECAY = 1e-4
NUM_CLASSES = 1
NUM_WORKERS = 0

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP = False
print(f"使用设备: {DEVICE}")

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


# ---------------- 指标：只算 IoU 和 F1（macro，按图平均） ----------------
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

        optimizer.zero_grad()
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
            f.write("CMUNeXt 训练日志（只记录 IoU 和 F1）\n")
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


def log_best(log_path, epoch, f1):
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"[best] epoch={epoch}, val_f1={f1:.4f}\n")


def log_final(log_path, best_f1, best_epoch):
    with open(log_path, "a", encoding="utf-8") as f:
        f.write("-" * 80 + "\n")
        f.write(f"训练完成。最佳 F1={best_f1:.4f} (epoch {best_epoch})\n")
        f.write("=" * 80 + "\n")

# ---------------- 主循环 ----------------
if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    TEST_LIST = os.path.join(BASE_DIR, "test.txt")

    TRAIN_LISTS = sorted(glob.glob(os.path.join(BASE_DIR, "train_data_list", "*.txt")))
    for TRAIN_LIST in TRAIN_LISTS:
        print("\n" + "=" * 80)
        print(f"实验: {TRAIN_LIST}")
        print("=" * 80)

        # model = cmunext_s().to(DEVICE)
        # model = UNext().to(DEVICE)
        # model = CMUNet().to(DEVICE)
        #model = MedNeXt2D().to(DEVICE)
        model = ConvUNeXt().to(DEVICE)

        criterion = BCEDiceLoss(bce_weight=0.5)
        optimizer = Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

        use_amp = USE_AMP and (DEVICE == "cuda")
        scaler = torch.amp.GradScaler("cuda") if use_amp else None

        # ---- 数据 ----
        train_ds = SegmentationDataset(TRAIN_LIST, IMG_SIZE, augment=True)
        test_ds = SegmentationDataset(TEST_LIST, IMG_SIZE, augment=False)

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                  num_workers=NUM_WORKERS, pin_memory=True,
                                  drop_last=True)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                                 num_workers=NUM_WORKERS, pin_memory=True)

        print(f"训练集: {len(train_ds)} 张，测试集: {len(test_ds)} 张")

        # ---- 输出路径 ----
        train_name = os.path.splitext(os.path.basename(TRAIN_LIST))[0]
        SAVE_DIR = os.path.join("checkpoints", "1007", "CMUNet", train_name)
        LOG_DIR = os.path.join("logs", "1007", "CMUNet", train_name)
        os.makedirs(SAVE_DIR, exist_ok=True)
        os.makedirs(LOG_DIR, exist_ok=True)
        LOG_PATH = os.path.join(LOG_DIR, "train_log.txt")

        config = {
            "TRAIN_LIST": TRAIN_LIST,
            "TEST_LIST": TEST_LIST,
            "IMG_SIZE": IMG_SIZE,
            "BATCH_SIZE": BATCH_SIZE,
            "EPOCHS": EPOCHS,
            "LR": LR,
            "WEIGHT_DECAY": WEIGHT_DECAY,
            "NUM_CLASSES": NUM_CLASSES,
            "DEVICE": DEVICE,
            "训练集数量": len(train_ds),
            "测试集数量": len(test_ds),
        }
        init_log(LOG_PATH, config)

        best_f1 = 0.0
        best_epoch = 0

        for epoch in range(1, EPOCHS + 1):
            t0 = time.time()

            tr = train_one_epoch(model, optimizer, criterion,
                                 train_loader, scaler, use_amp)   # (loss, iou, f1)
            va = evaluate(model, criterion, test_loader, scaler, use_amp)
            scheduler.step()
            dt = time.time() - t0
            lr_now = optimizer.param_groups[0]["lr"]

            print(f"\nEpoch [{epoch}/{EPOCHS}]  ({dt:.1f}s)  lr={lr_now:.2e}")
            print(f"  train  loss={tr[0]:.4f}  IoU={tr[1]:.4f}  F1={tr[2]:.4f}")
            print(f"  val    loss={va[0]:.4f}  IoU={va[1]:.4f}  F1={va[2]:.4f}")

            # 写 txt
            append_log(LOG_PATH, epoch, lr_now, tr, va, dt)

            # 保存最佳模型（按验证集 F1）
            if va[2] > best_f1:
                best_f1 = va[2]
                best_epoch = epoch
                save_path = os.path.join(SAVE_DIR, "best.pth")
                torch.save({
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "best_f1": best_f1,
                    "val_metrics": {
                        "loss": va[0],
                        "iou": va[1],
                        "f1": va[2],
                    },
                }, save_path)
                print(f"  -> 保存最佳模型 (val F1={best_f1:.4f}) 到 {save_path}")
                log_best(LOG_PATH, epoch, best_f1)

        print(f"\n训练完成。最佳 F1={best_f1:.4f} (epoch {best_epoch})")
        print(f"训练日志: {LOG_PATH}")
        log_final(LOG_PATH, best_f1, best_epoch)

    print("\n所有实验完成。")