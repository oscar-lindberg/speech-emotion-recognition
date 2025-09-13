#!/usr/bin/env python3
"""
Train a SER model on precomputed features.
"""
from __future__ import annotations
import os
import math
import json
import time
import argparse
import datetime as dt
from typing import Dict, List, Tuple, Optional, Iterable
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from speechEmotionRecognition.config_loader import load_config
from speechEmotionRecognition.train.audio_dataset import AudioDataset
try:
    from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix
    HAVE_SKLEARN = True
except Exception:
    HAVE_SKLEARN = False


# ──────────────────────────────────────────────────────────────────────────────
# Pretty printing
# ──────────────────────────────────────────────────────────────────────────────

def _rule(title: str = "") -> None:
    line = "─" * 78
    print(f"{line}\n{title}\n{line}" if title else line)

def _kv(label: str, value) -> None:
    print(f"  {label:<22}: {value}")

def _ok(msg: str) -> None:
    print(f"[OK] {msg}")


# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def pick_device(pref: Optional[str]) -> torch.device:
    if pref and pref != "auto":
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def now_str() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")

def ensure_dir(d: str) -> None:
    os.makedirs(d, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# Class weights and losses
# ──────────────────────────────────────────────────────────────────────────────

def _labels_numpy(ds: AudioDataset, num_classes: int) -> np.ndarray:
    # Pull labels by iterating the dataset (simple, robust).
    lbls: List[int] = [int(y) for _, y in ds]
    arr = np.asarray(lbls, dtype=np.int64)
    if arr.size and arr.max(initial=0) >= num_classes:
        arr = np.clip(arr, 0, num_classes - 1)
    return arr

def class_weights_from_counts(counts: np.ndarray, mode: str) -> np.ndarray:
    counts = counts.astype(np.float64)
    if mode == "inverse_freq":
        w = 1.0 / np.maximum(counts, 1.0)
    elif mode == "effective":
        N = float(counts.sum())
        beta = (N - 1.0) / N if N > 1 else 0.0
        w = (1.0 - beta) / (1.0 - np.power(beta, np.maximum(counts, 1.0)))
    else:
        w = np.ones_like(counts)
    return (w / (w.mean() + 1e-12)).astype(np.float32)

class FocalLoss(nn.Module):
    def __init__(self, weight: Optional[torch.Tensor], gamma: float = 1.5, label_smoothing: float = 0.0):
        super().__init__()
        self.gamma = float(gamma)
        self.ce = nn.CrossEntropyLoss(weight=weight, label_smoothing=label_smoothing, reduction="none")

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce_i = self.ce(logits, target)                              # [B]
        with torch.no_grad():
            p = torch.softmax(logits, dim=-1)                       # [B, C]
            pt = torch.gather(p, 1, target.unsqueeze(1)).squeeze(1) # [B]
            pt = pt.clamp_(1e-6, 1.0)
        loss = ((1.0 - pt) ** self.gamma) * ce_i                    # [B]
        return loss.mean()


# ──────────────────────────────────────────────────────────────────────────────
# Model factory
# ──────────────────────────────────────────────────────────────────────────────

def build_model(cfg: Dict, feature_dims: Dict[str, int], num_classes: int) -> nn.Module:
    name = str(cfg["model"]["name"]).lower()
    if name == "spectral_crnn":
        # feature_dims: {name: F_i}, values are frequency rows per branch
        from speechEmotionRecognition.models.spectral_crnn import SpectralCRNN
        mcfg = cfg["model"].get("spectral_crnn", {})
        return SpectralCRNN(
            feature_shapes=feature_dims,
            num_classes=num_classes,
            branch_out_dim=int(mcfg.get("branch_out_dim", 128)),
            rnn_hidden_size=int(mcfg.get("rnn_hidden_size", 128)),
            rnn_layers=int(mcfg.get("rnn_layers", 1)),
            cnn_dropout=float(mcfg.get("cnn_dropout", 0.2)),
            fusion_dropout=float(mcfg.get("fusion_dropout", 0.2)),
            dropout=float(mcfg.get("dropout", 0.4))
        )

    elif name == "ssl_spectral_moe":
        from speechEmotionRecognition.models.ssl_spectral_moe import SpectralSSLMoE
        mcfg = cfg["model"].get("moe", {})
        return SpectralSSLMoE(
            feature_shapes=feature_dims,
            num_classes=num_classes,
            # SSL
            ssl_proj_dim=mcfg.get("ssl_proj_dim", 256),
            ssl_encoder_layers=int(mcfg.get("ssl_encoder_layers", 1)),
            ssl_encoder_nhead=int(mcfg.get("ssl_encoder_nhead", 4)),
            ssl_encoder_ff=int(mcfg.get("ssl_encoder_ff", 512)),
            ssl_encoder_dropout=float(mcfg.get("ssl_encoder_dropout", 0.10)),
            # Spectral experts
            spec_hidden=int(mcfg.get("spec_hidden", 128)),
            spec_conv_layers=int(mcfg.get("spec_conv_layers", 2)),
            spec_kernel=int(mcfg.get("spec_kernel", 5)),
            spec_encoder_layers=int(mcfg.get("spec_encoder_layers", 0)),
            spec_encoder_nhead=int(mcfg.get("spec_encoder_nhead", 4)),
            spec_encoder_ff=int(mcfg.get("spec_encoder_ff", 512)),
            spec_encoder_dropout=float(mcfg.get("spec_encoder_dropout", 0.10)),
            # Pooling / fusion / gating / clf
            attn_hidden=int(mcfg.get("attn_hidden", 128)),
            fusion_dim=int(mcfg.get("fusion_dim", 256)),
            mlp_hidden=int(mcfg.get("mlp_hidden", 256)),
            gate_hidden=int(mcfg.get("gate_hidden", 128)),
            gate_temperature=float(mcfg.get("gate_temperature", 1.0)),
            gate_dropout=float(mcfg.get("gate_dropout", 0.10)),
            dropout=float(mcfg.get("dropout", 0.4)),
            # Aux
            use_aux=bool(mcfg.get("use_aux", False)),
        )

    raise ValueError(f"Unknown model name: {cfg['model']['name']}")


# ──────────────────────────────────────────────────────────────────────────────
# Data
# ──────────────────────────────────────────────────────────────────────────────

def build_loaders(cfg: Dict, device: torch.device):
    features = list(cfg["model"]["features"])
    label_map = dict(cfg["emotion_labels"])

    train_ds = AudioDataset(
        cfg["project"]["train_csv"],
        cfg["project"]["features_dir"],
        features,
        label_map,
        mode="train",
        specaugment_cfg=(cfg["specaugment"] if bool(cfg["specaugment"]["enabled"]) else None),
    )
    val_ds = AudioDataset(
        cfg["project"]["val_csv"],
        cfg["project"]["features_dir"],
        features,
        label_map,
        mode="val",
        specaugment_cfg=None,
    )
    test_ds = AudioDataset(
        cfg["project"]["test_csv"],
        cfg["project"]["features_dir"],
        features,
        label_map,
        mode="test",
        specaugment_cfg=None,
    )

    num_workers = int(cfg["train"]["num_workers"])
    persistent_workers = num_workers > 0
    pin_memory = (device.type == "cuda")

    # Optional weighted sampler for class imbalance
    use_weighted_sampler = bool(cfg["train"]["use_weighted_sampler"])
    if use_weighted_sampler:
        labels = _labels_numpy(train_ds, num_classes=len(label_map))
        counts = np.bincount(labels, minlength=len(label_map))
        inv = 1.0 / np.maximum(counts, 1.0)
        sample_w = inv[labels]
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sample_w, dtype=torch.double),
            num_samples=len(train_ds),
            replacement=True,
        )
        shuffle_train = False
    else:
        sampler = None
        shuffle_train = True

    batch_size = int(cfg["train"]["batch_size"])
    common = dict(num_workers=num_workers, pin_memory=pin_memory,
                  persistent_workers=persistent_workers, drop_last=False)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size,
        shuffle=shuffle_train if sampler is None else False,
        sampler=sampler, **common
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **common)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, **common)
    return train_loader, val_loader, test_loader


# ──────────────────────────────────────────────────────────────────────────────
# Optimizer / Scheduler
# ──────────────────────────────────────────────────────────────────────────────

def _group_params_for_adamw(named_params: Iterable, weight_decay: float):
    decay, no_decay = [], []
    for n, p in named_params:
        if not p.requires_grad:
            continue
        if p.ndim == 1 or n.endswith(".bias") or "norm" in n.lower() or "bn" in n.lower():
            no_decay.append(p)
        else:
            decay.append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]

def build_optim_scheduler(model: nn.Module, cfg: Dict):
    train_cfg = cfg["train"]
    wd = float(train_cfg["weight_decay"])
    lr = float(train_cfg.get("learning_rate", train_cfg.get("lr", 1e-4)))
    epochs = int(train_cfg["epochs"])
    optimizer_name = str(train_cfg.get("optimizer", "adamw")).lower()
    scheduler_name = str(train_cfg.get("scheduler", "cosine")).lower()
    warmup_pct = float(train_cfg.get("warmup_pct", train_cfg.get("warmup_ratio", 0.1)))

    if optimizer_name == "adamw":
        opt = torch.optim.AdamW(_group_params_for_adamw(model.named_parameters(), wd), lr=lr)
    elif optimizer_name == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    if scheduler_name == "cosine":
        total_epochs = max(1, epochs)
        warmup_epochs = max(1, int(round(warmup_pct * total_epochs)))

        def lr_lambda(epoch_idx: int):
            e = epoch_idx + 1
            if e <= warmup_epochs:
                return e / float(warmup_epochs)
            progress = (e - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)
    elif scheduler_name in ("none", "constant"):
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda _: 1.0)
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")

    return opt, sched


# ──────────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────────

def _compute_stats(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    if HAVE_SKLEARN:
        acc = float(accuracy_score(y_true, y_pred))
        f1m = float(f1_score(y_true, y_pred, average="macro"))
        uar = float(recall_score(y_true, y_pred, average="macro"))
    else:
        acc = float((y_true == y_pred).mean())
        C = int(y_true.max(initial=0) + 1)
        recs = []
        for c in range(C):
            idx = (y_true == c)
            recs.append(float((y_pred[idx] == c).mean()) if idx.any() else 0.0)
        uar = float(np.mean(recs)); f1m = uar
    return {"acc": acc, "macro_f1": f1m, "uar": uar}

def _per_class_report(y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str]) -> Dict[str, Dict[str, float]]:
    C = len(class_names)
    report: Dict[str, Dict[str, float]] = {}
    for c in range(C):
        tn = int(np.sum((y_true != c) & (y_pred != c)))
        tp = int(np.sum((y_true == c) & (y_pred == c)))
        fp = int(np.sum((y_true != c) & (y_pred == c)))
        fn = int(np.sum((y_true == c) & (y_pred != c)))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        report[class_names[c]] = {
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "support": int(np.sum(y_true == c)),
        }
    return report

@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    metric: str,
    loss_fn: Optional[nn.Module] = None,
    want_preds: bool = False,
    use_amp: bool = False,
) -> Tuple[float, Dict]:
    """Returns (selection_metric, stats); stats has acc/macro_f1/uar/loss and optionally y_true/y_pred."""
    model.eval()
    all_logits, all_y = [], []
    running_loss = 0.0
    n_batches = 0

    # Autocast only for CUDA; MPS/CPU run in fp32.
    ctx = torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp and device.type == "cuda")
    with ctx:
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            out = model(xb)
            logits = out[0] if isinstance(out, (tuple, list)) else out
            if loss_fn is not None:
                running_loss += float(loss_fn(logits, yb).item())
                n_batches += 1
            all_logits.append(logits.detach().cpu())
            all_y.append(yb.detach().cpu())

    logits = torch.cat(all_logits, dim=0)
    y_true = torch.cat(all_y, dim=0).numpy()
    y_pred = logits.argmax(dim=1).numpy()

    stats = _compute_stats(y_true, y_pred)
    stats["loss"] = running_loss / max(1, n_batches) if n_batches > 0 else float("nan")
    if want_preds:
        stats["y_true"] = y_true
        stats["y_pred"] = y_pred
    sel = {"uar": stats["uar"], "macro_f1": stats["macro_f1"], "acc": stats["acc"]}[metric]
    return sel, stats


# ──────────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.amp.GradScaler],
    device: torch.device,
    loss_fn: nn.Module,
    grad_clip: float
) -> Tuple[float, Dict[str, float]]:
    """One epoch; CUDA uses AMP, MPS/CPU run fp32."""
    model.train()
    running = 0.0
    n_batches = 0
    y_true_all: List[int] = []
    y_pred_all: List[int] = []

    pbar = tqdm(loader, leave=False, desc="train", ncols=100)

    for xb, yb in pbar:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            # CUDA path: autocast + GradScaler (MPS/CPU do not enter here)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                logits = model(xb)
                loss = loss_fn(logits, yb)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            # CPU / MPS (or CUDA without AMP): plain fp32
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()

        running += float(loss.item()); n_batches += 1
        y_pred = logits.argmax(dim=1)
        y_true_all.extend(yb.detach().cpu().tolist())
        y_pred_all.extend(y_pred.detach().cpu().tolist())
        pbar.set_postfix(loss=f"{running / n_batches:.4f}")

    train_loss = running / max(1, n_batches)
    tr_stats = _compute_stats(np.asarray(y_true_all), np.asarray(y_pred_all))
    tr_stats["loss"] = train_loss
    return train_loss, tr_stats


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Train a SER model on precomputed features.")
    p.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    p.add_argument("--epochs", type=int, default=None, help="Override epochs.")
    p.add_argument("--run-name", type=str, default=None, help="Optional run name suffix.")
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 0)))
    device = pick_device(cfg.get("device", "auto"))

    # Data and shapes
    train_loader, val_loader, test_loader = build_loaders(cfg, device)
    feature_shapes_ds = train_loader.dataset.feature_shapes  # {name: (F_i, T)}
    feature_dims = {k: int(v[0]) for k, v in feature_shapes_ds.items()}
    num_classes = int(len(cfg["emotion_labels"]))
    inv_label = {int(v): str(k) for k, v in cfg["emotion_labels"].items()}
    class_names = [inv_label[i] for i in range(len(inv_label))]

    # Model
    model = build_model(cfg, feature_dims, num_classes).to(device)

    # Optimizer and scheduler
    opt, sched = build_optim_scheduler(model, cfg)

    # Class weights
    if bool(cfg["train"]["class_weighted_loss"]):
        labels = _labels_numpy(train_loader.dataset, num_classes)
        counts = np.bincount(labels, minlength=num_classes)
        weights_np = class_weights_from_counts(counts, mode=str(cfg["train"]["class_balancing"]))
        class_weights = torch.tensor(weights_np, dtype=torch.float32, device=device)
    else:
        class_weights = None

    # Loss
    loss_type = str(cfg["train"].get("loss", "ce")).lower()
    label_smoothing = float(cfg["train"].get("label_smoothing", 0.0))
    if loss_type == "focal":
        focal_gamma = float(cfg["train"].get("focal_gamma", 1.5))
        loss_fn = FocalLoss(weight=class_weights, gamma=focal_gamma, label_smoothing=label_smoothing)
    else:
        loss_fn = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)

    # AMP: CUDA only (MPS/CPU run fp32)
    want_amp = bool(cfg["train"].get("amp", True))
    use_amp = want_amp and (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    # Run dir + snapshot cfg
    run_root = cfg["project"]["run_dir"]
    tag = args.run_name or cfg["model"]["name"]
    run_dir = os.path.join(run_root, f"{now_str()}_{tag}")
    ensure_dir(run_dir)
    with open(os.path.join(run_dir, "cfg.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    # Training loop
    metric_name = str(cfg["train"]["val_metric"]).lower()  # 'uar' | 'macro_f1' | 'acc'
    best_score = -1.0
    best_epoch = -1
    patience = int(cfg["train"]["early_stopping_patience"])
    epochs = int(args.epochs if args.epochs is not None else cfg["train"]["epochs"])

    history: List[Dict] = []

    _rule("TRAIN")
    _kv("device", device)
    _kv("model", cfg["model"]["name"])
    _kv("features", cfg["model"]["features"])
    _kv("metric", metric_name)
    _kv("run_dir", run_dir)
    _rule()

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_stats = train_one_epoch(
            model, train_loader, opt, scaler, device, loss_fn,
            grad_clip=float(cfg["train"]["grad_clip"])
        )
        sched.step()

        val_sel, val_stats = evaluate(
            model, val_loader, device, metric=metric_name, loss_fn=loss_fn, use_amp=use_amp
        )

        lr_cur = opt.param_groups[0]["lr"]
        dt_epoch = time.time() - t0

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={tr_stats['loss']:.4f} | train_acc={tr_stats['acc']:.4f} | "
            f"train_f1={tr_stats['macro_f1']:.4f} | train_uar={tr_stats['uar']:.4f} || "
            f"val_loss={val_stats['loss']:.4f} | val_acc={val_stats['acc']:.4f} | "
            f"val_f1={val_stats['macro_f1']:.4f} | val_uar={val_stats['uar']:.4f} | "
            f"lr={lr_cur:.8f} | time={dt_epoch:.1f}s"
        )

        # Save best
        if val_sel > best_score:
            best_score = val_sel
            best_epoch = epoch
            torch.save(
                {"model_state_dict": model.state_dict(), "epoch": epoch, "best": best_score, "cfg": cfg},
                os.path.join(run_dir, "best.pth"),
            )

        # Record history
        history.append({
            "epoch": epoch,
            "lr": lr_cur,
            "time_sec": dt_epoch,
            "train_loss": tr_stats["loss"],
            "train_acc": tr_stats["acc"],
            "train_macro_f1": tr_stats["macro_f1"],
            "train_uar": tr_stats["uar"],
            "val_loss": val_stats["loss"],
            "val_acc": val_stats["acc"],
            "val_macro_f1": val_stats["macro_f1"],
            "val_uar": val_stats["uar"],
            "best_epoch": best_epoch,
            f"best_{metric_name}": best_score,
        })

        # Early stopping
        if (epoch - best_epoch) >= patience:
            print(f"Early stopping at epoch {epoch} (best @ {best_epoch}, {metric_name}={best_score:.4f})")
            break

    _ok(f"best {metric_name} = {best_score:.4f} @ epoch {best_epoch}")

    # Save last checkpoint and history
    torch.save(
        {"model_state_dict": model.state_dict(), "epoch": history[-1]["epoch"], "cfg": cfg},
        os.path.join(run_dir, "last.pth")
    )

    import csv as _csv
    with open(os.path.join(run_dir, "history.csv"), "w", newline="") as f:
        writer = _csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    # Final test (with best checkpoint)
    ckpt_path = os.path.join(run_dir, "best.pth")
    if os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state["model_state_dict"])

    test_sel, test_stats = evaluate(
        model, test_loader, device, metric=metric_name, loss_fn=loss_fn, want_preds=True, use_amp=use_amp
    )
    y_true = test_stats.pop("y_true"); y_pred = test_stats.pop("y_pred")

    # Confusion matrix + per-class report
    if HAVE_SKLEARN:
        cm = confusion_matrix(y_true, y_pred)
    else:
        C = len(class_names)
        cm = np.zeros((C, C), dtype=np.int64)
        for t, p in zip(y_true, y_pred):
            cm[int(t), int(p)] += 1
    per_class = _per_class_report(y_true, y_pred, class_names)

    # Save test report
    test_report = {
        "metric": metric_name,
        "selection_value": float(test_sel),
        "macro": {
            "acc": float(test_stats["acc"]),
            "macro_f1": float(test_stats["macro_f1"]),
            "uar": float(test_stats["uar"]),
            "loss": float(test_stats["loss"]),
        },
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "classes": class_names,
        "best_epoch": int(best_epoch),
        f"best_{metric_name}": float(best_score),
    }
    with open(os.path.join(run_dir, "test_metrics.json"), "w") as f:
        json.dump(test_report, f, indent=2)

    # Also save confusion matrix CSV (readable)
    import csv as _csv2
    with open(os.path.join(run_dir, "confusion_matrix.csv"), "w", newline="") as f:
        w = _csv2.writer(f)
        w.writerow([""] + class_names)
        for i, row in enumerate(cm.tolist()):
            w.writerow([class_names[i]] + row)

    # Pretty TEST print
    _rule("TEST")
    _kv(f"test_{metric_name}", f"{test_sel:.4f}")
    _kv("test_acc", f"{test_stats['acc']:.4f}")
    _kv("test_macro_f1", f"{test_stats['macro_f1']:.4f}")
    _kv("test_uar", f"{test_stats['uar']:.4f}")
    _kv("test_loss", f"{test_stats['loss']:.4f}")
    print("Per-class (precision / recall / f1 / support):")
    for cls in class_names:
        r = per_class[cls]
        print(f"  {cls:<10} p={r['precision']:.3f} r={r['recall']:.3f} f1={r['f1']:.3f} n={r['support']}")
    print("Confusion matrix:")
    header = " ".join([f"{c[:4]:>4}" for c in class_names])
    print(f"      {header}")
    for i, row in enumerate(cm):
        row_str = " ".join([f"{v:>4d}" for v in row])
        print(f"{class_names[i][:4]:>4}  {row_str}")
    _rule()
    _ok(f"artifacts → {run_dir}")


if __name__ == "__main__":
    main()
