from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from tqdm import tqdm

from .analysis import MisclassifiedSample, build_error_analysis, build_model_profile
from .config import ExperimentConfig
from .data import DatasetBundle, build_dataloaders
from .models import VisionTransformer
from .plotting import plot_confusion_matrix, plot_history, plot_misclassified, plot_per_class_accuracy
from .report import write_experiment_report
from .utils import autocast_context, dump_csv, dump_json, ensure_dir, resolve_device, set_seed


@dataclass(slots=True)
class EpochResult:
    loss: float
    accuracy: float


@dataclass(slots=True)
class ExperimentResult:
    run_dir: Path
    best_val_acc: float
    best_epoch: int
    summary: dict


class WarmupCosineScheduler:
    def __init__(self, optimizer: torch.optim.Optimizer, warmup_steps: int, total_steps: int, base_lr: float) -> None:
        self.optimizer = optimizer
        self.warmup_steps = max(0, warmup_steps)
        self.total_steps = max(total_steps, 1)
        self.base_lr = base_lr
        self.step_count = 0

    def step(self) -> None:
        self.step_count += 1
        if self.step_count <= self.warmup_steps and self.warmup_steps > 0:
            lr = self.base_lr * self.step_count / self.warmup_steps
        else:
            progress = (self.step_count - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            lr = 0.5 * self.base_lr * (1.0 + torch.cos(torch.tensor(progress * torch.pi)).item())
        for group in self.optimizer.param_groups:
            group["lr"] = lr


def _unpack_batch(batch):
    if len(batch) == 3:
        images, targets, indices = batch
        return images, targets, indices
    images, targets = batch
    return images, targets, None


def _accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    predictions = logits.argmax(dim=1)
    return (predictions == targets).float().mean().item()


def train_one_epoch(
    model: nn.Module,
    loader,
    criterion,
    optimizer,
    scheduler,
    device: torch.device,
    amp: bool,
    grad_clip_norm: float | None,
) -> EpochResult:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    scaler = torch.cuda.amp.GradScaler(enabled=amp and device.type == "cuda")

    for batch in tqdm(loader, desc="train", leave=False):
        images, targets, _ = _unpack_batch(batch)
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, amp):
            logits = model(images)
            loss = criterion(logits, targets)

        scaler.scale(loss).backward()
        if grad_clip_norm is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        batch_size = targets.size(0)
        total_examples += batch_size
        total_loss += loss.item() * batch_size
        total_correct += int((logits.argmax(dim=1) == targets).sum().item())

    return EpochResult(loss=total_loss / total_examples, accuracy=total_correct / total_examples)


@torch.no_grad()
def evaluate(model: nn.Module, loader, criterion, device: torch.device, amp: bool, max_misclassified: int) -> tuple[EpochResult, list[int], list[int], list[MisclassifiedSample]]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    y_true: list[int] = []
    y_pred: list[int] = []
    mistakes: list[MisclassifiedSample] = []

    for batch in tqdm(loader, desc="eval", leave=False):
        images, targets, _ = _unpack_batch(batch)
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with autocast_context(device, amp):
            logits = model(images)
            loss = criterion(logits, targets)

        probs = logits.softmax(dim=1)
        preds = probs.argmax(dim=1)
        batch_size = targets.size(0)
        total_examples += batch_size
        total_loss += loss.item() * batch_size
        total_correct += int((preds == targets).sum().item())
        y_true.extend(targets.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())

        wrong_mask = preds != targets
        if wrong_mask.any() and len(mistakes) < max_misclassified:
            wrong_images = images[wrong_mask].detach().cpu()
            wrong_targets = targets[wrong_mask].detach().cpu().tolist()
            wrong_preds = preds[wrong_mask].detach().cpu().tolist()
            wrong_conf = probs[wrong_mask, preds[wrong_mask]].detach().cpu().tolist()
            for img, tgt, pred, conf in zip(wrong_images, wrong_targets, wrong_preds, wrong_conf):
                mistakes.append(MisclassifiedSample(img, int(tgt), int(pred), float(conf)))
                if len(mistakes) >= max_misclassified:
                    break

    return EpochResult(loss=total_loss / total_examples, accuracy=total_correct / total_examples), y_true, y_pred, mistakes


def run_experiment(cfg: ExperimentConfig) -> ExperimentResult:
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    cfg.data.image_size = cfg.model.image_size

    run_dir = ensure_dir(Path(cfg.output_dir) / cfg.run_name)
    plots_dir = ensure_dir(run_dir / "plots")
    ckpt_dir = ensure_dir(run_dir / "checkpoints")

    bundle: DatasetBundle = build_dataloaders(cfg.data, cfg.training)
    cfg.model.num_classes = bundle.num_classes
    model = VisionTransformer(cfg.model).to(device)
    profile = build_model_profile(model, cfg.model)

    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.training.label_smoothing)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )
    total_steps = cfg.training.epochs * len(bundle.train_loader)
    warmup_steps = cfg.training.warmup_epochs * len(bundle.train_loader)
    scheduler = WarmupCosineScheduler(optimizer, warmup_steps, total_steps, cfg.training.learning_rate)

    history: list[dict] = []
    best_val_acc = -1.0
    best_epoch = -1
    checkpoint_path = ckpt_dir / "best.pt"

    for epoch in range(1, cfg.training.epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            loader=bundle.train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            amp=cfg.training.amp,
            grad_clip_norm=cfg.training.grad_clip_norm,
        )
        val_metrics, _, _, _ = evaluate(
            model=model,
            loader=bundle.val_loader,
            criterion=criterion,
            device=device,
            amp=cfg.training.amp,
            max_misclassified=cfg.training.max_misclassified,
        )
        row = {
            "epoch": epoch,
            "train_loss": train_metrics.loss,
            "train_acc": train_metrics.accuracy,
            "val_loss": val_metrics.loss,
            "val_acc": val_metrics.accuracy,
        }
        history.append(row)

        if val_metrics.accuracy > best_val_acc:
            best_val_acc = val_metrics.accuracy
            best_epoch = epoch
            torch.save({"model_state": model.state_dict(), "config": cfg.to_dict()}, checkpoint_path)

    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state["model_state"])
    final_metrics, y_true, y_pred, mistakes = evaluate(
        model=model,
        loader=bundle.val_loader,
        criterion=criterion,
        device=device,
        amp=cfg.training.amp,
        max_misclassified=cfg.training.max_misclassified,
    )

    analysis = build_error_analysis(y_true, y_pred, bundle.class_names)

    plot_history(history, plots_dir / "history.png")
    plot_confusion_matrix(analysis.confusion, bundle.class_names, plots_dir / "confusion_matrix.png")
    plot_per_class_accuracy(analysis.per_class_accuracy, plots_dir / "per_class_accuracy.png")
    plot_misclassified(mistakes, bundle.class_names, plots_dir / "misclassified.png")

    dump_json(run_dir / "config.json", cfg.to_dict())
    dump_csv(run_dir / "history.csv", history)

    summary = {
        "run_name": cfg.run_name,
        "dataset": cfg.data.dataset,
        "train_size": bundle.train_size,
        "val_size": bundle.val_size,
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "final_val_loss": final_metrics.loss,
        "patch_size": cfg.model.patch_size,
        "sequence_length": cfg.model.sequence_length,
        "token_dim": cfg.model.token_dim,
        "depth": cfg.model.depth,
        "num_heads": cfg.model.num_heads,
        "params_total": profile.params_total,
        "attn_scores_per_block": profile.attn_scores_per_block,
        "attn_memory_mb_fp32_per_block": profile.attn_memory_mb_fp32_per_block,
        "attn_flops_per_block": profile.attn_flops_per_block,
        "attn_flops_total": profile.attn_flops_total,
    }
    dump_json(run_dir / "summary.json", summary)
    dump_json(run_dir / "error_analysis.json", analysis.to_dict())

    artifacts = {
        "history_plot": "plots/history.png",
        "confusion_plot": "plots/confusion_matrix.png",
        "per_class_plot": "plots/per_class_accuracy.png",
        "misclassified_plot": "plots/misclassified.png",
        "checkpoint": "checkpoints/best.pt",
    }
    write_experiment_report(run_dir / "report.md", cfg, profile, best_epoch, best_val_acc, analysis, artifacts)

    return ExperimentResult(run_dir=run_dir, best_val_acc=best_val_acc, best_epoch=best_epoch, summary=summary)