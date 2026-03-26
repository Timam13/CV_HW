from __future__ import annotations

import argparse

from vit_system.config import DataConfig, ExperimentConfig, TrainingConfig, build_model_config
from vit_system.engine import run_experiment
from vit_system.utils import timestamp


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Train a minimal Vision Transformer")
    parser.add_argument("--preset", default="patch8_tiny_cls")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--dataset", default="imagefolder", choices=["imagefolder", "cifar10", "cifar100", "flowers102", "oxfordiiitpet"])
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--train-dir", default=None)
    parser.add_argument("--val-dir", default=None)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--val-ratio", type=float, default=0.2)

    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--patch-size", type=int, default=None)
    parser.add_argument("--embed-dim", type=int, default=None)
    parser.add_argument("--depth", type=int, default=None)
    parser.add_argument("--num-heads", type=int, default=None)
    parser.add_argument("--mlp-ratio", type=float, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--attention-dropout", type=float, default=None)
    parser.add_argument("--drop-path", type=float, default=None)
    parser.add_argument("--pooling", choices=["cls", "mean"], default=None)
    parser.add_argument("--no-cls-token", action="store_true")

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--max-misclassified", type=int, default=25)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    model_cfg = build_model_config(
        args.preset,
        image_size=args.image_size,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
        attention_dropout=args.attention_dropout,
        drop_path=args.drop_path,
        pooling=args.pooling,
        cls_token=False if args.no_cls_token else None,
    )
    if args.no_cls_token and args.pooling is None:
        model_cfg.pooling = "mean"

    data_cfg = DataConfig(
        dataset=args.dataset,
        data_root=args.data_root,
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        download=args.download,
        val_ratio=args.val_ratio,
        image_size=model_cfg.image_size,
        seed=args.seed,
    )
    train_cfg = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        label_smoothing=args.label_smoothing,
        num_workers=args.num_workers,
        grad_clip_norm=args.grad_clip_norm,
        amp=not args.no_amp,
        max_misclassified=args.max_misclassified,
    )
    run_name = args.run_name or f"{args.dataset}_{args.preset}_{timestamp()}"
    cfg = ExperimentConfig(
        run_name=run_name,
        output_dir=args.output_dir,
        device=args.device,
        seed=args.seed,
        model=model_cfg,
        training=train_cfg,
        data=data_cfg,
    )
    result = run_experiment(cfg)
    print(f"Done. Report: {result.run_dir / 'report.md'}")
    print(f"Best validation accuracy: {result.best_val_acc:.4f}")


if __name__ == "__main__":
    main()