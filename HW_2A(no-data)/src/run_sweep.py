from __future__ import annotations

import argparse
from pathlib import Path

from vit_system.config import DataConfig, ExperimentConfig, SWEEPS, TrainingConfig, build_model_config
from vit_system.engine import run_experiment
from vit_system.plotting import plot_sweep_accuracy, plot_sweep_tradeoff
from vit_system.report import write_sweep_report
from vit_system.utils import dump_csv, ensure_dir, timestamp


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Run a preset sweep for Vision Transformer")
    parser.add_argument("--sweep", default="patch_size", choices=sorted(SWEEPS))
    parser.add_argument("--run-prefix", default=None)
    parser.add_argument("--output-dir", default="outputs/sweeps")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--dataset", default="imagefolder", choices=["imagefolder", "cifar10", "cifar100", "flowers102", "oxfordiiitpet"])
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--train-dir", default=None)
    parser.add_argument("--val-dir", default=None)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--val-ratio", type=float, default=0.2)

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--no-amp", action="store_true")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    sweep_name = args.run_prefix or f"{args.sweep}_{timestamp()}"
    sweep_dir = ensure_dir(Path(args.output_dir) / sweep_name)

    rows: list[dict] = []
    for preset in SWEEPS[args.sweep]:
        model_cfg = build_model_config(preset)
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
        training_cfg = TrainingConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            warmup_epochs=args.warmup_epochs,
            label_smoothing=args.label_smoothing,
            num_workers=args.num_workers,
            grad_clip_norm=args.grad_clip_norm,
            amp=not args.no_amp,
        )
        cfg = ExperimentConfig(
            run_name=preset,
            output_dir=str(sweep_dir),
            device=args.device,
            seed=args.seed,
            model=model_cfg,
            training=training_cfg,
            data=data_cfg,
        )
        result = run_experiment(cfg)
        rows.append(result.summary)

    summary_csv = sweep_dir / "comparison.csv"
    dump_csv(summary_csv, rows)
    accuracy_plot = sweep_dir / "accuracy.png"
    tradeoff_plot = sweep_dir / "tradeoff.png"
    plot_sweep_accuracy(rows, accuracy_plot)
    plot_sweep_tradeoff(rows, tradeoff_plot)
    write_sweep_report(
        sweep_dir / "report.md",
        args.sweep,
        rows,
        {
            "summary_csv": str(summary_csv.relative_to(sweep_dir)),
            "accuracy_plot": str(accuracy_plot.relative_to(sweep_dir)),
            "tradeoff_plot": str(tradeoff_plot.relative_to(sweep_dir)),
        },
    )
    print(f"Done. Sweep report: {sweep_dir / 'report.md'}")


if __name__ == "__main__":
    main()