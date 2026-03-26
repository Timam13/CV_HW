from __future__ import annotations

from pathlib import Path

from .analysis import ErrorAnalysis, ModelProfile
from .config import ExperimentConfig
from .utils import ensure_dir, format_large_int


def write_experiment_report(
    path: str | Path,
    cfg: ExperimentConfig,
    profile: ModelProfile,
    best_epoch: int,
    best_val_acc: float,
    analysis: ErrorAnalysis,
    artifacts: dict[str, str],
) -> None:
    path = Path(path)
    ensure_dir(path.parent)

    hardest_classes_md = "\n".join(
        f"- **{row['class']}** — accuracy={row['accuracy']:.3f} ({row['correct']}/{row['total']})"
        for row in analysis.hardest_classes
    ) or "- n/a"

    hardest_pairs_md = "\n".join(
        f"- **{row['true_class']} → {row['predicted_as']}** — count={row['count']}, error_rate={row['error_rate']:.3f}"
        for row in analysis.hardest_pairs
    ) or "- n/a"

    text = f"""# Vision Transformer experiment report

## 1. Run
- run_name: `{cfg.run_name}`
- dataset: `{cfg.data.dataset}`
- best_epoch: `{best_epoch}`
- best_val_acc: `{best_val_acc:.4f}`

## 2. Architecture
- image_size: `{cfg.model.image_size}`
- patch_size: `{cfg.model.patch_size}`
- patch_grid: `{cfg.model.patch_grid} x {cfg.model.patch_grid}`
- num_patches: `{cfg.model.num_patches}`
- sequence_length: `{cfg.model.sequence_length}`
- token_dim: `{cfg.model.token_dim}`
- depth: `{cfg.model.depth}`
- heads: `{cfg.model.num_heads}`
- cls_token: `{cfg.model.cls_token}`
- pooling: `{cfg.model.pooling}`

## 3. Capacity and attention cost
- trainable parameters: `{profile.params_total}` ({format_large_int(profile.params_total)})
- attention scores per block: `{profile.attn_scores_per_block}`
- attention memory per block (fp32): `{profile.attn_memory_mb_fp32_per_block:.3f} MB`
- attention FLOPs proxy per block: `{format_large_int(profile.attn_flops_per_block)}`
- attention FLOPs proxy total: `{format_large_int(profile.attn_flops_total)}`

## 4. Error analysis
### Hardest classes
{hardest_classes_md}

### Strongest confusion pairs
{hardest_pairs_md}

## 5. Artifacts
- history: `{artifacts.get('history_plot', 'n/a')}`
- confusion matrix: `{artifacts.get('confusion_plot', 'n/a')}`
- per-class accuracy: `{artifacts.get('per_class_plot', 'n/a')}`
- misclassified gallery: `{artifacts.get('misclassified_plot', 'n/a')}`
- best checkpoint: `{artifacts.get('checkpoint', 'n/a')}`

## 6. Notes
- Patch embedding is explicit: image -> non-overlapping patches -> token sequence.
- Sequence length grows quadratically in attention cost, so smaller patches increase cost fast.
- The report uses validation predictions for confusion analysis.
"""
    path.write_text(text, encoding="utf-8")


def write_sweep_report(path: str | Path, sweep_name: str, rows: list[dict], artifacts: dict[str, str]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    rows_md = "\n".join(
        f"| {row['run_name']} | {row['patch_size']} | {row['sequence_length']} | {format_large_int(int(row['params_total']))} | {row['best_val_acc']:.4f} |"
        for row in rows
    )

    text = f"""# Vision Transformer sweep report

## Sweep
- name: `{sweep_name}`
- runs: `{len(rows)}`

## Results
| run | patch_size | seq_len | params | best_val_acc |
| --- | ---: | ---: | ---: | ---: |
{rows_md}

## Artifacts
- accuracy bars: `{artifacts.get('accuracy_plot', 'n/a')}`
- accuracy/params trade-off: `{artifacts.get('tradeoff_plot', 'n/a')}`
- raw summary: `{artifacts.get('summary_csv', 'n/a')}`

## Reading guide
- Compare sequence length against validation accuracy to understand the patch-size trade-off.
- Compare parameter count against accuracy to see where extra capacity stops paying off.
"""
    path.write_text(text, encoding="utf-8")