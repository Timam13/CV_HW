# Vision Transformer experiment report

## 1. Run
- run_name: `cifar10_baseline`
- dataset: `cifar10`
- best_epoch: `37`
- best_val_acc: `0.7911`

## 2. Architecture
- image_size: `32`
- patch_size: `4`
- patch_grid: `8 x 8`
- num_patches: `64`
- sequence_length: `65`
- token_dim: `256`
- depth: `6`
- heads: `4`
- cls_token: `True`
- pooling: `cls`

## 3. Capacity and attention cost
- trainable parameters: `4771082` (4.77M)
- attention scores per block: `16900`
- attention memory per block (fp32): `0.064 MB`
- attention FLOPs proxy per block: `19.20M`
- attention FLOPs proxy total: `115.22M`

## 4. Error analysis
### Hardest classes
- **cat** — accuracy=0.635 (614/967)
- **dog** — accuracy=0.692 (709/1024)
- **deer** — accuracy=0.745 (717/963)
- **bird** — accuracy=0.746 (750/1006)
- **airplane** — accuracy=0.800 (778/973)
- **horse** — accuracy=0.830 (859/1035)
- **frog** — accuracy=0.833 (854/1025)
- **truck** — accuracy=0.842 (841/999)
- **ship** — accuracy=0.886 (884/998)
- **automobile** — accuracy=0.896 (905/1010)

### Strongest confusion pairs
- **dog → cat** — count=162, error_rate=0.158
- **cat → dog** — count=151, error_rate=0.156
- **truck → automobile** — count=79, error_rate=0.079
- **deer → horse** — count=73, error_rate=0.076
- **bird → deer** — count=61, error_rate=0.061
- **automobile → truck** — count=51, error_rate=0.050
- **ship → airplane** — count=50, error_rate=0.050
- **frog → cat** — count=49, error_rate=0.048
- **horse → deer** — count=49, error_rate=0.047
- **airplane → bird** — count=40, error_rate=0.041

## 5. Artifacts
- history: `plots/history.png`
- confusion matrix: `plots/confusion_matrix.png`
- per-class accuracy: `plots/per_class_accuracy.png`
- misclassified gallery: `plots/misclassified.png`
- best checkpoint: `checkpoints/best.pt`

## 6. Notes
- Patch embedding is explicit: image -> non-overlapping patches -> token sequence.
- Sequence length grows quadratically in attention cost, so smaller patches increase cost fast.
- The report uses validation predictions for confusion analysis.
