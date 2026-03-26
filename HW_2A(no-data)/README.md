## Мини-система классификации изображений на основе Vision Transformer

Этот репозиторий содержит реализацию **минимального Vision Transformer (ViT)** для задачи классификации изображений и соответствует **варианту A** из домашнего задания: токенизация изображения на патчи, обработка последовательности через self-attention и классификация через линейную голову.  

В текущем архиве основной эксперимент выполнен на **CIFAR-10**, а кодовая база поддерживает запуск дополнительных архитектурных сравнений через готовые пресеты и sweep-сценарии.

---

## 1. Что сделано в проекте

В проекте реализованы все базовые элементы трансформерного пайплайна для компьютерного зрения:

- **Patch embedding**: изображение переводится из тензора `B × C × H × W` в последовательность токенов;
- **позиционные эмбеддинги**;
- **class token** и альтернативный режим **mean pooling**;
- **минимальный Vision Transformer** из нескольких блоков self-attention;
- **классификационная голова** для предсказания класса;
- **обучение, валидация, сохранение лучшей модели**;
- **автоматический подсчёт архитектурных характеристик**:
  - числа параметров,
  - длины последовательности,
  - оценки стоимости self-attention по памяти и вычислениям;
- **визуализации и error analysis**:
  - график обучения,
  - матрица ошибок,
  - per-class accuracy,
  - примеры ошибочных предсказаний.

---

## 2. Структура репозитория

```text
HW_2A/
├── README.md
├── requirements.txt
├── data/
│   └── cifar-10-batches-py/
├── outputs/
│   └── cifar10_baseline/
│       ├── checkpoints/
│       │   └── best.pt
│       ├── plots/
│       │   ├── confusion_matrix.png
│       │   ├── history.png
│       │   ├── misclassified.png
│       │   └── per_class_accuracy.png
│       ├── config.json
│       ├── error_analysis.json
│       ├── history.csv
│       ├── report.md
│       └── summary.json
└── src/
    ├── run_experiment.py
    ├── run_sweep.py
    └── vit_system/
        ├── __init__.py
        ├── analysis.py
        ├── config.py
        ├── data.py
        ├── engine.py
        ├── plotting.py
        ├── report.py
        ├── utils.py
        └── models/
            ├── __init__.py
            └── vit.py
```

---

## 3. Архитектура модели

### 3.1. Patch embedding

Patch embedding реализован через:

```python
nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
```

Это означает, что изображение разбивается на **неперекрывающиеся патчи** размера `P × P`, а затем каждый патч проецируется в вектор размерности `embed_dim`.

Если:
- размер изображения = `image_size`,
- размер патча = `patch_size`,

то:

- число патчей по одной стороне:  
  `patch_grid = image_size / patch_size`
- общее число патчей:  
  `num_patches = patch_grid^2`
- длина последовательности:
  - `num_patches + 1`, если используется `cls_token`,
  - `num_patches`, если используется `mean pooling`.

### 3.2. Минимальный Vision Transformer

Модель `VisionTransformer` состоит из следующих частей:

1. `PatchEmbedding`
2. `cls_token` (опционально)
3. `pos_embed`
4. стек `TransformerBlock`
5. финальная нормализация
6. классификационная голова `Linear(embed_dim, num_classes)`

Каждый `TransformerBlock` содержит:
- `LayerNorm`
- `MultiHeadSelfAttention`
- residual connection
- `MLP`
- residual connection

### 3.3. Self-attention

В `MultiHeadSelfAttention` вычисляются:
- `Q`, `K`, `V` через линейную проекцию,
- матрица attention-весов,
- взвешенная сумма значений `V`.

Формально внимание работает по схеме:

```text
Attention(Q, K, V) = softmax(QK^T / sqrt(d_head)) V
```

---

## 4. Датасет и препроцессинг

### Используемый датасет в сохранённом эксперименте
- **датасет**: `CIFAR-10`
- **число классов**: `10`
- **размер изображения**: `32 × 32`
- **обучающая выборка после split**: `40 000`
- **валидационная выборка**: `10 000`

### Аугментации обучения
Для train используются:
- `RandomResizedCrop`
- `RandAugment`
- `RandomHorizontalFlip`
- `Normalize`
- `RandomErasing`

### Преобразования для валидации
Для validation используются:
- `Resize`
- `CenterCrop`
- `Normalize`

Такой пайплайн делает обучение устойчивее и улучшает качество по сравнению с полностью “сырым” входом.

---

## 5. Основная конфигурация

Текущий сохранённый запуск: **`cifar10_baseline`**

| Параметр | Значение |
|---|---:|
| Dataset | CIFAR-10 |
| Image size | 32 |
| Patch size | 4 |
| Patch grid | 8 × 8 |
| Number of patches | 64 |
| Sequence length | 65 |
| Token dimension (`embed_dim`) | 256 |
| Depth | 6 |
| Heads | 4 |
| Pooling | `cls` |
| Class token | True |
| Epochs | 40 |
| Batch size | 128 |
| Learning rate | 5e-4 |
| Weight decay | 0.05 |
| Warmup epochs | 3 |
| Label smoothing | 0.1 |

---

## 6. Результаты baseline-эксперимента

По сохранённому запуску `cifar10_baseline`:

| Метрика | Значение |
|---|---:|
| Best epoch | 37 |
| Best validation accuracy | **0.7911** |
| Final validation loss | 0.9788 |

---

## 7. Error analysis

По данным `error_analysis.json` и `report.md` наиболее сложными классами оказались:

| Класс | Accuracy |
|---|---:|
| cat | 0.635 |
| dog | 0.692 |
| deer | 0.745 |
| bird | 0.746 |
| airplane | 0.800 |

Наиболее сильные пары путаницы:

| Истинный класс | Предсказан как | Count | Error rate |
|---|---|---:|---:|
| dog | cat | 162 | 0.158 |
| cat | dog | 151 | 0.156 |
| truck | automobile | 79 | 0.079 |
| deer | horse | 73 | 0.076 |
| bird | deer | 61 | 0.061 |

### Интерпретация ошибок
- **cat / dog** — классы визуально близки по текстуре, позе и масштабу объекта на маленьких изображениях `32 × 32`;
- **truck / automobile** — обе категории содержат схожие дорожные сцены и близкую форму объекта;
- **deer / horse** — путаница возникает из-за похожего силуэта и фона на части изображений;
- при небольшом размере изображения локальные детали ограничены, поэтому чисто токенная схема иногда теряет тонкие различия между близкими классами.

---

## 8. Автоматизация отчета и графиков

После запуска эксперимента автоматически сохраняются:

```text
outputs/<run_name>/
├── checkpoints/best.pt
├── plots/history.png
├── plots/confusion_matrix.png
├── plots/per_class_accuracy.png
├── plots/misclassified.png
├── history.csv
├── config.json
├── summary.json
├── error_analysis.json
└── report.md
```

Для текущего baseline-запуска уже присутствуют:

- `outputs/cifar10_baseline/plots/history.png`
- `outputs/cifar10_baseline/plots/confusion_matrix.png`
- `outputs/cifar10_baseline/plots/per_class_accuracy.png`
- `outputs/cifar10_baseline/plots/misclassified.png`
- `outputs/cifar10_baseline/report.md`

---

## 9. Поддерживаемые архитектурные сравнения

В `src/vit_system/config.py` уже подготовлены пресеты, позволяющие провести сравнения, требуемые заданием:

### CIFAR / универсальные пресеты
- `patch4_tiny_cls`
- `patch8_tiny_cls`
- `patch8_tiny_mean`
- `patch8_deep_cls`
- `patch8_wide_cls`
- `patch4_wide_cls`
- `cifar_patch4_strong_cls`
- `cifar_patch4_strong_mean`

### Preset sweeps
- `patch_size`
- `pooling`
- `depth`
- `embedding`
- `cifar_pooling`

### Что именно можно сравнивать
- **разный размер патча**;
- **разную глубину модели**;
- **class token vs mean pooling**;
- **разную размерность токена**.

---

## 10. Как запустить проект

### 10.1. Установка зависимостей

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 10.2. Запуск baseline-эксперимента на CIFAR-10

```bash
python src/run_experiment.py   --dataset cifar10   --data-root data   --download   --preset cifar_patch4_strong_cls   --epochs 40   --batch-size 128   --lr 5e-4   --weight-decay 0.05   --warmup-epochs 3   --label-smoothing 0.1   --run-name cifar10_baseline
```

### 10.3. Пример сравнения pooling-стратегий

```bash
python src/run_sweep.py   --sweep cifar_pooling   --dataset cifar10   --data-root data   --download   --epochs 40   --batch-size 128   --lr 5e-4   --run-prefix cifar_pooling_study
```

### 10.4. Пример сравнения размера патча

```bash
python src/run_sweep.py   --sweep patch_size   --dataset cifar10   --data-root data   --download   --epochs 20   --batch-size 128   --lr 3e-4   --run-prefix cifar_patchsize_study
```

---

## 11. Вывод

В работе построена **учебная, но полноценная мини-система классификации на основе Vision Transformer**, в которой изображение проходит весь путь:

**image → patches → token sequence → self-attention blocks → classification head**

Проект демонстрирует ключевые свойства ViT-подхода:

- простую и прозрачную токенную схему;
- явную зависимость стоимости внимания от длины последовательности;
- возможность гибко сравнивать архитектурные варианты;
- интерпретируемый анализ ошибок на уровне классов и конкретных предсказаний.

Для задачи классификации на небольшом изображении **CIFAR-10** модель достигла **79.11% validation accuracy**, а результаты показывают, что основными источниками ошибок остаются визуально близкие классы и ограничения маленького пространственного разрешения.