# Отчёт: LK vs Farnebäck (движение / трекинг)

**Видео:** `data/videos/person-bicycle-car-detection.mp4`  
**Проанализировано кадров:** `150`

## 1) Реализация

- Sparse LK (Lucas–Kanade): `cv2.goodFeaturesToTrack` + `cv2.calcOpticalFlowPyrLK` + (опционально) forward-backward check.
- Dense Farnebäck: `cv2.calcOpticalFlowFarneback` + magnitude threshold + морфология + connected components → bbox.

## 2) Основные артефакты

- LK tracks video: `lk/lk_tracks.mp4`
- Farnebäck HSV flow: `farneback/flow_hsv.mp4`
- Farnebäck motion mask: `farneback/motion_mask.mp4`
- Farnebäck motion boxes: `farneback/motion_boxes.mp4`

## 3) Количественные метрики

### LK (Sparse)
- стартовых точек: **174**
- активных точек в конце: **457**
- mean retention: **1.010**, p10 retention: **0.990**
- mean FB-error: **0.034 px**, p95 FB-error: **0.059 px**
- mean длина трека: **48.1 кадров**, p10 длины: **15.0**

Графики:
- retention: `plots/lk_retention.png`
- forward-backward error: `plots/lk_fb_error.png`
- траектории в координатах кадра: `lk/trajectories_plot.png`

### Farnebäck (Dense)
- mean motion ratio: **0.0612**, p95: **0.1448**
- mean #components: **1.20**, p95: **3.00**
- mean largest-comp ratio: **0.42**, p10: **0.00**
- mean |∇mag|: **0.1291** (прокси шума)

График:
- mask stats: `plots/mask_stats.png`

## 4) Качественный разбор ошибок (где и почему)

- **Шум/ложные движения в Farnebäck**: маска не собирается в крупные компоненты (низкая доля крупнейшей компоненты).

Типовые наблюдения (ориентир, что искать глазами в видео):
- **Где LK теряет точки:** однородные области без углов/текстуры; быстрые движения; motion blur; окклюзии; сильные изменения освещения.
- **Где Farnebäck шумит:** мелкий сенсорный шум → “дрожание” поля; тени/блики дают ложный поток; компрессия → зернистая mag-карта.
- **Где маска фрагментируется:** слишком низкий порог `mag_thr` или слабая морфология; много мелких движений (листва/вода/дождь).

## 5) Как студент выбирает метод в реальной задаче

- **Выбираю LK**, если нужно *отслеживать несколько объектов* по ключевым точкам, важна скорость и интерпретируемые траектории, а сцена содержит текстуру (углы/границы).
- **Выбираю Farnebäck**, если нужна *карта движения для всех пикселей* (маска/сегментация движения), есть большие/плавные смещения, либо мало надёжных углов для LK.
- При сильных тенях/шуме/blur: LK чаще теряет точки, Farnebäck чаще “шумит” в маске — компенсируется морфологией, сглаживанием и порогами.

## 6) Параметры и рекомендации

- LK: увеличьте `lk_win` и `lk_max_level` при более быстром движении, но следите за ростом ошибки.
- Farnebäck: если много шума/фрагментации — повышайте `mag_thr` (или percentile), увеличивайте `morph_ksize`, поднимайте `min_area`.
