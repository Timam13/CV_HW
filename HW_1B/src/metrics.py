from __future__ import annotations

import os
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt

from lk_tracker import LKMetrics
from farneback_flow import FBMetrics
from utils import ensure_dir


def _save_plot(path: str) -> None:
    ensure_dir(os.path.dirname(path))
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_lk(metrics: LKMetrics, out_dir: str) -> Tuple[str, str]:
    ensure_dir(out_dir)

    x = np.array(metrics.frame_idx)
    retention = np.array(metrics.retention_rate)
    fb = np.array(metrics.fb_error_mean)
    blur = np.array(metrics.blur_score)
    illum = np.array(metrics.illum_change)

    p1 = os.path.join(out_dir, "lk_retention.png")
    plt.figure()
    plt.plot(x, retention)
    plt.xlabel("Frame")
    plt.ylabel("Retention rate (active_t / active_{t-1})")
    plt.title("LK: point retention over time")
    _save_plot(p1)

    p2 = os.path.join(out_dir, "lk_fb_error.png")
    plt.figure()
    plt.plot(x, fb)
    plt.xlabel("Frame")
    plt.ylabel("Forward-backward error (px)")
    plt.title("LK: forward-backward tracking error")
    _save_plot(p2)

    p3 = os.path.join(out_dir, "lk_vs_blur_illum.png")
    plt.figure()
    plt.plot(x, blur, label="blur score (var Laplacian)")
    plt.plot(x, illum, label="illumination change")
    plt.xlabel("Frame")
    plt.title("Scene factors: blur / illumination change")
    plt.legend()
    _save_plot(p3)

    return p1, p2


def plot_farneback(metrics: FBMetrics, out_dir: str) -> str:
    ensure_dir(out_dir)

    x = np.array(metrics.frame_idx)
    motion = np.array(metrics.motion_ratio)
    comps = np.array(metrics.n_components)
    largest = np.array(metrics.largest_comp_ratio)
    gradm = np.array(metrics.grad_mag_mean)

    p = os.path.join(out_dir, "mask_stats.png")
    plt.figure()
    plt.plot(x, motion, label="motion ratio")
    plt.plot(x, comps, label="#components")
    plt.plot(x, largest, label="largest comp ratio")
    plt.plot(x, gradm, label="mean |∇mag| (noise proxy)")
    plt.xlabel("Frame")
    plt.title("Farnebäck: mask & noise/fragmentation stats")
    plt.legend()
    _save_plot(p)
    return p


def plot_trajectories_frame_coords(tracks_np: np.ndarray, frame_w: int, frame_h: int, out_path: str) -> str:
    plt.figure(figsize=(7, 5))
    for k in range(tracks_np.shape[0]):
        tr = tracks_np[k]
        ok = ~np.isnan(tr[:, 0])
        if ok.sum() < 2:
            continue
        xs = tr[ok, 0]
        ys = tr[ok, 1]
        plt.plot(xs, ys, linewidth=1)

    plt.gca().invert_yaxis()
    plt.xlim([0, frame_w])
    plt.ylim([frame_h, 0])
    plt.xlabel("x (px)")
    plt.ylabel("y (px)")
    plt.title("LK trajectories in frame coordinates")
    ensure_dir(os.path.dirname(out_path))
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
    return out_path


def _summarize_lk(m: LKMetrics) -> dict:
    rr = np.array(m.retention_rate[1:]) if len(m.retention_rate) > 1 else np.array([0.0])
    fb = np.array(m.fb_error_mean[1:]) if len(m.fb_error_mean) > 1 else np.array([0.0])
    blur = np.array(m.blur_score)
    illum = np.array(m.illum_change)

    out = dict(
        lk_start_points=int(m.n_active[0]) if m.n_active else 0,
        lk_end_points=int(m.n_active[-1]) if m.n_active else 0,
        lk_mean_retention=float(rr.mean()) if rr.size else 0.0,
        lk_p10_retention=float(np.percentile(rr, 10)) if rr.size else 0.0,
        lk_mean_fb_error=float(fb.mean()) if fb.size else 0.0,
        lk_p95_fb_error=float(np.percentile(fb, 95)) if fb.size else 0.0,
        lk_mean_track_len=float(np.mean(m.finished_track_lengths)) if m.finished_track_lengths else 0.0,
        lk_p10_track_len=float(np.percentile(m.finished_track_lengths, 10)) if m.finished_track_lengths else 0.0,
        scene_blur_min=float(np.min(blur)) if blur.size else 0.0,
        scene_blur_median=float(np.median(blur)) if blur.size else 0.0,
        scene_illum_p95=float(np.percentile(illum, 95)) if illum.size else 0.0,
    )
    return out


def _summarize_fb(m: FBMetrics) -> dict:
    motion = np.array(m.motion_ratio[1:]) if len(m.motion_ratio) > 1 else np.array([0.0])
    comps = np.array(m.n_components[1:]) if len(m.n_components) > 1 else np.array([0.0])
    largest = np.array(m.largest_comp_ratio[1:]) if len(m.largest_comp_ratio) > 1 else np.array([0.0])
    gradm = np.array(m.grad_mag_mean[1:]) if len(m.grad_mag_mean) > 1 else np.array([0.0])
    illum = np.array(m.illum_change)

    out = dict(
        fb_mean_motion_ratio=float(motion.mean()) if motion.size else 0.0,
        fb_p95_motion_ratio=float(np.percentile(motion, 95)) if motion.size else 0.0,
        fb_mean_components=float(comps.mean()) if comps.size else 0.0,
        fb_p95_components=float(np.percentile(comps, 95)) if comps.size else 0.0,
        fb_mean_largest_ratio=float(largest.mean()) if largest.size else 0.0,
        fb_p10_largest_ratio=float(np.percentile(largest, 10)) if largest.size else 0.0,
        fb_mean_grad_mag=float(gradm.mean()) if gradm.size else 0.0,
        scene_illum_p95=float(np.percentile(illum, 95)) if illum.size else 0.0,
    )
    return out


def write_report(
    report_path: str,
    video_path: str,
    max_frames: int,
    lk_metrics: LKMetrics,
    fb_metrics: FBMetrics,
    paths: dict,
) -> str:
    ensure_dir(os.path.dirname(report_path))

    s_lk = _summarize_lk(lk_metrics)
    s_fb = _summarize_fb(fb_metrics)

    notes = []
    if s_lk["lk_p10_track_len"] < 10:
        notes.append("- **LK часто теряет точки**: много коротких треков (p10 длины < 10 кадров). Причины: blur / мало текстуры / окклюзии.")
    if s_lk["lk_p95_fb_error"] > 2.0:
        notes.append("- **LK нестабилен на быстрых/сложных движениях**: высокий p95 forward-backward error (> 2 px).")
    if s_fb["fb_p95_components"] > 8:
        notes.append("- **Маска Farnebäck фрагментируется**: много компонент (p95 #components > 8). Обычно помогает увеличить `morph_ksize` или `min_area`.")
    if s_fb["fb_p10_largest_ratio"] < 0.5 and s_fb["fb_mean_motion_ratio"] > 0.02:
        notes.append("- **Шум/ложные движения в Farnebäck**: маска не собирается в крупные компоненты (низкая доля крупнейшей компоненты).")
    if max(s_lk["scene_illum_p95"], s_fb["scene_illum_p95"]) > 15:
        notes.append("- **Заметные изменения освещения/тени**: высокий p95 |ΔI|. Dense flow часто реагирует на тени как на движение.")

    choose = [
        "- **Выбираю LK**, если нужно *отслеживать несколько объектов* по ключевым точкам, важна скорость и интерпретируемые траектории, а сцена содержит текстуру (углы/границы).",
        "- **Выбираю Farnebäck**, если нужна *карта движения для всех пикселей* (маска/сегментация движения), есть большие/плавные смещения, либо мало надёжных углов для LK.",
        "- При сильных тенях/шуме/blur: LK чаще теряет точки, Farnebäck чаще “шумит” в маске — компенсируется морфологией, сглаживанием и порогами."
    ]

    md = f"""# Отчёт: LK vs Farnebäck (движение / трекинг)

**Видео:** `{video_path}`  
**Проанализировано кадров:** `{max_frames}`

## 1) Реализация

- Sparse LK (Lucas–Kanade): `cv2.goodFeaturesToTrack` + `cv2.calcOpticalFlowPyrLK` + (опционально) forward-backward check.
- Dense Farnebäck: `cv2.calcOpticalFlowFarneback` + magnitude threshold + морфология + connected components → bbox.

## 2) Основные артефакты

- LK tracks video: `{paths.get("lk_tracks_video","")}`
- Farnebäck HSV flow: `{paths.get("fb_hsv_video","")}`
- Farnebäck motion mask: `{paths.get("fb_mask_video","")}`
- Farnebäck motion boxes: `{paths.get("fb_boxes_video","")}`

## 3) Количественные метрики

### LK (Sparse)
- стартовых точек: **{s_lk["lk_start_points"]}**
- активных точек в конце: **{s_lk["lk_end_points"]}**
- mean retention: **{s_lk["lk_mean_retention"]:.3f}**, p10 retention: **{s_lk["lk_p10_retention"]:.3f}**
- mean FB-error: **{s_lk["lk_mean_fb_error"]:.3f} px**, p95 FB-error: **{s_lk["lk_p95_fb_error"]:.3f} px**
- mean длина трека: **{s_lk["lk_mean_track_len"]:.1f} кадров**, p10 длины: **{s_lk["lk_p10_track_len"]:.1f}**

Графики:
- retention: `{paths.get("plot_lk_retention","")}`
- forward-backward error: `{paths.get("plot_lk_fb","")}`
- траектории в координатах кадра: `{paths.get("traj_plot","")}`

### Farnebäck (Dense)
- mean motion ratio: **{s_fb["fb_mean_motion_ratio"]:.4f}**, p95: **{s_fb["fb_p95_motion_ratio"]:.4f}**
- mean #components: **{s_fb["fb_mean_components"]:.2f}**, p95: **{s_fb["fb_p95_components"]:.2f}**
- mean largest-comp ratio: **{s_fb["fb_mean_largest_ratio"]:.2f}**, p10: **{s_fb["fb_p10_largest_ratio"]:.2f}**
- mean |∇mag|: **{s_fb["fb_mean_grad_mag"]:.4f}** (прокси шума)

График:
- mask stats: `{paths.get("plot_fb_mask_stats","")}`

## 4) Качественный разбор ошибок (где и почему)

{os.linesep.join(notes) if notes else "- По выбранному видео существенных провалов метрики не показали; ошибки смотрите по визуализациям."}

Типовые наблюдения (ориентир, что искать глазами в видео):
- **Где LK теряет точки:** однородные области без углов/текстуры; быстрые движения; motion blur; окклюзии; сильные изменения освещения.
- **Где Farnebäck шумит:** мелкий сенсорный шум → “дрожание” поля; тени/блики дают ложный поток; компрессия → зернистая mag-карта.
- **Где маска фрагментируется:** слишком низкий порог `mag_thr` или слабая морфология; много мелких движений (листва/вода/дождь).

## 5) Как студент выбирает метод в реальной задаче

{os.linesep.join(choose)}

## 6) Параметры и рекомендации

- LK: увеличьте `lk_win` и `lk_max_level` при более быстром движении, но следите за ростом ошибки.
- Farnebäck: если много шума/фрагментации — повышайте `mag_thr` (или percentile), увеличивайте `morph_ksize`, поднимайте `min_area`.
"""
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(md)
    return report_path