from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterator, Tuple, Optional, List

import cv2
import numpy as np


@dataclass
class VideoInfo:
    width: int
    height: int
    fps: float
    frame_count: int


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def open_video(path: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    return cap


def get_video_info(cap: cv2.VideoCapture) -> VideoInfo:
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) if cap.get(cv2.CAP_PROP_FPS) > 0 else 25.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0 else -1
    return VideoInfo(width, height, fps, frame_count)


def make_writer(path: str, fps: float, size: Tuple[int, int]) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(path, fourcc, fps, size, True)


def iter_frames(cap: cv2.VideoCapture, max_frames: Optional[int] = None) -> Iterator[Tuple[int, np.ndarray]]:
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        yield i, frame
        i += 1
        if max_frames is not None and i >= max_frames:
            break


def bgr_to_gray(frame_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)


def hsv_flow_viz(flow: np.ndarray, clip_mag: Optional[float] = None) -> np.ndarray:
    '''
    Convert dense optical flow (H,W,2) to a BGR visualization using HSV:
    Hue = angle, Value = magnitude.
    '''
    fx, fy = flow[..., 0], flow[..., 1]
    mag, ang = cv2.cartToPolar(fx, fy, angleInDegrees=True)

    if clip_mag is not None and clip_mag > 0:
        mag = np.clip(mag, 0, clip_mag)

    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = (ang / 2).astype(np.uint8)      # OpenCV hue: [0,179]
    hsv[..., 1] = 255
    mag_norm = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    hsv[..., 2] = mag_norm.astype(np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr


def draw_tracks(
    frame: np.ndarray,
    tracks: List[List[Tuple[float, float]]],
    max_len: int = 50,
    point_radius: int = 2,
) -> np.ndarray:
    out = frame.copy()
    for tr in tracks:
        if len(tr) < 2:
            continue
        pts = tr[-max_len:]
        for j in range(1, len(pts)):
            x1, y1 = pts[j - 1]
            x2, y2 = pts[j]
            cv2.line(out, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2, cv2.LINE_AA)
        x, y = pts[-1]
        cv2.circle(out, (int(x), int(y)), point_radius, (0, 0, 255), -1, cv2.LINE_AA)
    return out


def overlay_mask(frame: np.ndarray, mask01: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    if mask01.dtype != np.uint8:
        mask01 = mask01.astype(np.uint8)
    m = mask01.copy()
    if m.max() == 1:
        m = m * 255
    color = np.zeros_like(frame)
    color[..., 2] = m  # red channel
    out = cv2.addWeighted(frame, 1.0, color, alpha, 0)
    return out


def connected_components(mask01: np.ndarray, min_area: int = 150):
    if mask01.max() == 1:
        mask_bin = (mask01 * 255).astype(np.uint8)
    else:
        mask_bin = mask01.astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask_bin, connectivity=8)
    comps = []
    for k in range(1, num):
        x, y, w, h, area = stats[k].tolist()
        if area >= min_area:
            comps.append((x, y, w, h, area))
    comps.sort(key=lambda t: t[-1], reverse=True)
    return comps


def draw_bboxes(frame: np.ndarray, comps) -> np.ndarray:
    out = frame.copy()
    for (x, y, w, h, area) in comps:
        cv2.rectangle(out, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(out, f"area={area}", (x, max(0, y - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
    return out


def frame_blur_score(gray: np.ndarray) -> float:
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(lap.var())


def frame_noise_score(gray: np.ndarray) -> float:
    blur = cv2.GaussianBlur(gray, (0, 0), 1.5)
    resid = (gray.astype(np.float32) - blur.astype(np.float32))
    return float(resid.std())


def illumination_change(prev_gray: np.ndarray, gray: np.ndarray) -> float:
    return float(np.mean(np.abs(gray.astype(np.float32) - prev_gray.astype(np.float32))))