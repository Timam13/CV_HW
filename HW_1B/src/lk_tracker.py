from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np
from tqdm import tqdm

from utils import (
    ensure_dir,
    open_video,
    get_video_info,
    iter_frames,
    bgr_to_gray,
    draw_tracks,
    frame_blur_score,
    frame_noise_score,
    illumination_change,
)


@dataclass
class LKMetrics:
    frame_idx: List[int]
    n_active: List[int]
    retention_rate: List[float]
    fb_error_mean: List[float]
    blur_score: List[float]
    noise_score: List[float]
    illum_change: List[float]
    finished_track_lengths: List[int]


def _detect_points(gray, max_points, quality, min_distance, block_size, mask=None):
    pts = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=max_points,
        qualityLevel=quality,
        minDistance=min_distance,
        blockSize=block_size,
        useHarrisDetector=False,
        mask=mask,
    )
    if pts is None:
        return np.empty((0, 1, 2), dtype=np.float32)
    return pts.astype(np.float32)

def _motion_mask_from_diff(
    prev_gray,
    gray,
    motion_thr=25.0,
    motion_percentile=95.0,   # используется, если motion_thr <= 0
    morph_ksize=7,
    open_iter=1,
    close_iter=2,
):
    diff = cv2.absdiff(prev_gray, gray)

    thr = float(motion_thr) if motion_thr > 0 else float(np.percentile(diff, motion_percentile))
    _, m = cv2.threshold(diff, thr, 255, cv2.THRESH_BINARY)

    k = max(3, morph_ksize | 1)  # odd
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    if open_iter > 0:
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel, iterations=open_iter)
    if close_iter > 0:
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=close_iter)

    return m


def run_lk(
    video_path: str,
    out_dir: str,
    max_frames: int = 150,
    lk_win: int = 21,
    lk_max_level: int = 3,
    lk_criteria_eps: float = 0.03,
    lk_criteria_count: int = 20,
    lk_max_points: int = 500,
    lk_quality: float = 0.01,
    lk_min_distance: int = 7,
    lk_block_size: int = 7,
    lk_redetect_every: int = 15,
    lk_min_active: int = 150,
    track_max_len: int = 60,
    fb_check: bool = True,
    fb_thresh: float = 1.0,
):
    ensure_dir(out_dir)
    cap = open_video(video_path)
    info = get_video_info(cap)

    lk_params = dict(
        winSize=(lk_win, lk_win),
        maxLevel=lk_max_level,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, lk_criteria_count, lk_criteria_eps),
    )

    out_video_path = os.path.join(out_dir, "lk_tracks.mp4")
    writer = cv2.VideoWriter(
        out_video_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        info.fps,
        (info.width, info.height),
        True,
    )
    if not writer.isOpened():
        raise RuntimeError("Failed to open VideoWriter for LK output.")

    ok, frame0 = cap.read()
    if not ok:
        raise RuntimeError("Empty video.")
    prev_gray = bgr_to_gray(frame0)

    ok, frame1 = cap.read()
    if not ok:
        raise RuntimeError("Video has only 1 frame.")
    gray1 = bgr_to_gray(frame1)

    motion_mask = _motion_mask_from_diff(prev_gray, gray1, motion_thr=25.0, morph_ksize=7)
    if int(motion_mask.sum()) == 0:
        motion_mask = None

    p0 = _detect_points(prev_gray, lk_max_points, lk_quality, lk_min_distance, lk_block_size, mask=motion_mask)
    if len(p0) == 0:
        p0 = _detect_points(prev_gray, lk_max_points, lk_quality, lk_min_distance, lk_block_size, mask=None)

    tracks = [[(float(p[0]), float(p[1]))] for p in p0.reshape(-1, 2)]

    # Сразу делаем LK шаг frame0 -> frame1, чтобы треки появились по движению
    if len(tracks) > 0:
        p_prev = np.array([[tr[-1]] for tr in tracks], dtype=np.float32)
        p_next, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray1, p_prev, None, **lk_params)
        st = st.reshape(-1).astype(bool)
        p_next = p_next.reshape(-1, 2)

        new_tracks = []
        for tr, good, pn in zip(tracks, st.tolist(), p_next.tolist()):
            if good:
                tr.append((float(pn[0]), float(pn[1])))
                new_tracks.append(tr)
        tracks = new_tracks

    # рисуем frame1
    vis1 = draw_tracks(frame1, tracks, max_len=track_max_len)
    writer.write(vis1)

    prev_gray = gray1

    m_frame_idx = [0]
    m_n_active = [len(tracks)]
    m_retention = [1.0]
    m_fb_err = [0.0]
    m_blur = [frame_blur_score(prev_gray)]
    m_noise = [frame_noise_score(prev_gray)]
    m_illum = [0.0]
    finished_lengths: List[int] = []

    for i, frame in tqdm(list(iter_frames(cap, max_frames=max_frames - 1)), desc="LK tracking", total=max_frames - 1):
        frame_idx = i + 1
        gray = bgr_to_gray(frame)

        if len(tracks) == 0:
            p_prev = np.empty((0, 1, 2), dtype=np.float32)
        else:
            p_prev = np.array([[tr[-1]] for tr in tracks], dtype=np.float32)

        if len(p_prev) > 0:
            p_next, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p_prev, None, **lk_params)
            st = st.reshape(-1).astype(bool)
            p_next = p_next.reshape(-1, 2)

            fb_err_vals = []
            if fb_check:
                p_back, st_back, _ = cv2.calcOpticalFlowPyrLK(gray, prev_gray, p_next.reshape(-1, 1, 2), None, **lk_params)
                st_back = st_back.reshape(-1).astype(bool)
                p_back = p_back.reshape(-1, 2)
                fb_dist = np.linalg.norm(p_back - p_prev.reshape(-1, 2), axis=1)
                valid_fb = st & st_back
                if valid_fb.any():
                    fb_err_vals = fb_dist[valid_fb].tolist()
                st = st & st_back & (fb_dist < fb_thresh)

            new_tracks: List[List[Tuple[float, float]]] = []
            for tr, good, pn in zip(tracks, st.tolist(), p_next.tolist()):
                if good:
                    tr.append((float(pn[0]), float(pn[1])))
                    if len(tr) > track_max_len:
                        tr[:] = tr[-track_max_len:]
                    new_tracks.append(tr)
                else:
                    finished_lengths.append(len(tr))
            tracks = new_tracks
            fb_mean = float(np.mean(fb_err_vals)) if len(fb_err_vals) > 0 else 0.0
        else:
            fb_mean = 0.0

        if (frame_idx % lk_redetect_every == 0) or (len(tracks) < lk_min_active):
            base_mask = _motion_mask_from_diff(prev_gray, gray, motion_thr=25.0, morph_ksize=7)
            if int(base_mask.sum()) == 0:
                base_mask = np.ones_like(gray, dtype=np.uint8) * 255

            mask = base_mask.copy()
            for tr in tracks:
                x, y = tr[-1]
                cv2.circle(mask, (int(x), int(y)), lk_min_distance, 0, -1)

            need = max(lk_max_points - len(tracks), 0)
            if need > 0:
                new_pts = cv2.goodFeaturesToTrack(
                    gray,
                    maxCorners=need,
                    qualityLevel=lk_quality,
                    minDistance=lk_min_distance,
                    blockSize=lk_block_size,
                    mask=mask,
                )
                if new_pts is not None:
                    for p in new_pts.reshape(-1, 2):
                        tracks.append([(float(p[0]), float(p[1]))])

        vis = draw_tracks(frame, tracks, max_len=track_max_len)
        cv2.putText(vis, f"LK | frame={frame_idx} | active={len(tracks)}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        writer.write(vis)

        prev_active = m_n_active[-1]
        retention = (len(tracks) / prev_active) if prev_active > 0 else 0.0
        m_frame_idx.append(frame_idx)
        m_n_active.append(len(tracks))
        m_retention.append(float(retention))
        m_fb_err.append(float(fb_mean))
        m_blur.append(frame_blur_score(gray))
        m_noise.append(frame_noise_score(gray))
        m_illum.append(illumination_change(prev_gray, gray))

        prev_gray = gray

    writer.release()
    cap.release()

    finished_lengths.extend([len(tr) for tr in tracks])

    metrics = LKMetrics(
        frame_idx=m_frame_idx,
        n_active=m_n_active,
        retention_rate=m_retention,
        fb_error_mean=m_fb_err,
        blur_score=m_blur,
        noise_score=m_noise,
        illum_change=m_illum,
        finished_track_lengths=finished_lengths,
    )
    return metrics, out_video_path, ""


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("LK sparse optical flow tracker (Lucas–Kanade)")
    p.add_argument("--video", required=True, help="Path to input video")
    p.add_argument("--out", default="outputs/lk", help="Output directory for LK artifacts")
    p.add_argument("--max-frames", type=int, default=150, help="Track length in frames (50–200 recommended)")
    p.add_argument("--lk-win", type=int, default=21)
    p.add_argument("--lk-max-level", type=int, default=3)
    p.add_argument("--lk-max-points", type=int, default=500)
    p.add_argument("--lk-quality", type=float, default=0.01)
    p.add_argument("--lk-min-distance", type=int, default=7)
    p.add_argument("--lk-block-size", type=int, default=7)
    p.add_argument("--lk-redetect-every", type=int, default=15)
    p.add_argument("--lk-min-active", type=int, default=150)
    p.add_argument("--track-max-len", type=int, default=60)
    p.add_argument("--no-fb-check", action="store_true")
    p.add_argument("--fb-thresh", type=float, default=1.0)
    return p


def main() -> None:
    args = build_argparser().parse_args()
    metrics, tracks_video, _ = run_lk(
        video_path=args.video,
        out_dir=args.out,
        max_frames=args.max_frames,
        lk_win=args.lk_win,
        lk_max_level=args.lk_max_level,
        lk_max_points=args.lk_max_points,
        lk_quality=args.lk_quality,
        lk_min_distance=args.lk_min_distance,
        lk_block_size=args.lk_block_size,
        lk_redetect_every=args.lk_redetect_every,
        lk_min_active=args.lk_min_active,
        track_max_len=args.track_max_len,
        fb_check=not args.no_fb_check,
        fb_thresh=args.fb_thresh,
    )
    print(f"Saved LK tracks video: {tracks_video}")
    print(f"Active points (start -> end): {metrics.n_active[0]} -> {metrics.n_active[-1]}")
    print(f"Mean retention rate: {float(np.mean(metrics.retention_rate[1:])):.3f}")
    if metrics.finished_track_lengths:
        print(f"Mean track length: {float(np.mean(metrics.finished_track_lengths)):.1f} frames")


if __name__ == "__main__":
    main()