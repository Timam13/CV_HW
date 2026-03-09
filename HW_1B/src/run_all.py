from __future__ import annotations

import argparse
import os

import numpy as np
import cv2

from lk_tracker import run_lk
from farneback_flow import run_farneback
from metrics import plot_lk, plot_farneback, plot_trajectories_frame_coords, write_report
from utils import ensure_dir, open_video, get_video_info, bgr_to_gray


def _serialize_tracks_from_lk(video_path: str, max_frames: int, lk_out_dir: str):
    ensure_dir(lk_out_dir)
    cap = open_video(video_path)
    info = get_video_info(cap)

    ok, first = cap.read()
    if not ok:
        raise RuntimeError("Empty video.")
    prev_gray = bgr_to_gray(first)

    p0 = cv2.goodFeaturesToTrack(prev_gray, maxCorners=500, qualityLevel=0.01, minDistance=7, blockSize=7)
    if p0 is None:
        tracks = np.full((0, max_frames, 2), np.nan, dtype=np.float32)
        path = os.path.join(lk_out_dir, "tracks.npy")
        np.save(path, tracks)
        return path, info.width, info.height

    p0 = p0.reshape(-1, 2).astype(np.float32)
    K = p0.shape[0]
    tracks = np.full((K, max_frames, 2), np.nan, dtype=np.float32)
    tracks[:, 0, :] = p0

    lk_params = dict(
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
    )

    p_prev = p0.reshape(-1, 1, 2)

    for t in range(1, max_frames):
        ok, frame = cap.read()
        if not ok:
            break
        gray = bgr_to_gray(frame)

        p_next, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p_prev, None, **lk_params)
        st = st.reshape(-1).astype(bool)
        p_next = p_next.reshape(-1, 2)

        p_back, st2, _ = cv2.calcOpticalFlowPyrLK(gray, prev_gray, p_next.reshape(-1, 1, 2), None, **lk_params)
        st2 = st2.reshape(-1).astype(bool)
        fb = np.linalg.norm(p_back.reshape(-1, 2) - p_prev.reshape(-1, 2), axis=1)
        good = st & st2 & (fb < 1.0)

        tracks[good, t, :] = p_next[good]
        p_prev = np.where(good[:, None, None], p_next.reshape(-1, 1, 2), p_prev)
        prev_gray = gray

    cap.release()
    path = os.path.join(lk_out_dir, "tracks.npy")
    np.save(path, tracks)
    return path, info.width, info.height


def run_all(video_path: str, out_dir: str, max_frames: int) -> str:
    out_dir = os.path.abspath(out_dir)
    lk_dir = os.path.join(out_dir, "lk")
    fb_dir = os.path.join(out_dir, "farneback")
    plots_dir = os.path.join(out_dir, "plots")
    ensure_dir(lk_dir)
    ensure_dir(fb_dir)
    ensure_dir(plots_dir)

    lk_metrics, lk_tracks_video, _ = run_lk(video_path=video_path, out_dir=lk_dir, max_frames=max_frames)
    fb_metrics, fb_hsv, fb_mask, fb_boxes = run_farneback(video_path=video_path, out_dir=fb_dir, max_frames=max_frames)

    p_lk_ret, p_lk_fb = plot_lk(lk_metrics, plots_dir)
    p_fb = plot_farneback(fb_metrics, plots_dir)

    tracks_npy, w, h = _serialize_tracks_from_lk(video_path, max_frames, lk_dir)
    tracks = np.load(tracks_npy)
    traj_plot = os.path.join(lk_dir, "trajectories_plot.png")
    plot_trajectories_frame_coords(tracks, w, h, traj_plot)

    report_path = os.path.join(out_dir, "report.md")
    paths = dict(
        lk_tracks_video=os.path.relpath(lk_tracks_video, out_dir),
        fb_hsv_video=os.path.relpath(fb_hsv, out_dir),
        fb_mask_video=os.path.relpath(fb_mask, out_dir),
        fb_boxes_video=os.path.relpath(fb_boxes, out_dir),
        plot_lk_retention=os.path.relpath(p_lk_ret, out_dir),
        plot_lk_fb=os.path.relpath(p_lk_fb, out_dir),
        plot_fb_mask_stats=os.path.relpath(p_fb, out_dir),
        traj_plot=os.path.relpath(traj_plot, out_dir),
    )
    write_report(report_path, video_path, max_frames, lk_metrics, fb_metrics, paths)
    return report_path


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Run full pipeline: LK + Farnebäck + plots + report")
    p.add_argument("--video", required=True)
    p.add_argument("--out", default="outputs")
    p.add_argument("--max-frames", type=int, default=150)
    p.add_argument("--start-frame", type=int, default=0)
    return p


def main() -> None:
    args = build_argparser().parse_args()
    report_path = run_all(args.video, args.out, args.max_frames)
    print(f"Done. Report: {report_path}")


if __name__ == "__main__":
    main()