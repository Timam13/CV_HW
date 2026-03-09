from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np
from tqdm import tqdm

from utils import (
    ensure_dir,
    open_video,
    get_video_info,
    iter_frames,
    bgr_to_gray,
    hsv_flow_viz,
    overlay_mask,
    connected_components,
    draw_bboxes,
    frame_blur_score,
    frame_noise_score,
    illumination_change,
)


@dataclass
class FBMetrics:
    frame_idx: List[int]
    mag_mean: List[float]
    mag_p95: List[float]
    motion_ratio: List[float]
    n_components: List[int]
    largest_comp_ratio: List[float]
    grad_mag_mean: List[float]
    blur_score: List[float]
    noise_score: List[float]
    illum_change: List[float]


def run_farneback(
    video_path: str,
    out_dir: str,
    max_frames: int = 150,
    start_frame: int = 0,
    pyr_scale: float = 0.5,
    levels: int = 3,
    winsize: int = 15,
    iterations: int = 3,
    poly_n: int = 5,
    poly_sigma: float = 1.2,
    flags: int = 0,
    mag_thr: float = 0.0,
    mag_percentile: float = 95.0,
    morph_ksize: int = 5,
    min_area: int = 2500,
    clip_mag: Optional[float] = None,
):
    start_frame: int = 0
    ensure_dir(out_dir)
    cap = open_video(video_path)
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    info = get_video_info(cap)

    out_hsv_path = os.path.join(out_dir, "flow_hsv.mp4")
    out_mask_path = os.path.join(out_dir, "motion_mask.mp4")
    out_boxes_path = os.path.join(out_dir, "motion_boxes.mp4")

    hsv_writer = cv2.VideoWriter(out_hsv_path, cv2.VideoWriter_fourcc(*"mp4v"), info.fps, (info.width, info.height), True)
    mask_writer = cv2.VideoWriter(out_mask_path, cv2.VideoWriter_fourcc(*"mp4v"), info.fps, (info.width, info.height), True)
    box_writer = cv2.VideoWriter(out_boxes_path, cv2.VideoWriter_fourcc(*"mp4v"), info.fps, (info.width, info.height), True)
    if not (hsv_writer.isOpened() and mask_writer.isOpened() and box_writer.isOpened()):
        raise RuntimeError("Failed to open VideoWriter(s) for Farnebäck outputs.")

    ok, first = cap.read()
    if not ok:
        raise RuntimeError("Empty video.")
    prev_gray = bgr_to_gray(first)

    f_idx = [0]
    mag_mean = [0.0]
    mag_p95 = [0.0]
    motion_ratio = [0.0]
    n_comp = [0]
    largest_ratio = [0.0]
    grad_mag_mean = [0.0]
    blur = [frame_blur_score(prev_gray)]
    noise = [frame_noise_score(prev_gray)]
    illum = [0.0]

    k = max(3, morph_ksize | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    thr_prev = None
    
    for i, frame in tqdm(list(iter_frames(cap, max_frames=max_frames - 1)), desc="Farnebäck flow", total=max_frames - 1):
        frame_idx = i + 1
        gray = bgr_to_gray(frame)

        flow = cv2.calcOpticalFlowFarneback(
            prev_gray,
            gray,
            None,
            pyr_scale,
            levels,
            winsize,
            iterations,
            poly_n,
            poly_sigma,
            flags,
        )

        fx, fy = flow[..., 0], flow[..., 1]
        mag, _ = cv2.cartToPolar(fx, fy, angleInDegrees=False)

        thr = mag_thr
        if thr <= 0:
            thr_frame = float(np.percentile(mag, mag_percentile))
            if thr_prev is None:
                thr = thr_frame
            else:
                alpha = 0.90
                thr = alpha * thr_prev + (1 - alpha) * thr_frame
            thr_prev = thr

        raw_mask = (mag >= thr).astype(np.uint8)

        mask = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        
        h, w = mask.shape
        pad_x = int(0.02 * w)
        pad_y = int(0.02 * h)
        mask[:pad_y, :] = 0
        mask[h - pad_y:, :] = 0
        mask[:, :pad_x] = 0
        mask[:, w - pad_x:] = 0

        min_area_eff = max(min_area, int(0.002 * mask.size))  # 0.2% кадра
        comps = connected_components(mask, min_area=min_area_eff)
        mask_area = int(mask.sum())
        if mask_area > 0 and len(comps) > 0:
            largest = comps[0][-1]
            largest_comp_ratio = float(largest / mask_area)
        else:
            largest_comp_ratio = 0.0

        gx = cv2.Sobel(mag, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(mag, cv2.CV_32F, 0, 1, ksize=3)
        gmag = cv2.magnitude(gx, gy)
        grad_mean = float(np.mean(gmag))

        hsv_vis = hsv_flow_viz(flow, clip_mag=clip_mag)
        mask_vis = overlay_mask(frame, mask, alpha=0.5)
        boxes_vis = draw_bboxes(mask_vis, comps)

        cv2.putText(hsv_vis, f"Farnebäck | frame={frame_idx}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(mask_vis, f"mask thr={thr:.3f} | motion={mask_area/(mask.size):.3f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        hsv_writer.write(hsv_vis)
        mask_writer.write(mask_vis)
        box_writer.write(boxes_vis)

        f_idx.append(frame_idx)
        mag_mean.append(float(np.mean(mag)))
        mag_p95.append(float(np.percentile(mag, 95)))
        motion_ratio.append(float(mask_area / mask.size))
        n_comp.append(len(comps))
        largest_ratio.append(largest_comp_ratio)
        grad_mag_mean.append(grad_mean)
        blur.append(frame_blur_score(gray))
        noise.append(frame_noise_score(gray))
        illum.append(illumination_change(prev_gray, gray))

        prev_gray = gray

    hsv_writer.release()
    mask_writer.release()
    box_writer.release()
    cap.release()

    metrics = FBMetrics(
        frame_idx=f_idx,
        mag_mean=mag_mean,
        mag_p95=mag_p95,
        motion_ratio=motion_ratio,
        n_components=n_comp,
        largest_comp_ratio=largest_ratio,
        grad_mag_mean=grad_mag_mean,
        blur_score=blur,
        noise_score=noise,
        illum_change=illum,
    )
    return metrics, out_hsv_path, out_mask_path, out_boxes_path


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Farnebäck dense optical flow + motion masks")
    p.add_argument("--video", required=True)
    p.add_argument("--out", default="outputs/farneback")
    p.add_argument("--max-frames", type=int, default=150)
    p.add_argument("--start-frame", type=int, default=0, help="Start from this frame index")
    p.add_argument("--pyr-scale", type=float, default=0.5)
    p.add_argument("--levels", type=int, default=3)
    p.add_argument("--winsize", type=int, default=15)
    p.add_argument("--iterations", type=int, default=3)
    p.add_argument("--poly-n", type=int, default=5)
    p.add_argument("--poly-sigma", type=float, default=1.2)
    p.add_argument("--mag-thr", type=float, default=0.0)
    p.add_argument("--mag-percentile", type=float, default=95.0)
    p.add_argument("--morph-ksize", type=int, default=5)
    p.add_argument("--min-area", type=int, default=2500)
    p.add_argument("--clip-mag", type=float, default=0.0)
    return p


def main() -> None:
    args = build_argparser().parse_args()
    clip = None if args.clip_mag <= 0 else float(args.clip_mag)
    metrics, hsv_path, mask_path, boxes_path = run_farneback(
        video_path=args.video,
        out_dir=args.out,
        max_frames=args.max_frames,
        start_frame=args.start_frame,
        pyr_scale=args.pyr_scale,
        levels=args.levels,
        winsize=args.winsize,
        iterations=args.iterations,
        poly_n=args.poly_n,
        poly_sigma=args.poly_sigma,
        mag_thr=args.mag_thr,
        mag_percentile=args.mag_percentile,
        morph_ksize=args.morph_ksize,
        min_area=args.min_area,
        clip_mag=clip,
    )
    print(f"Saved Farnebäck HSV flow video: {hsv_path}")
    print(f"Saved motion mask video: {mask_path}")
    print(f"Saved bboxes video: {boxes_path}")
    print(f"Mean motion ratio: {float(np.mean(metrics.motion_ratio[1:])):.3f}")
    print(f"Mean components: {float(np.mean(metrics.n_components[1:])):.2f}")


if __name__ == "__main__":
    main()