#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Merge EgoAllo segmented NPZ outputs without modifying motion.

This replaces the old smoothing-centered post-process when EgoAllo is run with
"overlap inference + center crop".  The expected input is <output_dir>/ego_tmp,
where each .npz already contains a non-overlapping target crop.  This script
concatenates those crops, writes one visualization npz, and produces a quality
report.  It deliberately does not smooth Ts_world_cpf, root, or body quats by
default, because smoothing can hide real boundary issues and can worsen hand,
foot, and floor constraints.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

EXPECTED_OUTPUT_KEYS = (
    "Ts_world_cpf",
    "Ts_world_root",
    "body_quats",
    "left_hand_quats",
    "right_hand_quats",
    "betas",
    "frame_nums",
    "timestamps_ns",
)

SAMPLE_TIME_KEYS = {
    "Ts_world_root",
    "body_quats",
    "left_hand_quats",
    "right_hand_quats",
    "betas",
    "contacts",
}


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def is_egoallo_output_npz(path: Path) -> bool:
    try:
        with np.load(path) as d:
            return all(k in d for k in EXPECTED_OUTPUT_KEYS)
    except Exception:
        return False


def start_frame(path: Path) -> int:
    with np.load(path) as d:
        return int(np.asarray(d["frame_nums"]).reshape(-1)[0])


def normalize_quat(q, eps=1e-8):
    q = np.asarray(q, dtype=np.float64)
    return q / np.clip(np.linalg.norm(q, axis=-1, keepdims=True), eps, None)


def quat_angle_deg(q1, q2):
    q1 = normalize_quat(q1)
    q2 = normalize_quat(q2)
    dot = np.clip(np.abs(np.sum(q1 * q2, axis=-1)), -1.0, 1.0)
    return np.degrees(2 * np.arccos(dot))


def load_segments(input_dir: Path):
    paths = [p for p in input_dir.glob("*.npz") if is_egoallo_output_npz(p)]
    if not paths:
        raise FileNotFoundError(f"No EgoAllo output npz files in {input_dir}")
    paths = sorted(paths, key=lambda p: (start_frame(p), p.name))

    segs: list[dict[str, np.ndarray]] = []
    segment_info = []
    for p in paths:
        with np.load(p) as d:
            seg = {k: np.asarray(d[k]).copy() for k in d.files}
        frames = np.asarray(seg["frame_nums"]).reshape(-1)
        if frames.size == 0:
            raise ValueError(f"Empty frame_nums in {p}")
        segs.append(seg)
        segment_info.append(
            {
                "path": str(p),
                "name": p.name,
                "start_frame": int(frames[0]),
                "end_frame_inclusive": int(frames[-1]),
                "end_frame_exclusive": int(frames[-1]) + 1,
                "length": int(frames.size),
                "infer_start_index": int(np.asarray(seg.get("infer_start_index", frames[0])).reshape(-1)[0]),
                "infer_end_index": int(np.asarray(seg.get("infer_end_index", frames[-1] + 1)).reshape(-1)[0]),
                "crop_offset": int(np.asarray(seg.get("crop_offset", 0)).reshape(-1)[0]),
            }
        )
    return segs, paths, segment_info


def concat_segments(segs: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    if not segs:
        raise ValueError("No segments to concatenate")
    num_samples = segs[0]["body_quats"].shape[0]
    out: dict[str, np.ndarray] = {}
    all_keys = set().union(*(s.keys() for s in segs))
    for k in sorted(all_keys):
        if not all(k in s for s in segs):
            continue
        arrs = [s[k] for s in segs]
        # Scalar metadata such as infer_start_index cannot be concatenated usefully.
        if arrs[0].ndim == 0:
            continue
        axis = 1 if (k in SAMPLE_TIME_KEYS and arrs[0].ndim >= 2 and arrs[0].shape[0] == num_samples) else 0
        out[k] = np.concatenate(arrs, axis=axis)
    return out


def boundaries_from_segment_info(segment_info: list[dict[str, Any]]) -> list[int]:
    boundaries = []
    total = 0
    for info in segment_info[:-1]:
        total += int(info["length"])
        boundaries.append(total)
    return boundaries


def frame_continuity_report(segment_info: list[dict[str, Any]]) -> dict[str, Any]:
    issues = []
    for prev, cur in zip(segment_info[:-1], segment_info[1:]):
        prev_end = int(prev["end_frame_exclusive"])
        cur_start = int(cur["start_frame"])
        if cur_start != prev_end:
            issues.append(
                {
                    "prev_segment": prev["name"],
                    "next_segment": cur["name"],
                    "prev_end_exclusive": prev_end,
                    "next_start": cur_start,
                    "gap_or_overlap": cur_start - prev_end,
                }
            )
    return {
        "continuous": len(issues) == 0,
        "issues": issues,
    }


def boundary_quality(out: dict[str, np.ndarray], boundaries: list[int]) -> list[dict[str, Any]]:
    reports = []
    root = out.get("Ts_world_root")
    cpf = out.get("Ts_world_cpf")
    body = out.get("body_quats")
    left = out.get("left_hand_quats")
    right = out.get("right_hand_quats")

    for b in boundaries:
        item: dict[str, Any] = {"boundary_index": int(b), "transition": f"{b-1}->{b}"}
        if root is not None and 0 < b < (root.shape[1] if root.ndim == 3 else root.shape[0]):
            a, c = (root[0, b - 1], root[0, b]) if root.ndim == 3 else (root[b - 1], root[b])
            item["root_translation_jump_m"] = float(np.linalg.norm(c[4:7] - a[4:7]))
            item["root_rotation_jump_deg"] = float(quat_angle_deg(a[:4], c[:4]))
        if cpf is not None and 0 < b < cpf.shape[0]:
            a, c = cpf[b - 1], cpf[b]
            item["cpf_translation_jump_m"] = float(np.linalg.norm(c[4:7] - a[4:7]))
            item["cpf_rotation_jump_deg"] = float(quat_angle_deg(a[:4], c[:4]))
        for name, arr in (("body", body), ("left_hand", left), ("right_hand", right)):
            if arr is None:
                continue
            T = arr.shape[1] if arr.ndim == 4 else arr.shape[0]
            if not (0 < b < T):
                continue
            a, c = (arr[0, b - 1], arr[0, b]) if arr.ndim == 4 else (arr[b - 1], arr[b])
            jumps = quat_angle_deg(a, c)
            item[f"{name}_quat_jump_deg_mean"] = float(np.mean(jumps))
            item[f"{name}_quat_jump_deg_p90"] = float(np.percentile(jumps, 90))
            item[f"{name}_quat_jump_deg_max"] = float(np.max(jumps))
        reports.append(item)
    return reports


def print_report(segment_info, continuity, boundary_reports):
    print("Loaded segments:")
    for info in segment_info:
        print(
            f"  {info['name']}: {info['start_frame']}->{info['end_frame_inclusive']}, "
            f"len={info['length']}, infer={info['infer_start_index']}-{info['infer_end_index']}, "
            f"crop_offset={info['crop_offset']}"
        )
    if continuity["continuous"]:
        print("Frame continuity: OK")
    else:
        print("[Warning] Frame continuity issues:")
        for issue in continuity["issues"]:
            print("  ", issue)
    if boundary_reports:
        print("Boundary quality:")
        for r in boundary_reports:
            msg = f"  {r['transition']}"
            if "root_translation_jump_m" in r:
                msg += f" | root Δt={r['root_translation_jump_m']:.4f}m ΔR={r['root_rotation_jump_deg']:.2f}deg"
            if "body_quat_jump_deg_mean" in r:
                msg += f" | body mean/max={r['body_quat_jump_deg_mean']:.2f}/{r['body_quat_jump_deg_max']:.2f}deg"
            print(msg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--output-name", default="merged_smoothed_vis.npz")
    ap.add_argument("--report-json", type=Path, default=None)
    ap.add_argument("--print-stats", action="store_true")
    ap.add_argument("--strict-frame-continuity", action="store_true")

    # Backward-compatible arguments from the old smoother.  They are accepted but
    # intentionally not applied unless you restore a separate smoothing stage.
    ap.add_argument("--boundary-blend-frames", type=int, default=0)
    ap.add_argument("--translation-smooth-window", type=int, default=0)
    ap.add_argument("--quat-lowpass-alpha", type=float, default=0.0)
    args = ap.parse_args()

    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if args.boundary_blend_frames or args.translation_smooth_window or args.quat_lowpass_alpha:
        print(
            "[Warning] 06_merge_egoallo_segments.py is merge-only. "
            "boundary/translation/quaternion smoothing arguments are ignored. "
            "Use overlap inference + center crop to reduce boundary artifacts instead.",
            flush=True,
        )

    segs, paths, segment_info = load_segments(input_dir)
    out = concat_segments(segs)
    boundaries = boundaries_from_segment_info(segment_info)
    continuity = frame_continuity_report(segment_info)
    boundary_reports = boundary_quality(out, boundaries)

    if args.print_stats:
        print_report(segment_info, continuity, boundary_reports)

    if args.strict_frame_continuity and not continuity["continuous"]:
        raise ValueError(f"Frame continuity check failed: {continuity['issues']}")

    out["merge_segment_boundaries"] = np.asarray(boundaries, dtype=np.int64)
    out["merge_segment_lengths"] = np.asarray([int(x["length"]) for x in segment_info], dtype=np.int64)
    out["merge_is_smoothed"] = np.asarray(False)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / args.output_name
    np.savez(out_path, **out)

    report = {
        "status": "ok" if continuity["continuous"] else "warning",
        "mode": "merge_only_no_smoothing",
        "input_dir": str(input_dir),
        "output_npz": str(out_path),
        "num_segments": len(segment_info),
        "segments": segment_info,
        "frame_continuity": continuity,
        "boundaries": boundaries,
        "boundary_quality": boundary_reports,
        "ignored_smoothing_args": {
            "boundary_blend_frames": int(args.boundary_blend_frames),
            "translation_smooth_window": int(args.translation_smooth_window),
            "quat_lowpass_alpha": float(args.quat_lowpass_alpha),
        },
    }

    report_json = args.report_json
    if report_json is not None:
        report_json = report_json.expanduser().resolve()
        report_json.parent.mkdir(parents=True, exist_ok=True)
        with report_json.open("w", encoding="utf-8") as f:
            json.dump(_json_safe(report), f, ensure_ascii=False, indent=2)
        print(f"Saved merge quality report: {report_json}")

    print(f"Saved merged EgoAllo output: {out_path}")
    print("Mode: merge-only; no smoothing was applied.")


if __name__ == "__main__":
    main()
