#!/usr/bin/env python3
from __future__ import annotations

import dataclasses
import os
import sys
import time
from pathlib import Path

# 本脚本固定放在 project/scripts/05_run_egoallo_from_head.py。
# 如果 conda 环境已经安装 egoallo，则不需要 EGOALLO_ROOT；直接 import egoallo。
# 如果未安装，也可以通过 EGOALLO_ROOT 临时加入 PYTHONPATH。
SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
PROJECT_ROOT = SCRIPT_DIR.parent


def _env_path(name: str, default: Path | str | None = None) -> Path | None:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        value = default
    if value is None or str(value).strip() == "":
        return None
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def _normalize_checkpoint_dir(path: Path | None) -> Path | None:
    if path is None:
        return None
    path = Path(path).expanduser().resolve()
    if (path / "model.safetensors").is_file():
        return path
    candidate = path / "checkpoints_3000000"
    if (candidate / "model.safetensors").is_file():
        return candidate
    return path


# 可选：只有在 conda 环境无法 import egoallo 时，才使用 EGOALLO_ROOT 作为源码 fallback。
EGOALLO_REPO = _env_path("EGOALLO_ROOT")
EGOALLO_CHECKPOINT_DIR = _normalize_checkpoint_dir(_env_path("EGOALLO_CHECKPOINT_DIR"))
SMPLH_NPZ_PATH = _env_path("SMPLH_NPZ_PATH", _env_path("EGOALLO_MODEL_NPZ"))
EGOALLO_MODEL_CONFIG_PATH = _env_path("EGOALLO_MODEL_CONFIG_PATH")
EGOALLO_SOURCE_LABEL = "conda/site-packages"

import numpy as np
import torch
import yaml


def _import_egoallo_modules():
    from egoallo import fncsmpl as _fncsmpl
    from egoallo import fncsmpl_extensions as _fncsmpl_extensions
    from egoallo.guidance_optimizer_jax import GuidanceMode as _GuidanceMode
    from egoallo.hand_detection_structs import CorrespondedHamerDetections as _CorrespondedHamerDetections
    from egoallo.sampling import run_sampling_with_stitching as _run_sampling_with_stitching
    from egoallo.transforms import SE3 as _SE3
    from egoallo.transforms import SO3 as _SO3
    return (
        _fncsmpl,
        _fncsmpl_extensions,
        _GuidanceMode,
        _CorrespondedHamerDetections,
        _run_sampling_with_stitching,
        _SE3,
        _SO3,
    )


def _load_egoallo_modules_with_fallback():
    global EGOALLO_SOURCE_LABEL
    try:
        modules = _import_egoallo_modules()
        EGOALLO_SOURCE_LABEL = "conda/site-packages"
        return modules
    except Exception as first_exc:
        if EGOALLO_REPO is None:
            raise ModuleNotFoundError(
                "Cannot import egoallo from the current environment, and no EGOALLO_ROOT was provided."
            ) from first_exc
        if not EGOALLO_REPO.exists():
            raise FileNotFoundError(f"EGOALLO_ROOT does not exist: {EGOALLO_REPO}") from first_exc
        repo_str = str(EGOALLO_REPO)
        if repo_str not in sys.path:
            sys.path.insert(0, repo_str)
        try:
            modules = _import_egoallo_modules()
            EGOALLO_SOURCE_LABEL = f"fallback source {EGOALLO_REPO}"
            return modules
        except Exception as second_exc:
            raise ModuleNotFoundError(
                "Cannot import egoallo from conda or fallback source. "
                f"Fallback source: {EGOALLO_REPO}"
            ) from second_exc


(
    fncsmpl,
    fncsmpl_extensions,
    GuidanceMode,
    CorrespondedHamerDetections,
    run_sampling_with_stitching,
    SE3,
    SO3,
) = _load_egoallo_modules_with_fallback()


@dataclasses.dataclass
class Args:
    traj_root: Path
    """Custom trajectory directory containing head_trajectory.npy and timestamps.npy/timestamps.txt."""

    script_dir: Path = SCRIPT_DIR
    project_root: Path = PROJECT_ROOT

    checkpoint_dir: Path | None = EGOALLO_CHECKPOINT_DIR
    smplh_npz_path: Path | None = SMPLH_NPZ_PATH

    output_dir: Path | None = None
    """Directory for EgoAllo segment files (.npz and _args.yaml). In this pipeline it is <output_dir>/ego_tmp."""

    hamer_outputs_path: Path | None = None

    glasses_x_angle_offset: float = 0.0
    traj_length: int = 128
    """Target output segment length. With context_frames>0, each inference window is longer, but only the center target range is saved."""

    context_frames: int = 0
    """Extra frames before/after each target segment used only as inference context."""

    save_only_center: bool = True
    """When context_frames>0, save only the target center crop and discard context frames."""

    start_index: int = 0
    end_index: int | None = None
    segment_index_offset: int = 0
    num_samples: int = 1

    guidance_mode: GuidanceMode = "no_hands"
    guidance_inner: bool = False
    guidance_post: bool = True

    floor_z: float = 0.0
    model_config_path: Path | None = EGOALLO_MODEL_CONFIG_PATH

    save_traj: bool = True
    save_args: bool = True
    empty_cache_each_segment: bool = False
    allow_tf32: bool = False

    expected_head_height_min: float = 1.0
    expected_head_height_max: float = 2.2


T_CAMERA_CPF = np.array(
    [
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)


def load_custom_transforms(traj_root: Path, device: torch.device):
    head_path = traj_root / "head_trajectory.npy"
    timestamps_npy_path = traj_root / "timestamps.npy"
    timestamps_txt_path = traj_root / "timestamps.txt"

    if not head_path.exists():
        raise FileNotFoundError(head_path)

    T_np = np.load(head_path).astype(np.float32)

    if timestamps_npy_path.exists():
        timestamps_np = np.load(timestamps_npy_path).astype(np.float64)
    elif timestamps_txt_path.exists():
        table = np.genfromtxt(
            timestamps_txt_path,
            names=True,
            dtype=None,
            encoding="utf-8",
        )
        if table.shape == ():
            timestamps_np = np.array([table["timestamp"]], dtype=np.float64)
        else:
            timestamps_np = np.asarray(table["timestamp"], dtype=np.float64)
    else:
        raise FileNotFoundError(
            f"Expected either {timestamps_npy_path} or {timestamps_txt_path}"
        )

    if T_np.ndim != 3 or T_np.shape[1:] != (4, 4):
        raise ValueError(f"head_trajectory.npy should be [N,4,4], got {T_np.shape}")
    if len(T_np) != len(timestamps_np):
        raise ValueError(f"num poses != num timestamps: {len(T_np)} vs {len(timestamps_np)}")

    T_torch = torch.from_numpy(T_np).to(device)
    timestamps_torch = torch.from_numpy(timestamps_np.astype(np.float32)).to(device)
    return T_torch, timestamps_torch, timestamps_np


def load_denoiser_with_config(checkpoint_dir: Path, model_config_path: Path | None = EGOALLO_MODEL_CONFIG_PATH):
    from egoallo.inference_utils import load_denoiser as original_load_denoiser
    import os

    checkpoint_dir = Path(checkpoint_dir).expanduser().resolve()
    if model_config_path is None:
        experiment_dir = checkpoint_dir.parent
        model_config_path = experiment_dir / "model_config.yaml"
    else:
        model_config_path = Path(model_config_path).expanduser().resolve()

    model_weight_path = checkpoint_dir / "model.safetensors"
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"EgoAllo checkpoint dir not found: {checkpoint_dir}")
    if not model_weight_path.is_file():
        raise FileNotFoundError(f"EgoAllo weight not found: {model_weight_path}")
    if not model_config_path.exists():
        raise FileNotFoundError(f"Model config not found: {model_config_path}")

    original_cwd = os.getcwd()
    os.chdir(model_config_path.parent)
    try:
        return original_load_denoiser(checkpoint_dir)
    finally:
        os.chdir(original_cwd)


def print_head_height_stats_once(T_world_cpf: torch.Tensor, floor_z: float, expected_min: float, expected_max: float) -> None:
    heights = T_world_cpf[:, 2, 3].detach().cpu().numpy() - float(floor_z)
    if heights.size == 0:
        return
    h_min = float(np.min(heights))
    h_med = float(np.median(heights))
    h_max = float(np.max(heights))
    print("\n[CPF/head height check: full sequence]")
    print(f"  floor_z: {float(floor_z):.6f}")
    print(f"  height above floor min/median/max: {h_min:.6f} / {h_med:.6f} / {h_max:.6f}")
    if h_med < expected_min or h_med > expected_max:
        print(
            "  [Warning] Median head height is outside the expected range "
            f"[{expected_min:.2f}, {expected_max:.2f}] meters."
        )


def maybe_slice_hamer(all_hamer, hamer_path: Path, pose_timesteps: torch.Tensor, start: int, length: int, device: torch.device):
    # Preferred path: load HaMeR once for the whole sequence on CPU and slice per segment,
    # then move only the current segment to CUDA. This preserves speed while avoiding
    # keeping all detections in GPU memory for long videos.
    if all_hamer is not None and hasattr(all_hamer, "slice"):
        return all_hamer.slice(start, start + length).to(device)

    # Fallback for older/custom EgoAllo structs without .slice(). This still works,
    # but loses the main HaMeR-loading speedup.
    pose_timesteps_np = pose_timesteps.detach().cpu().numpy()
    segment_ts = tuple(float(x) for x in pose_timesteps_np[start + 1 : start + length + 1])
    return CorrespondedHamerDetections.load(hamer_path, segment_ts).to(device)



def _slice_time_tensor(x: torch.Tensor, start: int, end: int, infer_len: int) -> torch.Tensor:
    """Slice a tensor along its time axis when that axis is identifiable."""
    if x.ndim >= 2 and x.shape[1] == infer_len:
        return x[:, start:end, ...]
    if x.ndim >= 1 and x.shape[0] == infer_len:
        return x[start:end, ...]
    return x


def _slice_time_numpy(x: np.ndarray, start: int, end: int, infer_len: int) -> np.ndarray:
    if x.ndim >= 2 and x.shape[1] == infer_len:
        return x[:, start:end, ...]
    if x.ndim >= 1 and x.shape[0] == infer_len:
        return x[start:end, ...]
    return x

def save_segment(
    args: Args,
    traj,
    body_model,
    Ts_world_cpf: torch.Tensor,
    pose_timestamps_sec: torch.Tensor,
    infer_start: int,
    infer_length: int,
    target_start: int,
    target_length: int,
    crop_offset: int,
    segment_index: int,
) -> None:
    """Save only the target center crop from a possibly longer inference window.

    EgoAllo uses `infer_start:infer_start+infer_length` for sampling.  When
    context_frames > 0, that window contains extra context before/after the
    target segment.  The saved .npz keeps only frames
    `target_start:target_start+target_length`, so the final segment files do not
    overlap and can be concatenated without smoothing.
    """
    if not args.save_traj:
        return

    out_root = args.output_dir if args.output_dir is not None else args.traj_root / "ego_tmp"
    out_root.mkdir(parents=True, exist_ok=True)

    crop_start = int(crop_offset)
    crop_end = int(crop_offset + target_length)
    if crop_start < 0 or crop_end > infer_length:
        raise ValueError(
            f"Invalid center crop: crop=[{crop_start},{crop_end}) infer_length={infer_length} "
            f"infer_start={infer_start} target_start={target_start} target_length={target_length}"
        )

    target_end = target_start + target_length
    save_name = f"{segment_index}_{target_start}-{target_end}"

    if args.save_args:
        payload = dataclasses.asdict(args)
        payload.update({
            "target_start_index": int(target_start),
            "target_end_index": int(target_end),
            "target_length": int(target_length),
            "infer_start_index": int(infer_start),
            "infer_end_index": int(infer_start + infer_length),
            "infer_length": int(infer_length),
            "crop_offset": int(crop_offset),
            "segment_index": int(segment_index),
            "saved_frame_range": [int(target_start), int(target_end)],
        })
        (out_root / (save_name + "_args.yaml")).write_text(yaml.dump(payload), encoding="utf-8")

    posed = traj.apply_to_body(body_model)
    Ts_world_root_full = fncsmpl_extensions.get_T_world_root_from_cpf_pose(
        posed, Ts_world_cpf[..., 1:, :]
    )

    # Ts_world_cpf has infer_length+1 entries.  Output frames are Ts_world_cpf[1:].
    Ts_world_cpf_frames = Ts_world_cpf[1:, :]
    Ts_world_cpf_saved = Ts_world_cpf_frames[crop_start:crop_end, :]
    pose_timestamps_saved = pose_timestamps_sec[crop_start:crop_end]

    body_quats_full = posed.local_quats[..., :21, :]
    left_hand_quats_full = posed.local_quats[..., 21:36, :]
    right_hand_quats_full = posed.local_quats[..., 36:51, :]

    Ts_world_root_saved = _slice_time_tensor(Ts_world_root_full, crop_start, crop_end, infer_length)
    body_quats_saved = _slice_time_tensor(body_quats_full, crop_start, crop_end, infer_length)
    left_hand_quats_saved = _slice_time_tensor(left_hand_quats_full, crop_start, crop_end, infer_length)
    right_hand_quats_saved = _slice_time_tensor(right_hand_quats_full, crop_start, crop_end, infer_length)
    contacts_saved = _slice_time_tensor(traj.contacts, crop_start, crop_end, infer_length)
    betas_saved = _slice_time_tensor(traj.betas, crop_start, crop_end, infer_length)

    print(
        f"Saving target crop {target_start}-{target_end} from inference window "
        f"{infer_start}-{infer_start + infer_length} to {out_root / (save_name + '.npz')}...",
        end="",
        flush=True,
    )
    np.savez(
        out_root / (save_name + ".npz"),
        Ts_world_cpf=Ts_world_cpf_saved.detach().cpu().numpy(),
        Ts_world_root=Ts_world_root_saved.detach().cpu().numpy(),
        body_quats=body_quats_saved.detach().cpu().numpy(),
        left_hand_quats=left_hand_quats_saved.detach().cpu().numpy(),
        right_hand_quats=right_hand_quats_saved.detach().cpu().numpy(),
        contacts=contacts_saved.detach().cpu().numpy(),
        betas=betas_saved.detach().cpu().numpy(),
        frame_nums=np.arange(target_start, target_end),
        timestamps_ns=(pose_timestamps_saved.detach().cpu().numpy() * 1e9).astype(np.int64),
        infer_start_index=np.asarray(infer_start, dtype=np.int64),
        infer_end_index=np.asarray(infer_start + infer_length, dtype=np.int64),
        crop_offset=np.asarray(crop_offset, dtype=np.int64),
    )
    print("saved!", flush=True)


def main(args: Args) -> None:
    if args.traj_length <= 0:
        raise ValueError(f"traj_length must be positive, got {args.traj_length}")
    if args.context_frames < 0:
        raise ValueError(f"context_frames must be >= 0, got {args.context_frames}")

    device = torch.device("cuda")
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = bool(args.allow_tf32)
    torch.backends.cudnn.allow_tf32 = bool(args.allow_tf32)

    # Load once for all segments.
    T_world_head, pose_timesteps, timestamps_np = load_custom_transforms(args.traj_root, device)
    num_poses = T_world_head.shape[0]
    max_output_frames = num_poses - 1

    if args.start_index < 0 or args.start_index >= max_output_frames:
        raise ValueError(f"start_index must be in [0, {max_output_frames - 1}], got {args.start_index}")

    final_end = max_output_frames if args.end_index is None else min(args.end_index, max_output_frames)
    if final_end <= args.start_index:
        raise ValueError(f"end_index must be > start_index, got {final_end} <= {args.start_index}")

    print(f"{T_world_head.shape=}")
    print(f"total output frames={max_output_frames}, run range=[{args.start_index}, {final_end})")
    print(f"target segment length={args.traj_length}")
    print(f"context frames={args.context_frames}, save_only_center={args.save_only_center}")

    T_camera_cpf = torch.from_numpy(T_CAMERA_CPF).to(device)
    T_world_cpf_all = T_world_head @ T_camera_cpf
    print_head_height_stats_once(
        T_world_cpf_all,
        floor_z=args.floor_z,
        expected_min=args.expected_head_height_min,
        expected_max=args.expected_head_height_max,
    )

    checkpoint_dir = _normalize_checkpoint_dir(args.checkpoint_dir)
    if checkpoint_dir is None:
        raise ValueError("EgoAllo checkpoint is required. Set EGOALLO_CHECKPOINT_DIR or pass --checkpoint-dir.")
    model_config_path = args.model_config_path
    if model_config_path is None:
        model_config_path = EGOALLO_MODEL_CONFIG_PATH

    if args.smplh_npz_path is None:
        raise ValueError("SMPL-H model is required. Set SMPLH_NPZ_PATH/EGOALLO_MODEL_NPZ or pass --smplh-npz-path.")
    smplh_npz_path = Path(args.smplh_npz_path).expanduser().resolve()
    if not smplh_npz_path.is_file():
        raise FileNotFoundError(f"SMPL-H model not found: {smplh_npz_path}")

    print("\n[Resolved EgoAllo paths]")
    print(f"  project_root: {args.project_root}")
    print(f"  egoallo_source: {EGOALLO_SOURCE_LABEL}")
    print(f"  checkpoint_dir: {checkpoint_dir}")
    print(f"  model_config_path: {model_config_path}")
    print(f"  smplh_npz_path: {smplh_npz_path}")

    print("\n[Load denoiser/body model once]")
    t0 = time.time()
    denoiser_network = load_denoiser_with_config(checkpoint_dir, model_config_path).to(device).eval()
    body_model = fncsmpl.SmplhModel.load(smplh_npz_path).to(device)
    print(f"  loaded in {time.time() - t0:.2f}s")

    if args.hamer_outputs_path is not None:
        if not args.hamer_outputs_path.is_file():
            raise FileNotFoundError(args.hamer_outputs_path)
        if args.guidance_mode == "no_hands":
            raise ValueError("--hamer-outputs-path was provided, but --guidance-mode=no-hands would ignore it.")
        print("\n[Load HaMeR detections once on CPU]")
        t0 = time.time()
        # Use the same float32 timestamp path as the original per-segment script.
        # This avoids subtle timestamp matching differences for detections keyed by time.
        pose_timesteps_np = pose_timesteps.detach().cpu().numpy()
        all_pose_ts = tuple(float(x) for x in pose_timesteps_np[1:])
        all_hamer = CorrespondedHamerDetections.load(args.hamer_outputs_path, all_pose_ts)
        print(f"  loaded in {time.time() - t0:.2f}s")
    else:
        if args.guidance_mode in ("hamer_wrist", "hamer_reproj2", "aria_hamer"):
            raise ValueError(f"--guidance-mode={args.guidance_mode} requires --hamer-outputs-path.")
        all_hamer = None
        print("No HaMeR/HaWoR detections provided.")

    aria_detections = None
    segment_index = args.segment_index_offset
    start = args.start_index

    while start < final_end:
        target_start = start
        target_length = min(args.traj_length, final_end - target_start)
        target_end = target_start + target_length

        if args.save_only_center and args.context_frames > 0:
            infer_start = max(args.start_index, target_start - args.context_frames)
            infer_end = min(final_end, target_end + args.context_frames)
        else:
            infer_start = target_start
            infer_end = target_end

        infer_length = infer_end - infer_start
        crop_offset = target_start - infer_start

        print("\n========================================")
        print(
            f"Segment {segment_index} | target={target_start}-{target_end} "
            f"target_length={target_length} | infer={infer_start}-{infer_end} "
            f"infer_length={infer_length} crop_offset={crop_offset}"
        )
        print("========================================")

        T_world_cpf_window = T_world_cpf_all[infer_start : infer_start + infer_length + 1]
        Ts_world_cpf = (
            SE3.from_matrix(T_world_cpf_window)
            @ SE3.from_rotation(SO3.from_x_radians(T_world_head.new_tensor(args.glasses_x_angle_offset)))
        ).parameters()
        pose_timestamps_sec = pose_timesteps[infer_start + 1 : infer_start + infer_length + 1]

        hamer_detections = None
        if all_hamer is not None:
            hamer_detections = maybe_slice_hamer(
                all_hamer,
                args.hamer_outputs_path,
                pose_timesteps,
                infer_start,
                infer_length,
                device,
            )

        with torch.inference_mode():
            traj = run_sampling_with_stitching(
                denoiser_network,
                body_model=body_model,
                guidance_mode=args.guidance_mode,
                guidance_inner=args.guidance_inner,
                guidance_post=args.guidance_post,
                Ts_world_cpf=Ts_world_cpf,
                hamer_detections=hamer_detections,
                aria_detections=aria_detections,
                num_samples=args.num_samples,
                device=device,
                floor_z=args.floor_z,
                guidance_verbose=False,
            )

        save_segment(
            args,
            traj,
            body_model,
            Ts_world_cpf,
            pose_timestamps_sec,
            infer_start,
            infer_length,
            target_start,
            target_length,
            crop_offset,
            segment_index,
        )

        del traj, hamer_detections
        if args.empty_cache_each_segment:
            torch.cuda.empty_cache()

        start = target_end
        segment_index += 1

    print("\nDone.")


if __name__ == "__main__":
    import tyro

    main(tyro.cli(Args))
