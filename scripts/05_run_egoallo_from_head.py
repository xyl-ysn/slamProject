#!/usr/bin/env python3
from __future__ import annotations

import dataclasses
import json
import os
import subprocess
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

    parallel_workers: int = 1
    """Number of separate worker processes to run EgoAllo segments in parallel."""

    available_vram_gb: float = 0.0
    """VRAM budget for concurrent EgoAllo workers. <=0 disables memory-budget scheduling."""

    estimated_model_vram_gb: float = 6.0
    """Conservative fixed VRAM estimate for one EgoAllo worker process."""

    estimated_vram_gb_per_frame: float = 0.020
    """Conservative extra VRAM estimate per inference frame."""

    worker_task_file: Path | None = None
    worker_segment_index: int | None = None
    worker_target_start: int | None = None
    worker_target_end: int | None = None
    worker_infer_start: int | None = None
    worker_infer_end: int | None = None
    worker_crop_offset: int | None = None


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


@dataclasses.dataclass(frozen=True)
class SegmentTask:
    segment_index: int
    target_start: int
    target_end: int
    infer_start: int
    infer_end: int
    crop_offset: int

    @property
    def target_length(self) -> int:
        return int(self.target_end - self.target_start)

    @property
    def infer_length(self) -> int:
        return int(self.infer_end - self.infer_start)


def build_segment_tasks(args: Args, max_output_frames: int) -> tuple[list[SegmentTask], int]:
    final_end = max_output_frames if args.end_index is None else min(args.end_index, max_output_frames)
    if final_end <= args.start_index:
        raise ValueError(f"end_index must be > start_index, got {final_end} <= {args.start_index}")

    tasks: list[SegmentTask] = []
    segment_index = int(args.segment_index_offset)
    start = int(args.start_index)
    while start < final_end:
        target_start = start
        target_length = min(int(args.traj_length), final_end - target_start)
        target_end = target_start + target_length

        if args.save_only_center and args.context_frames > 0:
            infer_start = max(args.start_index, target_start - args.context_frames)
            infer_end = min(final_end, target_end + args.context_frames)
        else:
            infer_start = target_start
            infer_end = target_end

        tasks.append(SegmentTask(
            segment_index=segment_index,
            target_start=target_start,
            target_end=target_end,
            infer_start=infer_start,
            infer_end=infer_end,
            crop_offset=target_start - infer_start,
        ))

        start = target_end
        segment_index += 1

    return tasks, final_end


def worker_task_from_args(args: Args) -> SegmentTask | None:
    fields = (
        args.worker_segment_index,
        args.worker_target_start,
        args.worker_target_end,
        args.worker_infer_start,
        args.worker_infer_end,
        args.worker_crop_offset,
    )
    if all(v is None for v in fields):
        return None
    if any(v is None for v in fields):
        raise ValueError("Worker segment args must be provided together.")
    return SegmentTask(
        segment_index=int(args.worker_segment_index),
        target_start=int(args.worker_target_start),
        target_end=int(args.worker_target_end),
        infer_start=int(args.worker_infer_start),
        infer_end=int(args.worker_infer_end),
        crop_offset=int(args.worker_crop_offset),
    )


def _parse_worker_tasks_file(worker_task_path: Path) -> list[SegmentTask]:
    with worker_task_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        tasks_payload = data.get("tasks", [])
    else:
        tasks_payload = data

    if not isinstance(tasks_payload, list):
        raise ValueError(f"worker task file must contain a list, got {type(tasks_payload)}: {worker_task_path}")
    tasks: list[SegmentTask] = []
    for i, item in enumerate(tasks_payload):
        if not isinstance(item, dict):
            raise ValueError(f"worker task entry {i} must be a dict: {item!r}")
        required = (
            "segment_index",
            "target_start",
            "target_end",
            "infer_start",
            "infer_end",
            "crop_offset",
        )
        missing = [k for k in required if k not in item]
        if missing:
            raise ValueError(f"worker task entry {i} missing keys {missing} in {worker_task_path}")
        tasks.append(
            SegmentTask(
                segment_index=int(item["segment_index"]),
                target_start=int(item["target_start"]),
                target_end=int(item["target_end"]),
                infer_start=int(item["infer_start"]),
                infer_end=int(item["infer_end"]),
                crop_offset=int(item["crop_offset"]),
            )
        )
    return tasks


def worker_tasks_from_args(args: Args) -> list[SegmentTask] | None:
    task_file = args.worker_task_file
    task_from_args = worker_task_from_args(args)

    if task_file is not None and task_from_args is not None:
        raise ValueError("Provide only worker_task_file or worker_segment_* args, not both.")

    if task_file is not None:
        if not task_file.exists():
            raise FileNotFoundError(f"worker task file not found: {task_file}")
        return _parse_worker_tasks_file(task_file)

    if task_from_args is not None:
        return [task_from_args]
    return None


def estimate_segment_vram_gb(args: Args, task: SegmentTask) -> float:
    return float(args.estimated_model_vram_gb) + float(args.estimated_vram_gb_per_frame) * float(task.infer_length)


def estimate_worker_vram_gb(args: Args, tasks: list[SegmentTask]) -> float:
    if not tasks:
        return 0.0
    return max(estimate_segment_vram_gb(args, task) for task in tasks)


def _append_bool_arg(cmd: list[str], name: str, value: bool) -> None:
    cmd.append(f"--{name}" if bool(value) else f"--no-{name}")


def build_worker_command(args: Args, tasks: list[SegmentTask], worker_task_file: Path | None = None) -> list[str]:
    cmd = [
        sys.executable,
        str(SCRIPT_PATH),
        "--traj-root", str(args.traj_root),
        "--output-dir", str(args.output_dir),
        "--checkpoint-dir", str(args.checkpoint_dir),
        "--model-config-path", str(args.model_config_path),
        "--smplh-npz-path", str(args.smplh_npz_path),
        "--glasses-x-angle-offset", str(args.glasses_x_angle_offset),
        "--traj-length", str(args.traj_length),
        "--context-frames", str(args.context_frames),
        "--start-index", str(args.start_index),
        "--segment-index-offset", str(args.segment_index_offset),
        "--num-samples", str(args.num_samples),
        "--guidance-mode", str(args.guidance_mode),
        "--floor-z", str(args.floor_z),
        "--expected-head-height-min", str(args.expected_head_height_min),
        "--expected-head-height-max", str(args.expected_head_height_max),
        "--parallel-workers", "1",
        "--available-vram-gb", "0",
        "--estimated-model-vram-gb", str(args.estimated_model_vram_gb),
        "--estimated-vram-gb-per-frame", str(args.estimated_vram_gb_per_frame),
    ]
    if args.hamer_outputs_path is not None:
        cmd.extend(["--hamer-outputs-path", str(args.hamer_outputs_path)])
    if args.end_index is not None:
        cmd.extend(["--end-index", str(args.end_index)])
    _append_bool_arg(cmd, "save-only-center", args.save_only_center)
    _append_bool_arg(cmd, "guidance-inner", args.guidance_inner)
    _append_bool_arg(cmd, "guidance-post", args.guidance_post)
    _append_bool_arg(cmd, "save-traj", args.save_traj)
    _append_bool_arg(cmd, "save-args", args.save_args)
    if args.empty_cache_each_segment:
        cmd.append("--empty-cache-each-segment")
    if args.allow_tf32:
        cmd.append("--allow-tf32")

    if worker_task_file is not None:
        cmd.extend(["--worker-task-file", str(worker_task_file)])
    elif len(tasks) == 1:
        task = tasks[0]
        cmd.extend(
            [
                "--worker-segment-index",
                str(task.segment_index),
                "--worker-target-start",
                str(task.target_start),
                "--worker-target-end",
                str(task.target_end),
                "--worker-infer-start",
                str(task.infer_start),
                "--worker-infer-end",
                str(task.infer_end),
                "--worker-crop-offset",
                str(task.crop_offset),
            ]
        )
    else:
        raise ValueError("build_worker_command requires either worker_task_file or one worker task.")
    return cmd


def _write_worker_task_file(output_dir: Path, worker_id: int, tasks: list[SegmentTask]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    task_path = output_dir / f"__egoallo_worker_{worker_id}_tasks.json"
    payload = [dataclasses.asdict(task) for task in tasks]
    task_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return task_path


def run_parallel_coordinator(args: Args) -> None:
    head_path = args.traj_root / "head_trajectory.npy"
    if not head_path.exists():
        raise FileNotFoundError(head_path)
    T_np = np.load(head_path, mmap_mode="r")
    if T_np.ndim != 3 or T_np.shape[1:] != (4, 4):
        raise ValueError(f"head_trajectory.npy should be [N,4,4], got {T_np.shape}")
    max_output_frames = int(T_np.shape[0] - 1)
    if args.start_index < 0 or args.start_index >= max_output_frames:
        raise ValueError(f"start_index must be in [0, {max_output_frames - 1}], got {args.start_index}")

    tasks, final_end = build_segment_tasks(args, max_output_frames)
    max_workers = max(1, int(args.parallel_workers))
    budget = float(args.available_vram_gb)

    print(f"{T_np.shape=}")
    print(f"total output frames={max_output_frames}, run range=[{args.start_index}, {final_end})")
    print(f"target segment length={args.traj_length}")
    print(f"context frames={args.context_frames}, save_only_center={args.save_only_center}")
    print(f"parallel_workers={max_workers}, available_vram_gb={budget:.3f}")
    print(
        "VRAM estimate: "
        f"{args.estimated_model_vram_gb:.3f} GB/process + "
        f"{args.estimated_vram_gb_per_frame:.4f} GB/frame"
    )

    if budget > 0:
        for task in tasks:
            est = estimate_segment_vram_gb(args, task)
            if est > budget:
                raise RuntimeError(
                    f"Segment {task.segment_index} estimated VRAM {est:.3f} GB exceeds "
                    f"available_vram_gb={budget:.3f}. Reduce segment/context length or increase budget."
                )

    if not tasks:
        print("[EgoAllo parallel] no tasks to run.")
        return

    num_workers = min(max_workers, len(tasks))
    task_groups: list[list[SegmentTask]] = [[] for _ in range(num_workers)]
    for i, task in enumerate(tasks):
        task_groups[i % num_workers].append(task)
    task_groups = [g for g in task_groups if g]

    pending = []
    for worker_idx, group in enumerate(task_groups):
        est = estimate_worker_vram_gb(args, group)
        if budget > 0 and est > budget:
            raise RuntimeError(
                f"Worker {worker_idx} estimated VRAM {est:.3f} GB exceeds "
                f"available_vram_gb={budget:.3f} for task batch."
            )
        worker_task_file = _write_worker_task_file(
            Path(args.output_dir) if args.output_dir is not None else args.traj_root / "ego_tmp",
            worker_idx,
            group,
        )
        target_summary = f"{group[0].target_start}-{group[-1].target_end}"
        infer_summary = f"{group[0].infer_start}-{group[-1].infer_end}"
        pending.append({
            "worker_id": int(worker_idx),
            "tasks": group,
            "task_file": worker_task_file,
            "worker_vram": float(est),
            "summary": f"segment={group[0].segment_index}-{group[-1].segment_index}, target={target_summary}, infer={infer_summary}",
        })

    running: list[dict[str, object]] = []
    completed = 0
    temp_files: set[Path] = {item["task_file"] for item in pending}

    while pending or running:
        still_running = []
        for item in running:
            proc = item["proc"]
            assert isinstance(proc, subprocess.Popen)
            code = proc.poll()
            if code is None:
                still_running.append(item)
                continue
            if code != 0:
                raise subprocess.CalledProcessError(code, item["cmd"])
            task_count = len(item["tasks"]) if isinstance(item["tasks"], list) else 0
            completed += task_count
            summary = item["summary"] if isinstance(item["summary"], str) else ""
            print(
                f"[EgoAllo parallel] completed worker {item['worker_id']} ({completed}/{len(tasks)} segments). "
                f"{summary}",
                flush=True,
            )
        running = still_running

        used_vram = sum(float(item["worker_vram"]) for item in running)
        launched = False
        while pending and len(running) < max_workers:
            item = pending[0]
            est = float(item["worker_vram"])
            if budget > 0 and used_vram + est > budget:
                break
            pending.pop(0)
            task_file = item["task_file"]
            assert isinstance(task_file, Path)
            cmd = build_worker_command(args, item["tasks"], task_file)
            print(
                f"[EgoAllo parallel] launch worker {item['worker_id']}: "
                f"{item['summary']} "
                f"estimated_vram={est:.3f}GB used_after={used_vram + est:.3f}GB",
                flush=True,
            )
            proc = subprocess.Popen(cmd, env=os.environ.copy())
            running.append(
                {
                    "proc": proc,
                    "worker_id": item["worker_id"],
                    "tasks": item["tasks"],
                    "worker_vram": item["worker_vram"],
                    "summary": item["summary"],
                    "task_file": item["task_file"],
                    "cmd": cmd,
                }
            )
            used_vram += est
            launched = True

        if not launched and running:
            time.sleep(1.0)
        elif pending and not running:
            raise RuntimeError("No EgoAllo worker could be launched under the current VRAM budget.")
    # cleanup coordinator-generated task files
    for f in temp_files:
        if f.is_file():
            f.unlink(missing_ok=True)

    print("\nDone.")



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
    worker_tasks = worker_tasks_from_args(args)
    if worker_tasks is None and int(args.parallel_workers) > 1:
        run_parallel_coordinator(args)
        return

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

    if worker_tasks is not None:
        tasks = worker_tasks
        final_end_for_log = max(task.target_end for task in tasks)
        if args.start_index < 0 or args.start_index >= tasks[0].infer_start + 1:
            raise ValueError(f"start_index should not be inconsistent with worker task definitions: {args.start_index}")
        if final_end_for_log <= args.start_index:
            raise ValueError(
                f"worker tasks final end must be > start_index, got {final_end_for_log} <= {args.start_index}"
            )
    else:
        tasks, final_end_for_log = build_segment_tasks(args, max_output_frames)

    print(f"{T_world_head.shape=}")
    print(f"total output frames={max_output_frames}, run range=[{args.start_index}, {final_end_for_log})")
    print(f"target segment length={args.traj_length}")
    print(f"context frames={args.context_frames}, save_only_center={args.save_only_center}")
    if worker_tasks is not None and len(worker_tasks) == 1:
        worker_task = worker_tasks[0]
        print(
            f"worker segment={worker_task.segment_index} "
            f"target={worker_task.target_start}-{worker_task.target_end} "
            f"infer={worker_task.infer_start}-{worker_task.infer_end}"
        )

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

    worker_mode = worker_tasks is not None
    if args.hamer_outputs_path is not None and not worker_mode:
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
    elif args.hamer_outputs_path is not None and worker_mode:
        if not args.hamer_outputs_path.is_file():
            raise FileNotFoundError(args.hamer_outputs_path)
        if args.guidance_mode == "no_hands":
            raise ValueError("--hamer-outputs-path was provided, but --guidance-mode=no-hands would ignore it.")
        print("\n[Worker mode] Load HaMeR detections once for all assigned segments.")
        pose_timesteps_np = pose_timesteps.detach().cpu().numpy()
        all_pose_ts = tuple(float(x) for x in pose_timesteps_np[1:])
        all_hamer = CorrespondedHamerDetections.load(args.hamer_outputs_path, all_pose_ts)
    else:
        if args.guidance_mode in ("hamer_wrist", "hamer_reproj2", "aria_hamer"):
            raise ValueError(f"--guidance-mode={args.guidance_mode} requires --hamer-outputs-path.")
        all_hamer = None
        print("No HaMeR/HaWoR detections provided.")

    aria_detections = None

    for task in tasks:
        target_start = task.target_start
        target_end = task.target_end
        target_length = task.target_length
        infer_start = task.infer_start
        infer_end = task.infer_end
        infer_length = task.infer_length
        crop_offset = task.crop_offset
        segment_index = task.segment_index
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
        if args.hamer_outputs_path is not None:
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

    print("\nDone.")


if __name__ == "__main__":
    import tyro

    main(tyro.cli(Args))
