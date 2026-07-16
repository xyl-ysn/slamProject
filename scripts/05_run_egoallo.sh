#!/bin/bash
set -euo pipefail

# 用法: ./05_run_egoallo.sh <traj_root> <segment_output_dir> [traj_length]
# 说明：
#   traj_root:          EgoAllo 输入目录，里面需要有 head_trajectory.npy / timestamps.txt / hamer_outputs.pkl
#   segment_output_dir: EgoAllo 分段输出目录，保存每段 .npz 和 _args.yaml；本项目传入 <output_dir>/ego_tmp
#   traj_length:        分段长度。通常由 run_body_mesh.py 从 config/body_mesh.yaml 的 egoallo.segment_length 传入。
#
# 本脚本不读取 YAML，只读取 run_body_mesh.py 注入的环境变量，并转成
# 05_run_egoallo_from_head.py 的命令行参数。

TRAJ_ROOT="${1:-}"
SEGMENT_OUT_ROOT="${2:-}"
TRAJ_LENGTH="${3:-256}"

if [ -z "$TRAJ_ROOT" ] || [ -z "$SEGMENT_OUT_ROOT" ]; then
    echo "❌ 用法: $0 <traj_root> <segment_output_dir> [traj_length]"
    echo "   示例: $0 /tmp/out/egoallo_inputs /tmp/out/ego_tmp 256"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_SCRIPT="$SCRIPT_DIR/05_run_egoallo_from_head.py"

PYTHON_BIN="${PYTHON_EXECUTABLE:-python3}"

EGOALLO_CHECKPOINT_DIR_VALUE="${EGOALLO_CHECKPOINT_DIR:-}"
SMPLH_NPZ_PATH_VALUE="${SMPLH_NPZ_PATH:-${EGOALLO_MODEL_NPZ:-}}"
MODEL_CONFIG_PATH_VALUE="${EGOALLO_MODEL_CONFIG_PATH:-}"
HAMER_OUTPUTS="$TRAJ_ROOT/hamer_outputs.pkl"

if [ ! -f "$PY_SCRIPT" ]; then
    echo "❌ 找不到脚本: $PY_SCRIPT"
    exit 1
fi

# EGOALLO_ROOT 只作为 Python 脚本的 fallback 源码路径。
# 这里不要提前加入 PYTHONPATH，否则会覆盖 conda/site-packages 的优先级。
if [ -n "${EGOALLO_ROOT:-}" ] && [ ! -d "$EGOALLO_ROOT" ]; then
    echo "❌ EGOALLO_ROOT 不存在: $EGOALLO_ROOT"
    exit 1
fi

if [ -z "$EGOALLO_CHECKPOINT_DIR_VALUE" ]; then
    echo "❌ 未设置 EGOALLO_CHECKPOINT_DIR。请在 config/body_mesh.yaml 的 model_weights.egoallo_checkpoint_dir 中配置。"
    exit 1
fi

# 兼容两种配置：
# 1) 指向包含 model.safetensors 的 checkpoint 目录；
# 2) 指向 checkpoint root，实际权重在 checkpoints_3000000/ 下。
if [ -f "$EGOALLO_CHECKPOINT_DIR_VALUE/model.safetensors" ]; then
    RESOLVED_CHECKPOINT_DIR="$EGOALLO_CHECKPOINT_DIR_VALUE"
elif [ -f "$EGOALLO_CHECKPOINT_DIR_VALUE/checkpoints_3000000/model.safetensors" ]; then
    RESOLVED_CHECKPOINT_DIR="$EGOALLO_CHECKPOINT_DIR_VALUE/checkpoints_3000000"
else
    echo "❌ 找不到 EgoAllo 权重 model.safetensors:"
    echo "   $EGOALLO_CHECKPOINT_DIR_VALUE/model.safetensors"
    echo "   或 $EGOALLO_CHECKPOINT_DIR_VALUE/checkpoints_3000000/model.safetensors"
    exit 1
fi

if [ -z "$MODEL_CONFIG_PATH_VALUE" ]; then
    MODEL_CONFIG_PATH_VALUE="$(cd "$RESOLVED_CHECKPOINT_DIR/.." && pwd)/model_config.yaml"
fi
if [ ! -f "$MODEL_CONFIG_PATH_VALUE" ]; then
    echo "❌ 找不到 EgoAllo 配置: $MODEL_CONFIG_PATH_VALUE"
    exit 1
fi

if [ -z "$SMPLH_NPZ_PATH_VALUE" ]; then
    echo "❌ 未设置 SMPLH_NPZ_PATH 或 EGOALLO_MODEL_NPZ。请在 config/body_mesh.yaml 中配置 SMPL-H model.npz。"
    exit 1
fi
if [ ! -f "$SMPLH_NPZ_PATH_VALUE" ]; then
    echo "❌ 找不到 SMPL-H 模型: $SMPLH_NPZ_PATH_VALUE"
    exit 1
fi
if [ ! -f "$TRAJ_ROOT/head_trajectory.npy" ]; then
    echo "❌ 找不到 head_trajectory.npy in $TRAJ_ROOT"
    exit 1
fi
if [ ! -f "$HAMER_OUTPUTS" ]; then
    echo "❌ Missing hand detections: $HAMER_OUTPUTS"
    exit 1
fi

mkdir -p "$SEGMENT_OUT_ROOT"

export EGOALLO_CHECKPOINT_DIR="$RESOLVED_CHECKPOINT_DIR"
export EGOALLO_MODEL_CONFIG_PATH="$MODEL_CONFIG_PATH_VALUE"
export SMPLH_NPZ_PATH="$SMPLH_NPZ_PATH_VALUE"

CMD=(
    "$PYTHON_BIN" "$PY_SCRIPT"
    --traj-root "$TRAJ_ROOT"
    --output-dir "$SEGMENT_OUT_ROOT"
    --hamer-outputs-path "$HAMER_OUTPUTS"
    --checkpoint-dir "$RESOLVED_CHECKPOINT_DIR"
    --model-config-path "$MODEL_CONFIG_PATH_VALUE"
    --smplh-npz-path "$SMPLH_NPZ_PATH_VALUE"
    --guidance-mode "${EGOALLO_GUIDANCE_MODE:-hamer_wrist}"
    --traj-length "$TRAJ_LENGTH"
    --context-frames "${EGOALLO_CONTEXT_FRAMES:-0}"
    --glasses-x-angle-offset "${EGOALLO_GLASSES_X_ANGLE_OFFSET:-0.0}"
    --start-index "${EGOALLO_START_INDEX:-0}"
    --segment-index-offset "${EGOALLO_SEGMENT_INDEX_OFFSET:-0}"
    --num-samples "${EGOALLO_NUM_SAMPLES:-1}"
    --floor-z "${EGOALLO_FLOOR_Z:-0.0}"
    --expected-head-height-min "${EGOALLO_EXPECTED_HEAD_HEIGHT_MIN:-1.0}"
    --expected-head-height-max "${EGOALLO_EXPECTED_HEAD_HEIGHT_MAX:-2.2}"
)

if [ -n "${EGOALLO_END_INDEX:-}" ]; then
    CMD+=(--end-index "$EGOALLO_END_INDEX")
fi

if [ "${EGOALLO_GUIDANCE_INNER:-1}" = "1" ]; then
    CMD+=(--guidance-inner)
fi

if [ "${EGOALLO_GUIDANCE_POST:-1}" = "1" ]; then
    CMD+=(--guidance-post)
else
    CMD+=(--no-guidance-post)
fi

if [ "${EGOALLO_SAVE_ONLY_CENTER:-1}" = "1" ]; then
    CMD+=(--save-only-center)
else
    CMD+=(--no-save-only-center)
fi

if [ "${EGOALLO_SAVE_TRAJ:-1}" = "1" ]; then
    CMD+=(--save-traj)
else
    CMD+=(--no-save-traj)
fi

if [ "${EGOALLO_SAVE_ARGS:-1}" = "1" ]; then
    CMD+=(--save-args)
else
    CMD+=(--no-save-args)
fi

if [ "${EGOALLO_EMPTY_CACHE_EACH_SEGMENT:-0}" = "1" ]; then
    CMD+=(--empty-cache-each-segment)
fi

if [ "${EGOALLO_ALLOW_TF32:-0}" = "1" ]; then
    CMD+=(--allow-tf32)
fi

echo "[EgoAllo CMD] ${CMD[*]}"
"${CMD[@]}"
