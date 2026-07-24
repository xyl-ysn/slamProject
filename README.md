# EgoPose Project Scripts Readme

本项目的 `scripts` 目录是一个“上游检测产物 -> 对齐/配准 -> 估计 -> 生成轨迹与人体网格” 的完整流水线。  
主入口为 `scripts/run_body_mesh.py`，其它脚本基本都是它的子步骤。

## 一、入口与输入

### 1.1 主入口

```bash
python /home/xieyongling/slam_project/ego-pose-project/scripts/run_body_mesh.py \
  --config /home/xieyongling/slam_project/ego-pose-project/config/body_mesh.yaml \
  --frames-dir /path/to/frames \
  --cam-space-dir /path/to/cam_space \
  --hawor-hands-vggt-world /path/to/hawor_hands.jsonl \
  --hawor-to-vggt-alignment /path/to/hawor_to_vggt_alignment.json \
  --vggt-head-pose /path/to/vggt_head_pose.jsonl \
  --scene-points-ply /path/to/scene_points.ply \
  --output-dir /path/to/output
```

必填输入：
- `frames-dir`：原视频抽帧图像目录
- `cam-space-dir`：双目/多目 `cam_space` 目录
- `hawor-hands-vggt-world`：HaWoR 手关键点
- `hawor-to-vggt-alignment`：对齐矩阵 json
- `vggt-head-pose`：VGGT 头部姿态 jsonl
- `scene-points-ply`：场景点云
- `config/body_mesh.yaml`：全局参数

### 1.2 插件入口

如果通过服务框架调用，插件在  
- [scripts/model_plugin.py](/home/xieyongling/slam_project/ego-pose-project/scripts/model_plugin.py)  
会直接调用上述 `run_body_mesh.py`。  

## 二、执行流程（run_body_mesh）

`run_body_mesh.py` 串联以下脚本（依次执行）：

1. `01_convert_jsonl_to_camera_poses.py`  
   从 VGGT head jsonl 生成：
   - `tmp/camera_poses.txt`
   - `egoallo_inputs/timestamps.txt`
   - `tmp/intrinsics.txt`

2. `02_extract_hawor_world_hands.py`  
   HaWoR 手检测预处理，加入权重/软起止策略。

3. `03_stabilize_hawor_hands_for_egoallo.py`  
   稳定手部轨迹（去抖/去抖动跳变）。

4. `extract_hawor_to_vggt_rotation.py`  
   读对齐 json，产出 `tmp/R_hawor_to_vggt.npy`。

5. `04_build_egoallo_hamer_from_hawor_world.py`  
   产出 `tmp/hamer_outputs.pkl`（EgoAllo 输入约束）。

6. `run_geocalib.py`  
   用样本帧跑 GeoCalib，产出 `tmp/geocalib_gravity.npz`。

7. `01_estimate_gravity_da3_geocalib.py` + `02_apply_alignment_da3_geocalib.py`  
   估计/应用重力和场景主方向，产出：
   - `tmp/R_align.npy`
   - `tmp/alignment_transform.npz`
   - `tmp/head_trajectory.npy`

8. `04_estimate_moge_metric_scale.py`（可选）  
   仅在 `metric_scale.scale_priority` 中需要且手部缩放失败时触发。  

9. `05_apply_metric_scale.py`（关键）
   按 `scale_priority` 执行：`hands -> moge -> height -> none`，并输出：
   - `egoallo_outputs` 前端 jsonl（可配置）
   - `egoallo_outputs/final_scene.ply`
   - `tmp/metric_scale_metadata.json`
   - `tmp/hand_scale_bones.json`
   - `tmp/moge_metric_scale/moge_scale_estimate.json`（如启用 MoGe）

10. `05_run_egoallo.sh` -> `05_run_egoallo_from_head.py`  
   对 `egoallo_inputs/` 中轨迹做分段推理（支持并行/预算调度）。

11. `06_smooth_egoallo_segments.py`  
   合并分段 npz，做分段边界 blend 与可选时序平滑。

12. 写总结日志  
   `tmp/pipeline_run_log.json` 记录每个子命令耗时与完整 stdout/stderr。

## 三、目录结构（输出）

默认输出结构（可在 `output_layout` 调整）：

- `tmp/`：中间产物（日志、临时轨迹/点云/对齐参数等）
- `egoallo_inputs/`：EgoAllo 输入文件
  - `head_trajectory.npy`
  - `timestamps.txt`
  - `hamer_outputs.pkl`
- `ego_tmp/`：每段推理输出
  - `xx_aa-bb.npz`
  - `xx_aa-bb_args.yaml`
- `egoallo_outputs/`：最终结果
  - `merged_smoothed_vis.npz`
  - `final_scene.ply`
  - `head_trajectory_egoallo.jsonl`（若启用 frontend）
  - `hand_keypoints_egoallo.jsonl`（若启用 frontend）

`tmp`、`ego_tmp`、`egoallo_inputs` 可按配置清理。

## 四、关键参数（`config/body_mesh.yaml`）

### 4.1 资源与路径
- `model_sources.*`：源码 fallback 目录（当前优先使用 conda/site-packages，导入失败时才 fallback 到 `EGOALLO_ROOT` / `GEOCALIB_ROOT` / `MOGE_ROOT`）。
- `model_weights.*`：`egoallo_checkpoint_dir`、`moge_model_pt`、`geocalib_pinhole_tar`、`mano_*.pkl` 等。
- `output_layout.*`：各目录名与清理开关。

### 4.2 手部预处理
- `hawor_extract.*`：提取阈值与平滑规则
- `hawor_stabilize.*`：去抖动/分段边界参数
- `hamer_build.*`：`cam_space` 子目录与 MANO 关键点 key 名

### 4.3 对齐与重力
- `geocalib.*`：GeoCalib 抽样与采样策略
- `alignment.*`：是否使用 geocalib prior、RANSAC fallback 及高度验收阈值

### 4.4 量纲缩放（重点）
- `metric_scale.scale_priority`：默认 `hands -> moge -> height -> none`
- `metric_scale.hand_fallback.*`：手部优先级参数
- `metric_scale.moge.*`：MoGe 候选/质量检查参数
- `metric_scale.scale_sanity.*`：MoGe 头高验收（只对 moge）
- `metric_scale.height_fallback.*`：疑似直立阶段高度 fallback

### 4.5 EgoAllo
- `egoallo.target_segment_length`
- `egoallo.context_frames`
- `egoallo.save_only_center`
- `egoallo.parallel_workers`
- `egoallo.available_vram_gb`
- `egoallo.estimated_model_vram_gb`
- `egoallo.estimated_vram_gb_per_frame`

估算公式（用于并行预算）：

`estimated_model_vram_gb + estimated_vram_gb_per_frame * infer_length`

其中 `infer_length` = `target_segment_length + 2*context_frames`（边界段会收敛）。

### 4.6 平滑
- `smooth.boundary_blend_frames`
- `smooth.translation_smooth_window`
- `smooth.quat_lowpass_alpha`

## 五、常用排查点

1. 并行启动失败（`available_vram_gb`）
   - 降低 `target_segment_length`/`context_frames` 或提高 `available_vram_gb`。

2. 量纲缩放跳到 `scale=1.0`
   - 看 `tmp/metric_scale_metadata.json`，通常是 hand/moge 验收全部失败。

3. 运行中参数来源不一致
   - 所有真实执行参数都在 `run_body_mesh.py -> env` 中转给 EgoAllo；单独在代码默认值和 `egoallo.*` 不要混淆。

4. 输出异常或缺文件
   - 优先看 `tmp/pipeline_run_log.json` 每步 `stderr/stdout` + `return_code`。
   - 再看 `tmp/motion_quality_report.json`（平滑质量）和 `tmp/metric_scale_metadata.json`（缩放决策）。

## 六、独立脚本快速入口

- [scripts/esmateG/run_geocalib.py](/home/xieyongling/slam_project/ego-pose-project/scripts/esmateG/run_geocalib.py)  
  `--img-dir ... --output ... --num-samples ...`

- [scripts/esmateG/01_estimate_gravity_da3_geocalib.py](/home/xieyongling/slam_project/ego-pose-project/scripts/esmateG/01_estimate_gravity_da3_geocalib.py)

- [scripts/esmateG/02_apply_alignment_da3_geocalib.py](/home/xieyongling/slam_project/ego-pose-project/scripts/esmateG/02_apply_alignment_da3_geocalib.py)

- [scripts/05_run_egoallo.sh](/home/xieyongling/slam_project/ego-pose-project/scripts/05_run_egoallo.sh)  
- [scripts/05_run_egoallo_from_head.py](/home/xieyongling/slam_project/ego-pose-project/scripts/05_run_egoallo_from_head.py)

- [scripts/06_smooth_egoallo_segments.py](/home/xieyongling/slam_project/ego-pose-project/scripts/06_smooth_egoallo_segments.py)
- [scripts/convert_to_ego/04_estimate_moge_metric_scale.py](/home/xieyongling/slam_project/ego-pose-project/scripts/convert_to_ego/04_estimate_moge_metric_scale.py)
- [scripts/convert_to_ego/05_apply_metric_scale.py](/home/xieyongling/slam_project/ego-pose-project/scripts/convert_to_ego/05_apply_metric_scale.py)

## 七、输出产物清单（建议关注）
- `tmp/pipeline_run_log.json`
- `tmp/metric_scale_metadata.json`
- `tmp/motion_quality_report.json`
- `egoallo_outputs/final_scene.ply`
- `egoallo_outputs/merged_smoothed_vis.npz`
- `egoallo_outputs/head_trajectory_egoallo.jsonl`（如启用）
- `egoallo_outputs/hand_keypoints_egoallo.jsonl`（如启用）
- `tmp/moge_metric_scale/moge_scale_estimate.json`
