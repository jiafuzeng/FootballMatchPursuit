"""本地足球视频的动作定位入口。

作用：把一段比赛视频交给 OpenSportsLib，找出动作时间点并写成 JSON。
写入前按 LABEL_TO_ACTION 把类别改成中文，并按 --min-confidence 丢掉低置信度事件。其余字段保持模型原样。
结果不含球员身份和接球人。

流程：
1. 读取视频（默认 test_data/test.mp4，也可用 --video 指定）。
2. 可选地用 --max-seconds 只保留开头若干秒，便于短时验证。
3. 加载 Hugging Face 上的 OSL-loc-snbas-2023-e2e（RegNetY-008 + GRU）。
4. 按 --fps 抽帧（默认 2 帧/秒）、每 100 帧一个窗口做定位。
5. 采用去重后的动作点，写入 outputs/opensportslib/test.step0.step1.json。

运行环境是 conda 环境 osl。这台机器的驱动只报告 CUDA 12.2，官方安装脚本
要求 CUDA 12.6，因此使用仍能在 Pascal 上执行的 PyTorch 2.5.1+cu121。
未安装 DALI，视频由 OpenCV 解码。6GB 显存下 batch size 固定为 1。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

from label_map import LABEL_TO_ACTION

# 库在推理时会读取 RUN_ID；关闭 W&B，避免没有登录信息时卡住。
os.environ.setdefault("RUN_ID", "football-match-pursuit")
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("WANDB_DISABLED", "true")
# 随后加载的足球权重会覆盖骨干，不必再下载 ImageNet 预训练参数。
os.environ.setdefault("OSL_PRETRAINED_WEIGHTS", "0")
# 本机访问不了 huggingface.co，只使用已经下载好的文件。
# 权重在当前用户的 ~/.cache/huggingface。不要指向 /root/autodl-tmp，那个目录没有读权限。
os.environ.setdefault("HF_HUB_OFFLINE", "1")

#MODEL_ID = "OpenSportsLab/OSL-loc-snbas-2023-e2e"
MODEL_ID = "OpenSportsLab/OSL-loc-snbas-2025-e2e"
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_VIDEO = PROJECT_ROOT / "test_data" / "1-2-6.mp4"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "opensportslib" / "test.step1.json"
# 官方配置 extract_fps=2、nms_window=2 帧，去重间隔是 1 秒。
_TRAINED_FPS = 2
_TRAINED_NMS_WINDOW = 2


def _parse_args() -> argparse.Namespace:
    """读取命令行：输入视频、可选的截取时长、结果 JSON 路径。"""
    parser = argparse.ArgumentParser(description="用 OpenSportsLib 定位足球视频中的动作，并按对照表保存中文类别。")
    parser.add_argument(
        "--video",
        type=Path,
        default=DEFAULT_VIDEO,
        help="输入视频路径，默认 test_data/test.mp4",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=None,
        help="只处理视频开头的若干秒，用来在 6GB 显存上做短流程验证",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="预测 JSON 输出路径，默认 outputs/opensportslib/test.step0.step1.json",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.90,
        help="只保留置信度不低于该值的事件，默认 0.8",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=2,
        help="监测抽帧率，默认 2，与权重训练时一致。30 帧/秒的视频在 2 时步长为 15",
    )
    return parser.parse_args()


def _trim_video(source: Path, seconds: float, destination: Path) -> Path:
    """用 PyAV 重编码开头片段。模型按 --fps 抽帧，窗口固定 100 帧。"""
    import av

    destination.parent.mkdir(parents=True, exist_ok=True)
    input_container = av.open(str(source))
    video_in = input_container.streams.video[0]
    output_container = av.open(str(destination), mode="w")
    video_out = output_container.add_stream("libx264", rate=video_in.average_rate)
    video_out.width = video_in.codec_context.width
    video_out.height = video_in.codec_context.height
    video_out.pix_fmt = "yuv420p"

    for frame in input_container.decode(video_in):
        if frame.time is not None and frame.time > seconds:
            break
        frame = frame.reformat(format="yuv420p")
        for packet in video_out.encode(frame):
            output_container.mux(packet)
    for packet in video_out.encode(None):
        output_container.mux(packet)

    output_container.close()
    input_container.close()
    return destination


def _local_checkpoint(repo_id: str) -> str:
    """从本地缓存取出权重。仓库名会触发在线登录，这里改用已经下载的 model.pt。"""
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=repo_id, filename="model.pt", local_files_only=True)


def _fill_opencv_dataloader(config) -> None:
    """本地权重不会走仓库适配。OpenCV 加载器仍要有 num_workers 和 pin_memory。"""
    splits = config._document["DATA"]["common"]["splits"]
    for split in splits.values():
        loader = split.get("dataloader") if isinstance(split, dict) else None
        if not isinstance(loader, dict):
            continue
        loader.setdefault("num_workers", 0)
        loader.setdefault("pin_memory", False)


def _nms_window(sample_fps: int) -> int:
    """把官方 1 秒去重间隔换算成当前抽帧率下的帧数。"""
    return max(1, round(sample_fps * _TRAINED_NMS_WINDOW / _TRAINED_FPS))


def _video_fps(path: Path) -> float:
    """读取容器帧率。DALI 的抽帧步长用的是配置里的 input_fps，不会自己读视频。"""
    import cv2

    capture = cv2.VideoCapture(str(path))
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    finally:
        capture.release()
    if fps <= 0:
        raise ValueError(f"读不到视频帧率: {path}")
    return fps


def _nominal_fps(fps: float) -> int:
    """29.97 这类分数帧率按标称整数算，和 OpenSportsLib 的 get_stride 一致。"""
    nominal = round(fps)
    if nominal > 0 and math.isclose(fps, nominal, rel_tol=0.0011):
        return nominal
    return int(fps)


def _sample_stride(video_fps: float, sample_fps: int) -> int:
    """源帧率除以目标帧率，向下取整。结果至少为 1。"""
    return max(1, int(_nominal_fps(video_fps) / sample_fps))


def _configure_model(output_dir: Path, sample_fps: int, video_fps: float):
    """按单卡、batch size 1 加载定位模型。有 CUDA 时用 GPU，否则退回 CPU。"""
    import torch
    from opensportslib.apis import Config, LocalizationModel

    if torch.cuda.is_available():
        device = "cuda"
        name = torch.cuda.get_device_name(0)
        total_mb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
        print(f"使用 GPU: {name}，显存 {total_mb:.0f} MiB")
    else:
        device = "cpu"
        print("CUDA 不可用，改为 CPU 推理")

    weights = _local_checkpoint(MODEL_ID)
    config = Config.from_pretrained(MODEL_ID)
    output_dir.mkdir(parents=True, exist_ok=True)
    config.update(
        runtime={"device": device, "output_dir": str(output_dir)},
        inference={"batch_size": 1},
        overrides={
            "SYSTEM.gpu.count": 1,
            "TRAIN.execution.multi_gpu": False,
            "DATA.inputs.video.sampling.input_fps": _nominal_fps(video_fps),
            "DATA.inputs.video.sampling.extract_fps": sample_fps,
            "DATA.common.splits.test.nms_window": _nms_window(sample_fps),
        },
    )
    _fill_opencv_dataloader(config)
    model = LocalizationModel(config=config, weights=weights)
    return model


def _keep_confident(predictions: dict, min_confidence: float) -> tuple[int, float | None]:
    """去掉置信度低于配置值的事件。没有分数字段的也不保留。同时返回被丢掉的最高分。"""
    removed = 0
    peak = None
    for sample in predictions.get("data", []):
        kept = []
        for event in sample.get("events", []):
            confidence = event.get("confidence")
            if confidence is None or float(confidence) < min_confidence:
                removed += 1
                if confidence is not None:
                    score = float(confidence)
                    peak = score if peak is None else max(peak, score)
                continue
            kept.append(event)
        sample["events"] = kept
    return removed, peak


def _map_labels(predictions: dict) -> None:
    """把事件类别换成对照表里的中文。不在表里的保持模型原样。"""
    for sample in predictions.get("data", []):
        for event in sample.get("events", []):
            action = LABEL_TO_ACTION.get(event.get("label"))
            if action:
                event["label"] = action


def _save_predictions(output_path: Path, predictions: dict) -> Path:
    """把预测写成带缩进的 JSON。库自带的保存函数会把整份结果挤成一行。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(predictions, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output_path


def _event_rows(predictions: dict) -> list[dict]:
    """从预测 JSON 里取出全部事件，并按发生时间排序，供终端摘要使用。"""
    rows = []
    for sample in predictions.get("data", []):
        for event in sample.get("events", []):
            rows.append(event)
    rows.sort(key=lambda event: (event.get("position_ms") or 0, event.get("frame") or 0))
    return rows


def main() -> int:
    """跑完一次定位：准备视频、推理、保存去重后的动作点并打印摘要。"""
    args = _parse_args()
    video = args.video.expanduser().resolve()
    if not video.is_file():
        print(f"找不到视频: {video}", file=sys.stderr)
        return 1
    if not 0.0 <= args.min_confidence <= 1.0:
        print("--min-confidence 必须在 0 到 1 之间", file=sys.stderr)
        return 1
    if args.fps <= 0:
        print("--fps 必须大于 0", file=sys.stderr)
        return 1
    source_fps = _nominal_fps(_video_fps(video))
    if args.fps > source_fps:
        print(f"--fps 不能高于视频帧率 {source_fps}", file=sys.stderr)
        return 1

    work_dir = PROJECT_ROOT / "outputs" / "opensportslib"
    source_video = video
    if args.max_seconds is not None:
        if args.max_seconds <= 0:
            print("--max-seconds 必须大于 0", file=sys.stderr)
            return 1
        source_video = _trim_video(
            video,
            args.max_seconds,
            work_dir / "clips" / f"{video.stem}-{int(args.max_seconds)}s.mp4",
        )
        print(f"已截取前 {args.max_seconds:g} 秒: {source_video}")

    video_fps = _video_fps(source_video)
    stride = _sample_stride(video_fps, args.fps)
    actual_fps = _nominal_fps(video_fps) / stride
    model = _configure_model(work_dir, args.fps, video_fps)
    print(
        f"开始推理: {source_video}，源视频 {_nominal_fps(video_fps)} 帧/秒，"
        f"步长 {stride}，实际监测 {actual_fps:g} 帧/秒"
    )
    predictions = model.infer(video_path=str(source_video), use_wandb=False)
    if not isinstance(predictions, dict):
        raise TypeError(f"推理结果不是字典: {type(predictions).__name__}")

    # infer() 返回高召回的逐帧候选，同一动作会重复出现。
    # 库还会把非极大值抑制之后的动作点写到 results_spotting_test.json，对外保存这一份。
    nms_files = sorted(work_dir.rglob("results_spotting_test.json"), key=lambda path: path.stat().st_mtime)
    if nms_files:
        predictions = json.loads(nms_files[-1].read_text(encoding="utf-8"))

    output_path = args.output
    removed, peak = _keep_confident(predictions, args.min_confidence)
    _map_labels(predictions)
    saved = _save_predictions(output_path, predictions)
    events = _event_rows(predictions)
    counts = {action: 0 for action in dict.fromkeys(LABEL_TO_ACTION.values())}
    for event in events:
        label = str(event.get("label", ""))
        counts[label] = counts.get(label, 0) + 1

    print(f"预测已写入: {saved}")
    if peak is None:
        print(f"置信度低于 {args.min_confidence:.0%} 已去掉 {removed} 条")
    else:
        print(f"置信度低于 {args.min_confidence:.0%} 已去掉 {removed} 条，最高 {peak:.0%}")
    print(f"事件总数: {len(events)}")
    for label, count in counts.items():
        print(f"  {label}: {count}")
    if not events:
        print("这个片段里没有超过阈值的动作")
    for event in events[:8]:
        seconds = (event.get("position_ms") or 0) / 1000
        print(
            f"  {seconds:7.2f}s  {event.get('label')}  "
            f"confidence={event.get('confidence')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
