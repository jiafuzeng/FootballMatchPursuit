"""本地足球视频的动作定位入口。

作用：把一段比赛视频交给 OpenSportsLib，找出其中的传球（PASS）和带球（DRIVE），
并把动作时间点写成 JSON。这是当前能在 GTX 1060 6GB 上跑通的推理流程，
不包含球员身份、接球人关联，也不覆盖进球、射门等其余事件。

流程：
1. 读取视频（默认 test_data/test.mp4，也可用 --video 指定）。
2. 可选地用 --max-seconds 只保留开头若干秒，便于短时验证。
3. 加载 Hugging Face 上的 OSL-loc-snbas-2023-e2e（RegNetY-008 + GRU）。
4. 按 2 帧/秒抽帧、每 100 帧一个窗口做定位。
5. 采用去重后的动作点，写入 outputs/opensportslib/<视频名>.predictions.json。

运行环境是 conda 环境 osl。这台机器的驱动只报告 CUDA 12.2，官方安装脚本
要求 CUDA 12.6，因此使用仍能在 Pascal 上执行的 PyTorch 2.5.1+cu121。
未安装 DALI，视频由 OpenCV 解码。6GB 显存下 batch size 固定为 1。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# 库在推理时会读取 RUN_ID；关闭 W&B，避免没有登录信息时卡住。
os.environ.setdefault("RUN_ID", "football-match-pursuit")
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("WANDB_DISABLED", "true")
# 随后加载的足球权重会覆盖骨干，不必再下载 ImageNet 预训练参数。
os.environ.setdefault("OSL_PRETRAINED_WEIGHTS", "0")

# 只识别两类球动作的公开定位模型。
#MODEL_ID = "OpenSportsLab/OSL-loc-snbas-2023-e2e"
MODEL_ID = "OpenSportsLab/OSL-loc-snbas-2025-e2e"
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_VIDEO = PROJECT_ROOT / "test_data" / "test.mp4"


def _parse_args() -> argparse.Namespace:
    """读取命令行：输入视频、可选的截取时长、结果 JSON 路径。"""
    parser = argparse.ArgumentParser(description="用 OpenSportsLib 定位足球视频中的传球和带球。")
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
        default=None,
        help="预测 JSON 输出路径",
    )
    return parser.parse_args()


def _trim_video(source: Path, seconds: float, destination: Path) -> Path:
    """用 PyAV 重编码开头片段。模型按 2fps 抽帧、窗口 100 帧，约 50 秒才满一个窗口。"""
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


def _configure_model(output_dir: Path):
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

    config = Config.from_pretrained(MODEL_ID)
    output_dir.mkdir(parents=True, exist_ok=True)
    config.update(
        runtime={"device": device, "output_dir": str(output_dir)},
        inference={"batch_size": 1},
        overrides={
            "SYSTEM.gpu.count": 1,
            "TRAIN.execution.multi_gpu": False,
        },
    )
    model = LocalizationModel(config=config, weights=MODEL_ID)
    return model


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

    model = _configure_model(work_dir)
    print(f"开始推理: {source_video}")
    predictions = model.infer(video_path=str(source_video), use_wandb=False)
    if not isinstance(predictions, dict):
        raise TypeError(f"推理结果不是字典: {type(predictions).__name__}")

    # infer() 返回高召回的逐帧候选，同一动作会重复出现。
    # 库还会把非极大值抑制之后的动作点写到 results_spotting_test.json，对外保存这一份。
    nms_files = sorted(work_dir.rglob("results_spotting_test.json"), key=lambda path: path.stat().st_mtime)
    if nms_files:
        predictions = json.loads(nms_files[-1].read_text(encoding="utf-8"))

    output_path = args.output or (work_dir / f"{source_video.stem}.predictions.json")
    saved = _save_predictions(output_path, predictions)
    events = _event_rows(predictions)
    counts: dict[str, int] = {}
    for event in events:
        label = str(event.get("label", "unknown"))
        counts[label] = counts.get(label, 0) + 1

    print(f"预测已写入: {saved}")
    print(f"事件总数: {len(events)}")
    if counts:
        for label, count in sorted(counts.items()):
            print(f"  {label}: {count}")
    else:
        print("这个片段里没有超过阈值的 PASS / DRIVE 事件")
    for event in events[:8]:
        seconds = (event.get("position_ms") or 0) / 1000
        print(
            f"  {seconds:7.2f}s  {event.get('label')}  "
            f"confidence={event.get('confidence')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
