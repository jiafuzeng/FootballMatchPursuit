"""用 Qwen3.8-27B 看一个足球片段，写出这一下的球队、动作和队员号。

输入是一个 clip 文件。按每秒 2 帧抽帧后再送进模型，避免全帧把脚下圆圈缩到认不出。
只输出一个 JSON：team、action、球衣号 player_code、脚下圆圈跟踪号 player_id，以及接球人的对应字段。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from label_map import LABEL_TO_ACTION

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_ID = "/root/autodl-tmp/models/Qwen3.8-27B"
# 全帧送入时像素预算会把每帧缩到约 288 高，脚下圆圈里的跟踪号认不出。
SAMPLE_FPS = 2.0
MAPPED_ACTIONS = tuple(dict.fromkeys(LABEL_TO_ACTION.values()))
DEFAULT_CLIP = Path(
    "/root/autodl-tmp/FootballMatchPursuit/outputs/opensportslib/clips/vl/005_传球_45000ms.mp4"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="用 Qwen3.8-27B 理解一个足球视频片段。")
    parser.add_argument(
        "clip",
        nargs="?",
        type=Path,
        default=DEFAULT_CLIP,
        help=f"一个视频片段，默认 {DEFAULT_CLIP}",
    )
    parser.add_argument("--output", type=Path, default=None, help="把模型原文写入这个文件；默认写到 outputs/qwen38_27b/")
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--thinking", action="store_true", help="打开模型自带的思考过程；默认直接作答")
    return parser.parse_args()


def _load_clip(path: Path):
    """读回完整片段。抽帧在送进模型时按 SAMPLE_FPS 做。"""
    import av
    import numpy as np

    container = av.open(str(path))
    stream = container.streams.video[0]
    fps = float(stream.average_rate) if stream.average_rate else 30.0
    frames = [frame.to_ndarray(format="rgb24") for frame in container.decode(stream)]
    container.close()
    if len(frames) < 2:
        raise RuntimeError(f"片段帧数不足: {path}")
    return np.stack(frames), fps


def _prompt() -> str:
    return (
        "看这段足球转播视频，它是这一个事件从开始到结束的过程。"
        "红队是 Team A，白队是 Team B，门将算自己一方。"
        "只输出一个 JSON 对象，不要数组，不要 markdown，不要解释。"
        "键只有 team, action, player_code, receiver_code, player_id, receiver_id。"
        f"action 只能是：{'、'.join(MAPPED_ACTIONS)}。"
        "team 只能是 A 或 B。"
        "player_id 和 receiver_id 是球员脚下彩色圆圈里的跟踪号，不是球衣上的号码。看不清写 unknown。"
        "player_code 和 receiver_code 是球衣上的号码。看不清写 unknown。"
    )


def _load_model(model_id: str):
    import torch
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

    # Qwen3.8-27B 的架构名仍是 Qwen3_5ForConditionalGeneration。
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor


def _ask(model, processor, clip, clip_fps: float, text: str, thinking: bool, max_new_tokens: int) -> str:
    import torch
    from transformers.video_utils import VideoMetadata

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": ""},
                {"type": "text", "text": text},
            ],
        }
    ]
    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=thinking,
    )
    metadata = VideoMetadata(
        total_num_frames=int(clip.shape[0]),
        fps=float(clip_fps),
        width=int(clip.shape[2]),
        height=int(clip.shape[1]),
        duration=float(clip.shape[0]) / clip_fps,
        frames_indices=list(range(int(clip.shape[0]))),
    )
    inputs = processor(
        text=[prompt],
        videos=[clip],
        do_sample_frames=True,
        fps=SAMPLE_FPS,
        video_metadata=metadata,
        return_tensors="pt",
    )
    inputs = {
        key: value.to(model.device)
        for key, value in inputs.items()
        if hasattr(value, "to")
    }
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    new_tokens = output[:, inputs["input_ids"].shape[1] :]
    return processor.batch_decode(new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]


def main() -> int:
    args = _parse_args()
    clip = args.clip.expanduser().resolve()
    if not clip.is_file():
        print(f"找不到片段: {clip}", file=sys.stderr)
        return 1
    max_new_tokens = args.max_new_tokens if args.max_new_tokens is not None else (8192 if args.thinking else 2048)
    print(f"片段 {clip.name}，模型 {args.model}，按每秒 {SAMPLE_FPS:g} 帧抽帧", flush=True)
    frames, clip_fps = _load_clip(clip)
    sampled = int(frames.shape[0] / clip_fps * SAMPLE_FPS)
    sampled = min(max(sampled, 4), 768, int(frames.shape[0]))
    print(
        f"片段 {frames.shape[0]} 帧、约 {frames.shape[0] / clip_fps:.0f} 秒，送入模型 {sampled} 帧",
        flush=True,
    )
    try:
        model, processor = _load_model(args.model)
    except ImportError as error:
        print(f"当前环境还不能加载 Qwen3.8-27B: {error}", file=sys.stderr)
        return 1
    answer = _ask(model, processor, frames, clip_fps, _prompt(), args.thinking, max_new_tokens)
    del frames
    output = args.output.expanduser().resolve() if args.output else PROJECT_ROOT / "outputs" / "qwen38_27b" / f"{clip.stem}.txt"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(answer.strip() + "\n", encoding="utf-8")
    print(answer.strip())
    print(f"\n已写入: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
