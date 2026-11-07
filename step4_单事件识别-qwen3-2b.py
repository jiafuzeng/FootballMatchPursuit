"""把一段视频和一段提示词交给本地 Qwen3-VL-2B，原样打印模型返回的文本。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DEFAULT_MODEL = "Qwen/Qwen3-VL-2B-Instruct"
PROJECT_ROOT = Path(__file__).resolve().parent
# 整段视频共用的像素上限。6GB 显卡装不下 21 帧原分辨率，这个上限大约相当于两帧 1080p。
VIDEO_MAX_PIXELS = 2 * 1920 * 1080
#DEFAULT_VIDEO = PROJECT_ROOT / "test_data" / "test_3min.mp4"
DEFAULT_VIDEO = "/home/gc/FootballMatchPursuit/outputs/opensportslib/clips/test_3min_boxed/001_长传_10000ms.mp4"
DEFAULT_PROMPT = """只识别控球人。控球人每换一次，输出一条。只输出 JSON 数组，不要解释，不要其他字段。

每条只有这三个键：

- player_code:控球人的球衣号。看不清写 unknown,不要瞎写。
- player_id:控球人的跟踪号。同一个人始终用同一个号，写成数字字符串。无法区分时写 unknown,不要瞎写。
- action_screen_time_period:这一帧画面上方比分牌里的比赛时钟，不是帧号换算出来的时间。时钟在比分正下方的蓝色底上，白色数字，只有分和秒，形如 15:26。

不要写动作、队名、接球人、帧号和坐标。

```json
[{"player_code": "26", "player_id": "6", "action_screen_time_period": "15:43"}]
```
"""


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="把视频和提示词交给本地 Qwen3-VL-2B，输出模型原文。")
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO, help="输入视频")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="提示词，默认只识别控球人的球衣号、跟踪号和比赛时钟")
    parser.add_argument("--output", type=Path, default=None, help="可选，把结果写入这个文件")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default=None, help="cuda 或 cpu，默认有显卡就用 cuda")
    return parser.parse_args()


def _read_clip(path: Path):
    import av
    import numpy as np

    container = av.open(str(path))
    stream = container.streams.video[0]
    rate = float(stream.average_rate or 30)
    frames = [frame.to_ndarray(format="rgb24") for frame in container.decode(stream)]
    container.close()
    if not frames:
        raise SystemExit(f"视频里没有画面: {path}")
    return np.stack(frames), rate


def _json_value(text: str):
    """从模型原文里取出 JSON 数组或对象。"""
    start = 0
    while True:
        bracket = min(
            (index for index in (text.find("[", start), text.find("{", start)) if index >= 0),
            default=-1,
        )
        if bracket < 0:
            return None
        depth = 0
        opener = text[bracket]
        closer = "]" if opener == "[" else "}"
        for end in range(bracket, len(text)):
            if text[end] == opener:
                depth += 1
            elif text[end] == closer:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[bracket : end + 1])
                    except json.JSONDecodeError:
                        break
        start = bracket + 1


def _events_of(value) -> list[dict]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if isinstance(value, dict):
        for key in ("events", "data"):
            if isinstance(value.get(key), list):
                return [item for item in value[key] if isinstance(item, dict)]
        if "player_code" in value or "action_screen_time_period" in value:
            return [value]
    return []


def _possession(event: dict) -> dict:
    """只留下控球人的球衣号、跟踪号和画面时钟。"""

    def text(value: object) -> str:
        if value is None:
            return "unknown"
        content = str(value).strip()
        return content or "unknown"

    return {
        "player_code": text(event.get("player_code")),
        "player_id": text(event.get("player_id")),
        "action_screen_time_period": text(event.get("action_screen_time_period")),
    }


def _summarize(events: list[dict]) -> list[dict]:
    """按比赛时钟排列，去掉同一控球人在同一时钟的重复。"""
    rows = [_possession(event) for event in events]
    rows.sort(key=lambda event: event["action_screen_time_period"])
    kept = []
    for event in rows:
        if any(
            item["player_id"] == event["player_id"] and item["action_screen_time_period"] == event["action_screen_time_period"]
            for item in kept
        ):
            continue
        kept.append(event)
    return kept


def _load_model(model_id: str, device: str):
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id,
        dtype=dtype,
        device_map=device,
        local_files_only=True,
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(model_id, local_files_only=True)
    return model, processor


def _ask(model, processor, video, fps: float, prompt: str) -> str:
    import torch
    from transformers.video_utils import VideoMetadata

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video"},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    metadata = VideoMetadata(
        total_num_frames=int(video.shape[0]),
        fps=fps,
        width=int(video.shape[2]),
        height=int(video.shape[1]),
        duration=video.shape[0] / fps,
        frames_indices=list(range(int(video.shape[0]))),
    )
    inputs = processor(
        text=[text],
        videos=[video],
        do_sample_frames=False,
        video_metadata=metadata,
        size={"shortest_edge": 128 * 128, "longest_edge": VIDEO_MAX_PIXELS},
        return_tensors="pt",
    )
    grid = inputs.get("video_grid_thw")
    if grid is not None:
        tokens = int(inputs["input_ids"].shape[1])
        print(f"画面网格 {grid.tolist()}，输入 {tokens} token", file=sys.stderr)
    inputs = {key: value.to(model.device) if hasattr(value, "to") else value for key, value in inputs.items()}
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=256, do_sample=False)
    new_tokens = output[:, inputs["input_ids"].shape[1] :]
    return processor.batch_decode(new_tokens, skip_special_tokens=True)[0]


def main() -> None:
    args = _parse_args()
    video = args.video.expanduser().resolve()
    if not video.is_file():
        raise SystemExit(f"找不到视频: {video}")
    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    clip, fps = _read_clip(video)
    print(f"本地 {args.model}，设备 {device}，{video.name} {clip.shape[0]} 帧", file=sys.stderr)
    model, processor = _load_model(args.model, device)
    text = _ask(model, processor, clip, fps, args.prompt)
    print(text)
    events = _summarize(_events_of(_json_value(text)))
    result = {
        "model": args.model,
        "video": str(video),
        "events": events,
    }
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    output = args.output or (PROJECT_ROOT / "outputs" / "opensportslib" / f"{video.stem}.qwen3-2b.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text + "\n", encoding="utf-8")
    print(f"已写入: {output}，合计 {len(events)} 条", file=sys.stderr)


if __name__ == "__main__":
    main()
