"""用 Qwen3.8-27B 直接看整段比赛，写出事件说明书要求的 JSON。

整段大约 10 分钟，一次放不进模型。按窗口每秒取 15 帧覆盖全场，再合成一份记录。
默认输入 test_data/test.mp4，默认输出 outputs/qwen38_27b/test.events.json。
跟踪号是球员脚下彩色圆圈里的数字，player_id 和 receiver_id 从画面读取。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SPEC_PATH = PROJECT_ROOT / "事件说明书.md"
MODEL_ID = "/root/autodl-tmp/models/Qwen3.8-27B"
DEFAULT_VIDEO = Path("/root/autodl-tmp/FootballMatchPursuit/test_data/test_3min.mp4")
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "qwen38_27b" / "test.events.json"
# 16 秒 × 15 帧/秒 = 240 帧，接近模型一次能看完的上限。
# 相邻窗口重叠 2 秒，避免一脚触球正好被切在边界上。
SAMPLE_FPS = 15.0
WINDOW_SECONDS = 16.0
WINDOW_OVERLAP = 2.0
# 事件说明书里的动作。角球不带队名，进球和扑救的写法在 _name_cn 里单独处理。
ACTIONS = (
    "传球",
    "停球（带球）",
    "长传",
    "抢断",
    "射门",
    "助攻",
    "界外手抛球",
    "进球",
    "扑救",
    "角球",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="用 Qwen3.8-27B 直接把整段比赛写成事件 JSON。")
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO, help=f"整段比赛，默认 {DEFAULT_VIDEO}")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help=f"事件 JSON，默认 {DEFAULT_OUTPUT}")
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--thinking", action="store_true", help="打开模型自带的思考过程；默认直接作答")
    return parser.parse_args()


def _video_meta(video: Path) -> tuple[float, float, int, int]:
    """读取源视频帧率、时长和分辨率。frame 按这个帧率计算，不按送进模型的 15 帧/秒。"""
    import av

    container = av.open(str(video))
    stream = container.streams.video[0]
    fps = float(stream.average_rate) if stream.average_rate else 0.0
    duration = float(stream.duration * stream.time_base) if stream.duration is not None else 0.0
    width = int(stream.codec_context.width)
    height = int(stream.codec_context.height)
    container.close()
    if fps <= 1.0 or duration <= 0.0:
        raise RuntimeError(f"无法读取视频: {video}")
    return fps, duration, width, height


def _windows(duration: float) -> list[tuple[float, float]]:
    """切成固定长度的窗口。下一步起点回退重叠时长，所以边界上的画面会被看两次。"""
    if duration <= WINDOW_SECONDS:
        return [(0.0, duration)]
    spans = []
    start = 0.0
    while start < duration - 0.5:
        end = min(start + WINDOW_SECONDS, duration)
        spans.append((start, end))
        if end >= duration:
            break
        start += WINDOW_SECONDS - WINDOW_OVERLAP
    return spans


def _load_window(video: Path, start: float, end: float):
    """只解码这一段，并按每秒 15 帧取样。"""
    import av
    import numpy as np

    container = av.open(str(video))
    stream = container.streams.video[0]
    # seek 会落到窗口起点之前的关键帧，后面再按时间丢掉窗口外的帧。
    if start > 0:
        container.seek(int(start / stream.time_base), stream=stream, backward=True)
    interval = 1.0 / SAMPLE_FPS
    next_time = start
    frames = []
    for frame in container.decode(stream):
        timestamp = frame.time
        if timestamp is None or timestamp < start:
            continue
        if timestamp >= end:
            break
        # 1 毫秒余量用来吸收帧时间戳的浮点误差。
        if timestamp + 1e-3 >= next_time:
            frames.append(frame.to_ndarray(format="rgb24"))
            next_time += interval
    container.close()
    if len(frames) < 2:
        raise RuntimeError(f"窗口帧数不足: {start:.1f}-{end:.1f}s")
    return np.stack(frames), SAMPLE_FPS


def _prompt(spec: str, start: float, end: float) -> str:
    """告诉模型这段在整场中的位置。time_sec 从窗口开头算，方便换回全场帧号。"""
    return (
        "下面是整段比赛中的一个窗口，不是单独剪出来的一个动作。\n"
        "红队是 A，白队是 B，门将算自己一方。team 只能写 A 或 B，不要写红、白、RED、WHITE、Team A 或其他说法。\n"
        "跟踪号是球员脚下彩色圆圈里的数字，不是球衣上的号码。player_id 读行动人脚下的跟踪号，receiver_id 读关联人脚下的跟踪号。\n"
        "只输出一个 JSON 数组，不要 markdown，不要解释。没有触球就输出 []。\n"
        "数组里一次触球一个对象，键为 time_sec, team, action, player_code, player_id, receiver_code, receiver_id, x, y, action_screen_time_period。\n"
        "time_sec 是相对这一窗口开头的秒数。\n"
        f"action 只能是：{'、'.join(ACTIONS)}。\n"
        "球衣号看不清写 unknown。没有关联人时 receiver_code 和 receiver_id 都写 unknown。跟踪号看不清也写 unknown。\n"
        "传球和长传的关联人是接球人。抢断的行动人是抢下球的人，关联人是被抢的人。"
        "射门只写射门的人。助攻的关联人是进球的人。扑救的行动人是门将，关联人是射门的人。"
        "界外手抛球只写抛球的人。角球不需要关联人。\n"
        "x 和 y 是行动人人物检测框中心。\n"
        "action_screen_time_period 只读这一下触球那一帧左上角的比赛时钟，固定 MM:SS，例如 15:46。画面没有百分秒，不要加小数。"
        "看画面上的数字，不要用视频播放进度。\n\n"
        "说明书原文：\n"
        f"{spec.strip()}\n"
    )


def _load_model(model_id: str):
    """Qwen3.8-27B 的架构名仍是 Qwen3_5ForConditionalGeneration。"""
    import torch
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor


def _ask(model, processor, clip, clip_fps: float, text: str, thinking: bool, max_new_tokens: int) -> str:
    """把已经抽好的帧送进模型。do_sample_frames 关闭，避免处理器再抽一次。"""
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
        do_sample_frames=False,  # 帧已按 15 帧/秒抽好
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
    # generate 的输出包含提示词，只保留新生成的部分。
    new_tokens = output[:, inputs["input_ids"].shape[1] :]
    return processor.batch_decode(new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]


def _json_array(text: str) -> list:
    """取出回答里第一段括号配平的 JSON 数组。模型有时会在数组外面加说明。"""
    start = text.find("[")
    if start < 0:
        return []
    depth = 0
    for index in range(start, len(text)):
        if text[index] == "[":
            depth += 1
        elif text[index] == "]":
            depth -= 1
            if depth == 0:
                try:
                    value = json.loads(text[start : index + 1])
                except json.JSONDecodeError:
                    return []
                return value if isinstance(value, list) else []
    return []


def _track_id(value) -> str:
    """跟踪号是脚下圆圈里的数字，去掉前导零。认不出就写 unknown。"""
    text = str(value).strip()
    if text.isdigit() and 1 <= len(text) <= 4 and int(text) > 0:
        return str(int(text))
    return "unknown"


def _jersey(value) -> str:
    """球衣号只收 1 到 99，并去掉前导零。其余一律 unknown。"""
    text = str(value).strip()
    if text.isdigit() and 1 <= len(text) <= 2 and 0 < int(text) <= 99:
        return str(int(text))
    return "unknown"


def _match_clock(value) -> str:
    """把左上角比赛时钟整理成 MM:SS。画面没有百分秒，小数部分丢掉。读不到时不改用视频进度。"""
    text = str(value).strip().replace("：", ":").replace("．", ".").replace("。", ".")
    matched = re.fullmatch(r"(\d{1,2}):(\d{2})(?:\.\d{1,2})?", text)
    if matched is None:
        return "unknown"
    minutes = int(matched.group(1))
    seconds = int(matched.group(2))
    if minutes > 99 or seconds > 59:
        return "unknown"
    return f"{minutes:02d}:{seconds:02d}"


def _name_cn(action: str, team: str | None) -> str:
    """按说明书拼 name_cn。角球不带队名，进球和扑救不用中文动作名。"""
    if action == "角球" or team is None:
        return action
    if action == "进球":
        return f"Team {team}Goal"
    if action == "扑救":
        return f"Team {team}_GKSave"
    return f"Team {team}{action}"


def _records(items: list, window_start: float, fps: float) -> list[dict]:
    """把模型对象写成说明书字段。帧号用源视频帧率，从窗口起点加上 time_sec。"""
    records = []
    for item in items:
        if not isinstance(item, dict):
            continue
        action = str(item.get("action", "")).strip()
        if action not in ACTIONS:
            continue
        try:
            offset = float(item.get("time_sec"))
        except (TypeError, ValueError):
            continue
        if offset < 0:
            continue
        team = str(item.get("team", "")).strip().upper()
        if team not in {"A", "B"}:
            team = None
        records.append(
            {
                "frame": int(round((window_start + offset) * fps)),
                "player_code": _jersey(item.get("player_code")),
                "player_id": _track_id(item.get("player_id")),
                "name_cn": _name_cn(action, team),
                "receiver_code": _jersey(item.get("receiver_code")),
                "receiver_id": _track_id(item.get("receiver_id")),
                "player_screen_xy": [item.get("x"), item.get("y")],
                "action_screen_time_period": _match_clock(item.get("action_screen_time_period")),
            }
        )
    return records


def _dedupe(records: list[dict], fps: float) -> list[dict]:
    """去掉重叠窗口里同一动作的重复记录。只和最近几条比较，记录已按帧号排好。"""
    records.sort(key=lambda item: (item["frame"], item["action_screen_time_period"], item["name_cn"]))
    span = int(round(WINDOW_OVERLAP * fps))
    kept = []
    for record in records:
        duplicate = any(
            abs(record["frame"] - previous["frame"]) <= span and record["name_cn"] == previous["name_cn"]
            for previous in kept[-12:]
        )
        if not duplicate:
            kept.append(record)
    return kept


def main() -> int:
    """模型只加载一次，逐个窗口推理，最后去重并写成一份 JSON。"""
    args = _parse_args()
    video = args.video.expanduser().resolve()
    if not video.is_file():
        print(f"找不到视频: {video}", file=sys.stderr)
        return 1
    if not SPEC_PATH.is_file():
        print(f"找不到事件说明书: {SPEC_PATH}", file=sys.stderr)
        return 1
    spec = SPEC_PATH.read_text(encoding="utf-8")
    fps, duration, _, _ = _video_meta(video)
    spans = _windows(duration)
    print(
        f"视频 {video.name}，{duration:.1f} 秒，{fps:g} 帧/秒，"
        f"按 {WINDOW_SECONDS:g} 秒窗口、每秒 {SAMPLE_FPS:g} 帧覆盖，共 {len(spans)} 段",
        flush=True,
    )
    try:
        model, processor = _load_model(args.model)
    except ImportError as error:
        print(f"当前环境还不能加载 Qwen3.8-27B: {error}", file=sys.stderr)
        return 1

    records = []
    for index, (start, end) in enumerate(spans, start=1):
        print(f"{index}/{len(spans)} 窗口 {start:.1f}-{end:.1f}s", flush=True)
        clip, clip_fps = _load_window(video, start, end)
        raw = _ask(
            model,
            processor,
            clip,
            clip_fps,
            _prompt(spec, start, end),
            args.thinking,
            args.max_new_tokens,
        )
        del clip
        found = _records(_json_array(raw), start, fps)
        print(f"  解析到 {len(found)} 条", flush=True)
        records.extend(found)

    records = _dedupe(records, fps)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"写入 {len(records)} 条事件: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
