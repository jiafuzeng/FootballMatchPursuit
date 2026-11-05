"""把每个定位事件的完整片段交给视觉语言模型，再写成 test.json 格式。

定位结果只有发生时刻，没有起止。每个事件的片段从与上一个事件的中点
取到与下一个事件的中点，单侧最多 2 秒，并保留原视频的帧率和分辨率。
Qwen3-VL-2B 直接看这段视频，在事件说明书的动作里选择，并读出球队、球衣号和画面位置。
跟踪号在转播画面里看不见，仍写 unknown。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_PREDICTIONS = PROJECT_ROOT / "outputs" / "opensportslib" / "test.predictions.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "opensportslib" / "test.events.json"
MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
WIDTH = 1920
HEIGHT = 1080
CLIP_PAD = 2.0
MODEL_FPS = 2
# 片段文件保持原分辨率。模型按整段时间抽帧，单次画面像素封顶，避免 6GB 显存装不下。
VIDEO_SHORTEST_EDGE = 4096
VIDEO_LONGEST_EDGE = 32 * 32 * 768
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
LABEL_HINT = {
    "HEADER": "头球触球，可能是传球、射门或争顶",
    "OUT": "球已经或即将出界",
    "CROSS": "向禁区传中",
    "FREE KICK": "定位球，只有明确发角球时才选角球",
    "BALL PLAYER BLOCK": "球员挡住球或抢断",
}


def _video_fps(path: Path) -> float:
    import cv2

    capture = cv2.VideoCapture(str(path))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    capture.release()
    if fps <= 1.0:
        raise RuntimeError(f"无法读取视频帧率: {path}")
    return fps


def _prediction_video(payload: dict) -> Path | None:
    for sample in payload.get("data", []):
        for item in sample.get("inputs", []):
            if item.get("type") == "video" and item.get("path"):
                return Path(item["path"])
    return None


def _screen_time(position_ms: int) -> str:
    centiseconds = int(round(position_ms / 10.0))
    minutes, remainder = divmod(centiseconds, 6000)
    seconds, centi = divmod(remainder, 100)
    return f"{minutes:02d}:{seconds:02d}.{centi:02d}"


def _frame(position_ms: int, fps: float) -> int:
    return int(round(position_ms / 1000.0 * fps))


def _name_cn(action: str, team: str | None) -> str:
    if action == "角球":
        return "角球"
    if team is None:
        return action
    if action == "进球":
        return f"Team {team}Goal"
    if action == "扑救":
        return f"Team {team}_GKSave"
    return f"Team {team}{action}"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="把每个事件的完整片段交给视觉语言模型重新判断。")
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--video", type=Path, default=None)
    parser.add_argument("--fps", type=float, default=None)
    return parser.parse_args()


def _events(payload: dict) -> list[dict]:
    events = []
    for sample in payload.get("data", []):
        events.extend(sample.get("events", []))
    events.sort(key=lambda event: int(event["position_ms"]))
    return events


def _windows(events: list[dict], duration: float | None) -> list[tuple[float, float]]:
    """每个事件占到相邻事件的中点；离得远时单侧最多保留 2 秒。"""
    times = [int(event["position_ms"]) / 1000.0 for event in events]
    windows = []
    for index, center in enumerate(times):
        previous_gap = center - times[index - 1] if index else CLIP_PAD * 2
        next_gap = times[index + 1] - center if index + 1 < len(times) else CLIP_PAD * 2
        start = max(0.0, center - min(CLIP_PAD, previous_gap / 2.0))
        end = center + min(CLIP_PAD, next_gap / 2.0)
        if duration is not None:
            end = min(end, max(duration - 0.05, 0.0))
        if end <= start:
            end = min(start + 0.5, duration if duration is not None else start + 0.5)
        windows.append((start, end))
    return windows


def _video_duration(video: Path) -> float | None:
    import av

    container = av.open(str(video))
    stream = container.streams.video[0]
    duration = float(stream.duration * stream.time_base) if stream.duration is not None else None
    container.close()
    return duration


def _clip_file(directory: Path, index: int, event: dict) -> Path:
    label = str(event.get("label", "event")).replace(" ", "_")
    return directory / f"{index:03d}_{label}_{int(event['position_ms'])}ms.mp4"


class _ClipWriter:
    def __init__(self, path: Path, width: int, height: int, rate: int, end: float):
        import av

        self.end = end
        self.count = 0
        self.container = av.open(str(path), mode="w")
        self.stream = self.container.add_stream("libx264", rate=rate)
        self.stream.width = width
        self.stream.height = height
        self.stream.pix_fmt = "yuv420p"
        self.stream.options = {"preset": "ultrafast", "crf": "28"}

    def write(self, frame) -> None:
        import av

        array = frame.to_ndarray(format="yuv420p")
        out = av.VideoFrame.from_ndarray(array, format="yuv420p")
        out.pts = self.count
        self.count += 1
        for packet in self.stream.encode(out):
            if packet.dts is None:
                packet.dts = packet.pts
            self.container.mux(packet)

    def close(self) -> None:
        for packet in self.stream.encode():
            if packet.dts is None:
                packet.dts = packet.pts
            self.container.mux(packet)
        self.container.close()


def _write_clips(video: Path, events: list[dict], windows: list[tuple[float, float]], directory: Path) -> list[dict]:
    """按原帧率、原分辨率写出每个事件的完整片段。"""
    import av

    directory.mkdir(parents=True, exist_ok=True)
    for old in directory.glob("*.mp4"):
        old.unlink()
    specs = []
    for index, (event, (start, end)) in enumerate(zip(events, windows), start=1):
        path = _clip_file(directory, index, event)
        specs.append({"event": event, "start": start, "end": end, "path": path})

    container = av.open(str(video))
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"
    rate = round(float(stream.average_rate)) if stream.average_rate else 30
    width = stream.codec_context.width
    height = stream.codec_context.height
    if specs and specs[0]["start"] > 0:
        container.seek(int(specs[0]["start"] / stream.time_base), stream=stream, backward=True)

    writers = []
    next_clip = 0
    for frame in container.decode(stream):
        timestamp = frame.time
        if timestamp is None:
            continue
        while next_clip < len(specs) and specs[next_clip]["start"] <= timestamp:
            spec = specs[next_clip]
            next_clip += 1
            if timestamp <= spec["end"]:
                writers.append((_ClipWriter(spec["path"], width, height, rate, spec["end"]), spec))
        still_open = []
        for writer, spec in writers:
            if timestamp > writer.end:
                writer.close()
                spec["frames"] = writer.count
            else:
                writer.write(frame)
                still_open.append((writer, spec))
        writers = still_open
        if next_clip >= len(specs) and not writers:
            break
    for writer, spec in writers:
        writer.close()
        spec["frames"] = writer.count
    container.close()
    for spec in specs:
        spec.setdefault("frames", 0)
        if spec["frames"] < 2:
            raise RuntimeError(f"片段帧数不足: {spec['path'].name}")
    return specs


def _load_clip(path: Path):
    """读回完整片段。模型自己按时间抽帧，这里不提前抽稀。"""
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


def _prompt(label: str, offset: float) -> str:
    hint = LABEL_HINT.get(label, "以画面为准")
    return (
        "看这段足球转播视频，它是这一个事件从开始到结束的完整片段。"
        "红队是 Team A，白队是 Team B，门将算自己一方。"
        "只输出一个 JSON 对象，不要数组，不要 markdown，不要解释。"
        "键为 team, action, player_code, receiver_code, x, y。"
        f"action 只能是：{'、'.join(ACTIONS)}。"
        "team 只能是 A 或 B。球衣号看不清写 unknown，否则写 1 到 2 位数字且不要前导零。"
        "没有关联人时 receiver_code 写 unknown。"
        "x 和 y 是行动人身体中心，相对画面宽高，范围 0 到 1，x 向右，y 向下。"
        f"事件发生在片段里的第 {offset:.1f} 秒。定位模型把这一刻标成 {label}（{hint}）。"
        "画面不一致时以画面为准。"
    )


def _load_model():
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        dtype=torch.float16,
        device_map="cuda",
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    return model, processor


def _ask(model, processor, clip, clip_fps: float, text: str) -> str:
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
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
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
        fps=MODEL_FPS,
        size={"shortest_edge": VIDEO_SHORTEST_EDGE, "longest_edge": VIDEO_LONGEST_EDGE},
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
            max_new_tokens=160,
            repetition_penalty=1.1,
            do_sample=False,
        )
    new_tokens = output[:, inputs["input_ids"].shape[1] :]
    return processor.batch_decode(new_tokens, skip_special_tokens=True)[0]


def _json_objects(text: str) -> list[dict]:
    """取出回答里每一个完整的 JSON 对象。模型有时会按三帧各写一条。"""
    found = []
    index = 0
    while index < len(text):
        if text[index] != "{":
            index += 1
            continue
        depth = 0
        for end in range(index, len(text)):
            if text[end] == "{":
                depth += 1
            elif text[end] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        value = json.loads(text[index : end + 1])
                    except json.JSONDecodeError:
                        value = None
                    if isinstance(value, dict):
                        found.append(value)
                    index = end + 1
                    break
        else:
            break
    return found


def _parse_answer(text: str) -> dict:
    objects = [item for item in _json_objects(text) if item.get("action") in ACTIONS]
    if not objects:
        return {}
    # 三帧里中间一帧最接近事件发生的时刻。
    return objects[len(objects) // 2]


def _jersey(value) -> str:
    text = str(value).strip()
    if text.isdigit() and 1 <= len(text) <= 2 and 0 < int(text) <= 99:
        return str(int(text))
    return "unknown"


def _team(value) -> str | None:
    text = str(value).strip().upper()
    if text in {"A", "TEAM A", "RED", "红", "红队"}:
        return "A"
    if text in {"B", "TEAM B", "WHITE", "白", "白队"}:
        return "B"
    return None


def _coordinate(value, limit: int) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if number > 1.0:
        number = number / 100.0 if number <= 100.0 else number / limit
    if not 0.0 <= number <= 1.0:
        return 0.0
    return number


def _record(event: dict, answer: dict, fps: float) -> dict | None:
    action = answer.get("action")
    if action not in ACTIONS:
        return None
    team = _team(answer.get("team"))
    x = _coordinate(answer.get("x"), WIDTH)
    y = _coordinate(answer.get("y"), HEIGHT)
    screen_xy = [0.0, 0.0] if x == 0.0 and y == 0.0 else [round(x * WIDTH, 2), round(y * HEIGHT, 2)]
    position_ms = int(event["position_ms"])
    return {
        "frame": _frame(position_ms, fps),
        "player_code": _jersey(answer.get("player_code")),
        "player_id": "unknown",
        "name_cn": _name_cn(action, team),
        "receiver_code": _jersey(answer.get("receiver_code")),
        "receiver_id": "unknown",
        "player_screen_xy": screen_xy,
        "action_screen_time_period": _screen_time(position_ms),
    }


def _write(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    args = _parse_args()
    predictions = args.predictions.expanduser().resolve()
    if not predictions.is_file():
        print(f"找不到预测文件: {predictions}", file=sys.stderr)
        return 1
    payload = json.loads(predictions.read_text(encoding="utf-8"))
    video = args.video.expanduser().resolve() if args.video else _prediction_video(payload)
    if video is None or not video.is_file():
        print("找不到原视频", file=sys.stderr)
        return 1
    fps = args.fps if args.fps is not None else _video_fps(video)
    events = _events(payload)
    print(f"待判断 {len(events)} 条，帧率 {fps:g}")
    clip_dir = args.output.expanduser().resolve().parent / "clips" / "vl"
    duration = _video_duration(video)
    specs = _write_clips(video, events, _windows(events, duration), clip_dir)
    print(f"已写出 {len(specs)} 段完整片段: {clip_dir}", flush=True)
    model, processor = _load_model()
    records = []
    for index, spec in enumerate(specs, start=1):
        event = spec["event"]
        center = int(event["position_ms"]) / 1000.0
        offset = center - spec["start"]
        clip, clip_fps = _load_clip(spec["path"])
        raw = _ask(model, processor, clip, clip_fps, _prompt(str(event.get("label", "")), offset))
        del clip
        answer = _parse_answer(raw)
        record = _record(event, answer, fps)
        if record is None:
            print(f"{index}/{len(specs)} {event.get('label')} 未能辨认: {raw.strip()}", flush=True)
            continue
        records.append(record)
        print(
            f"{index}/{len(specs)} {event.get('label')} {record['action_screen_time_period']} "
            f"[{spec['start']:.2f}-{spec['end']:.2f}s {spec['frames']}帧] -> {record['name_cn']} {raw.strip()}",
            flush=True,
        )
    records.sort(key=lambda item: (item["frame"], item["action_screen_time_period"], item["name_cn"]))
    output = args.output.expanduser().resolve()
    _write(output, records)
    print(f"写入 {len(records)} 条事件: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
