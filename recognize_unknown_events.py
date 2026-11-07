"""把每个定位事件的完整片段交给视觉语言模型，再写成 test.json 格式。

定位结果只有发生时刻，没有起止。每条事件按自己需要的范围切，不避让相邻片段。
普通事件在发生时刻前后各留 2 秒。传球、长传、传中、头球从发球前保留到下一次触球，
让发球和接球都在同一段里，飞行太远时不再把后面的空白算进去。片段保留原视频的帧率和分辨率。
Qwen3.8-27B 只看这一下的过程：球队、动作，以及球从哪个队员到哪个队员。
送进模型前按每秒 2 帧抽取，一段最多 12 帧，画面保持接近原分辨率，脚下圆圈里的跟踪号才看得清。
队员号读脚下彩色圆圈里的跟踪号，不读球衣。不读比赛时钟，也不写画面坐标。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from label_map import LABEL_TO_ACTION

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_PREDICTIONS = PROJECT_ROOT / "outputs" / "opensportslib" / "test_3min.step1.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "opensportslib" / "test_3min.events.json"
MODEL_ID = "/root/autodl-tmp/models/Qwen3.8-27B"
CLIP_PAD = 2.0
# 传球一类收到下一次触球。下一次触球太远时，不再把后面的空白算进这一脚。
PASS_ACTIONS = frozenset({"传球", "长传", "传中", "头球"})
RECEPTION_TAIL = 0.5
MAX_PASS_FLIGHT = 15.0
# 全帧或抽完仍超过 12 帧时，每帧会被缩到约 700 高，脚下圆圈认不出。12 帧能保持 1088 高。
SAMPLE_FPS = 2.0
MAX_MODEL_FRAMES = 12
MAPPED_ACTIONS = tuple(dict.fromkeys(LABEL_TO_ACTION.values()))


def _video_meta(video: Path) -> tuple[float, float | None]:
    """一次打开视频，读出帧率和时长。"""
    import av

    container = av.open(str(video))
    stream = container.streams.video[0]
    fps = float(stream.average_rate) if stream.average_rate else 0.0
    duration = float(stream.duration * stream.time_base) if stream.duration is not None else None
    container.close()
    if fps <= 1.0:
        raise RuntimeError(f"无法读取视频帧率: {video}")
    return fps, duration


def _prediction_video(payload: dict) -> Path | None:
    for sample in payload.get("data", []):
        for item in sample.get("inputs", []):
            if item.get("type") == "video" and item.get("path"):
                return Path(item["path"])
    return None


def _name_cn(action: str, team: str | None) -> str:
    if team is None:
        return action
    if action == "进球":
        return f"Team {team}Goal"
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


def _bound(start: float, end: float, duration: float | None) -> tuple[float, float]:
    """把片段收在视频里面。起止挤在一起时至少留半秒。"""
    start = max(0.0, start)
    if duration is not None:
        end = min(end, max(duration - 0.05, 0.0))
    if end <= start:
        end = start + 0.5
        if duration is not None:
            end = min(end, duration)
            start = max(0.0, end - 0.5)
    return start, end


def _windows(events: list[dict], duration: float | None) -> list[tuple[float, float]]:
    """按当前事件自己的起止来切，不避让相邻片段。

    普通事件前后各 2 秒，够看清这一下，也不把更远的画面算进去。
    传球一类从发球前留到下一次触球之后 0.5 秒，飞行最长 15 秒。
    """
    times = [int(event["position_ms"]) / 1000.0 for event in events]
    labels = [str(event.get("label", "")) for event in events]
    windows = []
    for index, center in enumerate(times):
        start = center - CLIP_PAD
        if labels[index] in PASS_ACTIONS and index + 1 < len(times):
            end = min(center + MAX_PASS_FLIGHT, times[index + 1] + RECEPTION_TAIL)
        else:
            end = center + CLIP_PAD
        windows.append(_bound(start, end, duration))
    return windows


def _clip_file(directory: Path, index: int, event: dict) -> Path:
    label = str(event.get("label", "event"))
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
    """按原帧率、原分辨率写出每个事件的完整片段。相邻片段可以重叠。"""
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


def _model_frame_count(n_frames: int, fps: float) -> int:
    """按每秒 SAMPLE_FPS 抽，最多 MAX_MODEL_FRAMES 帧。"""
    wanted = int(n_frames / fps * SAMPLE_FPS) if fps else n_frames
    return min(max(wanted, 4), MAX_MODEL_FRAMES, n_frames)


def _load_clip(path: Path):
    """读回完整片段。抽帧在送进模型时做，这里保留原帧。"""
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


def _load_model():
    import torch
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

    # Qwen3.8-27B 的架构名仍是 Qwen3_5ForConditionalGeneration。
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        device_map="auto",
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
    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
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
        fps=None,
        num_frames=_model_frame_count(int(clip.shape[0]), clip_fps),
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
    objects = [item for item in _json_objects(text) if item.get("action") in MAPPED_ACTIONS]
    if not objects:
        return {}
    # 三帧里中间一帧最接近事件发生的时刻。
    return objects[len(objects) // 2]


def _jersey(value) -> str:
    """球衣号只收 1 到 99，并去掉前导零。"""
    text = str(value).strip()
    if text.isdigit() and 1 <= len(text) <= 2 and 0 < int(text) <= 99:
        return str(int(text))
    return "unknown"


def _track_id(value) -> str:
    """脚下圆圈里的跟踪号。去掉前导零，认不出就写 unknown。"""
    text = str(value).strip()
    if text.isdigit() and 1 <= len(text) <= 4 and int(text) > 0:
        return str(int(text))
    return "unknown"


def _team(value) -> str | None:
    text = str(value).strip().upper()
    if text in {"A", "TEAM A", "RED", "红", "红队"}:
        return "A"
    if text in {"B", "TEAM B", "WHITE", "白", "白队"}:
        return "B"
    return None


def _record(answer: dict) -> dict | None:
    action = answer.get("action")
    if not action:
        return None
    team = _team(answer.get("team"))
    return {
        "team": team,
        "player_code": _jersey(answer.get("player_code")),
        "player_id": _track_id(answer.get("player_id")),
        "name_cn": _name_cn(action, team),
        "receiver_code": _jersey(answer.get("receiver_code")),
        "receiver_id": _track_id(answer.get("receiver_id")),
    }


def _dedupe(records: list[dict]) -> list[dict]:
    """相邻两条内容完全相同，只保留先出现的一条。"""
    kept = []
    for record in records:
        if kept and record == kept[-1]:
            continue
        kept.append(record)
    return kept


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
    fps, duration = _video_meta(video)
    if args.fps is not None:
        fps = args.fps
    events = _events(payload)
    print(f"待判断 {len(events)} 条，帧率 {fps:g}")
    clip_dir = args.output.expanduser().resolve().parent / "clips" / "vl"
    specs = _write_clips(video, events, _windows(events, duration), clip_dir)
    print(f"已写出 {len(specs)} 段完整片段: {clip_dir}", flush=True)
    model, processor = _load_model()
    records = []
    for index, spec in enumerate(specs, start=1):
        event = spec["event"]
        center = int(event["position_ms"]) / 1000.0
        offset = center - spec["start"]
        clip, clip_fps = _load_clip(spec["path"])
        sent = _model_frame_count(int(clip.shape[0]), clip_fps)
        raw = _ask(model, processor, clip, clip_fps, _prompt(str(event.get("label", "")), offset))
        del clip
        answer = _parse_answer(raw)
        record = _record(answer)
        if record is None:
            print(f"{index}/{len(specs)} {event.get('label')} 未能辨认: {raw.strip()}", flush=True)
            continue
        records.append(record)
        print(
            f"{index}/{len(specs)} {event.get('label')} "
            f"[{spec['start']:.2f}-{spec['end']:.2f}s 送入 {sent} 帧] "
            f"-> {record['name_cn']} {record['player_code']}->{record['receiver_code']} {raw.strip()}",
            flush=True,
        )
    output = args.output.expanduser().resolve()
    records = _dedupe(records)
    _write(output, records)
    print(f"写入 {len(records)} 条事件: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
