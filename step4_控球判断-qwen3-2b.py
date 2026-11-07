"""根据 step3 的画框视频和检测 JSON，用本地 Qwen3-VL-2B 判断控球人。

每个事件读对应的 mp4 和 JSON。发生帧整幅画面用来看左上角时钟和球，离球最近的几个人裁出来看球衣号和衣色。
检测 JSON 提供框和球的坐标。控球人的画面坐标用选中框的中心，不让模型凭空写像素。
step3 没有跨片段跟踪，player_id 写 unknown。name_cn 由衣色和已有动作组成：红衣 Team A，白衣 Team B。
这台机器是 6GB 显卡，照片会缩小后再送入。
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DETECTIONS = PROJECT_ROOT / "outputs" / "opensportslib" / "detections" / "test_3min"
DEFAULT_MODEL = "Qwen/Qwen3-VL-2B-Instruct"
FRAME_MAX_PIXELS = 256 * 144
CROP_MAX_PIXELS = 160 * 256
CLOCK_CROP = (0, 0, 640, 220)
CANDIDATE_LIMIT = 3
JERSEY_RE = re.compile(r"^(?:[1-9]|[1-9][0-9])$")
CLOCK_RE = re.compile(r"^\d{1,2}:\d{2}(?:\.\d{1,2})?$")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="用本地 Qwen 根据 step3 视频和检测 JSON 判断控球人。")
    parser.add_argument("--detections", type=Path, default=DEFAULT_DETECTIONS, help="step3 的 JSON 文件或目录")
    parser.add_argument("--output", type=Path, default=None, help="默认 outputs/opensportslib/<视频名>.possession.json")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default=None, help="cuda 或 cpu，默认有显卡就用 cuda")
    return parser.parse_args()


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _events(path: Path) -> list[dict]:
    if path.is_dir():
        files = sorted(path.glob("*.json"))
        if not files:
            raise SystemExit(f"目录里没有检测 JSON: {path}")
        return [_load(item) for item in files]
    if not path.is_file():
        raise SystemExit(f"找不到 step3 结果: {path}")
    document = _load(path)
    if "frames" in document:
        return [document]
    return list(document.get("events") or [])


def _read_video(path: Path):
    import av

    container = av.open(str(path))
    stream = container.streams.video[0]
    images = [frame.to_image() for frame in container.decode(stream)]
    container.close()
    return images


def _center(xyxy: list[float]) -> list[float]:
    left, top, right, bottom = (float(value) for value in xyxy)
    return [round((left + right) / 2.0, 2), round((top + bottom) / 2.0, 2)]


def _ball_distance(person: dict, ball_xy: list[float]) -> float:
    left, top, right, bottom = (float(value) for value in person["xyxy"])
    x, y = (float(value) for value in ball_xy)
    dx = 0.0 if left <= x <= right else min(abs(x - left), abs(x - right))
    dy = 0.0 if top <= y <= bottom else min(abs(y - top), abs(y - bottom))
    return math.hypot(dx, dy)


def _event_offset(event: dict) -> int:
    target = int(event["source_frame"])
    for index, frame in enumerate(event["frames"]):
        if int(frame["source_frame"]) == target:
            return index
    return len(event["frames"]) // 2


def _ball_on(frames: list[dict], center: int) -> tuple[dict | None, list[float] | None]:
    order = sorted(range(len(frames)), key=lambda index: (abs(index - center), index))
    for index in order:
        ball = frames[index].get("ball")
        if ball and ball.get("xy"):
            return ball, [float(value) for value in ball["xy"]]
    return None, None


def _candidates(frame: dict, ball_xy: list[float] | None) -> list[dict]:
    people = list(frame.get("players") or [])
    ranked = []
    for index, person in enumerate(people):
        item = dict(person)
        item["index"] = index
        item["distance"] = None if ball_xy is None else round(_ball_distance(person, ball_xy), 2)
        ranked.append(item)
    if ball_xy is None:
        ranked.sort(key=lambda item: item["confidence"], reverse=True)
    else:
        ranked.sort(key=lambda item: (item["distance"], -item["confidence"]))
    return ranked[:CANDIDATE_LIMIT]


def _fit(image, max_pixels: int):
    from PIL import Image

    width, height = image.size
    if width * height <= max_pixels:
        return image
    scale = (max_pixels / float(width * height)) ** 0.5
    resized = image.resize((max(1, int(width * scale)), max(1, int(height * scale))), Image.Resampling.LANCZOS)
    return resized


def _crop(image, xyxy: list[float]):
    width, height = image.size
    left, top, right, bottom = (float(value) for value in xyxy)
    box_width = max(right - left, 1.0)
    box_height = max(bottom - top, 1.0)
    left = max(0, int(left - box_width * 0.2))
    top = max(0, int(top - box_height * 0.15))
    right = min(width, int(right + box_width * 0.2))
    bottom = min(height, int(bottom + box_height * 0.1))
    if right <= left or bottom <= top:
        return image
    return image.crop((left, top, right, bottom))


def _action_name(label: str) -> str:
    text = str(label).replace("(", "（").replace(")", "）")
    if text == "进球":
        return "Goal"
    if text == "扑救":
        return "_GKSave"
    return text


def _name_cn(team: str | None, label: str) -> str:
    action = _action_name(label)
    if team in {"A", "B"} and action == "Goal":
        return f"Team {team}Goal"
    if team in {"A", "B"} and action == "_GKSave":
        return f"Team {team}_GKSave"
    if team in {"A", "B"} and action != "角球":
        return f"Team {team}{action}"
    return action


def _jersey(value) -> str:
    text = str(value or "").strip()
    return text if JERSEY_RE.fullmatch(text) else "unknown"


def _clock(value) -> str | None:
    text = re.sub(r"\s+", "", str(value or "")).replace("：", ":")
    return text if CLOCK_RE.fullmatch(text) else None


def _team(value) -> str | None:
    text = str(value or "").strip().upper()
    return text if text in {"A", "B"} else None


def _prompt(event: dict, candidates: list[dict], ball_xy: list[float] | None) -> str:
    rows = []
    for item in candidates:
        center = _center(item["xyxy"])
        distance = "无球" if item["distance"] is None else f"离球 {item['distance']} 像素"
        rows.append(
            f"{len(rows)}: {item['class']} 置信度 {item['confidence']} 框中心 {center} {distance}"
        )
    timeline = []
    for frame in event["frames"]:
        ball = frame.get("ball")
        timeline.append(
            {
                "source_frame": frame["source_frame"],
                "stats": frame["stats"],
                "ball_xy": None if not ball else ball.get("xy"),
            }
        )
    ball_text = "检测 JSON 里这个片段没有球。" if ball_xy is None else f"用来对照的球中心是 { [round(value, 2) for value in ball_xy] }。"
    return (
        f"动作是{event['label']}，事件帧是 {event['source_frame']}。"
        "第 1 张图是画面左上角，只用来读比赛时钟。第 2 张是这段检测视频的事件帧，绿色框是球员，蓝色框是门将，黄色框是裁判，红色框是球。"
        "后面的图按 0、1、2 的顺序对应下面的候选人，用来看球衣号码和衣服颜色。"
        f"{ball_text}"
        "红衣是 Team A，白衣是 Team B。"
        "只输出一个 JSON，不要解释。键为 player_index, player_code, team, action_screen_time_period。"
        "player_index 只能是 0、1 或 2，无法判断写 null。"
        "player_code 只写这个人球衣上能看清的号码，看不清写 unknown。不要把时钟、比分或广告上的数字当成球衣号。"
        "team 只能是 A 或 B，看不出写 null。"
        "action_screen_time_period 只写第 1 张图时钟上的分和秒，用一个冒号连接。不要补百分秒，也不要写画面上没有的时间。"
        f"候选人：{'；'.join(rows)}。"
        f"检测 JSON 逐帧摘要：{json.dumps(timeline, ensure_ascii=False)}"
    )


def _images_for(event: dict, pictures, candidates: list[dict]) -> list:
    offset = _event_offset(event)
    if offset >= len(pictures):
        raise SystemExit(f"{event.get('clip')} 的视频帧数 {len(pictures)} 少于检测 JSON")
    frame = pictures[offset]
    images = [
        _fit(frame.crop(CLOCK_CROP), FRAME_MAX_PIXELS),
        _fit(frame, FRAME_MAX_PIXELS),
    ]
    for item in candidates:
        images.append(_fit(_crop(frame, item["xyxy"]), CROP_MAX_PIXELS))
    return images


def _json_object(text: str) -> dict | None:
    start = text.find("{")
    while start >= 0:
        depth = 0
        for end in range(start, len(text)):
            if text[end] == "{":
                depth += 1
            elif text[end] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        value = json.loads(text[start : end + 1])
                    except json.JSONDecodeError:
                        value = None
                    if isinstance(value, dict):
                        return value
                    break
        start = text.find("{", start + 1)
    return None


def _ask(model, processor, images, text: str) -> str:
    import torch

    messages = [
        {
            "role": "user",
            "content": [{"type": "image", "image": image} for image in images] + [{"type": "text", "text": text}],
        }
    ]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[prompt],
        images=images,
        max_pixels=max(FRAME_MAX_PIXELS, CROP_MAX_PIXELS),
        min_pixels=64 * 64,
        return_tensors="pt",
    )
    grid = inputs.get("image_grid_thw")
    if grid is not None:
        tokens = int(inputs["input_ids"].shape[1])
        print(f"画面网格 {grid.tolist()}，输入 {tokens} token", flush=True)
    inputs = {key: value.to(model.device) if hasattr(value, "to") else value for key, value in inputs.items()}
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=120, do_sample=False)
    new_tokens = output[:, inputs["input_ids"].shape[1] :]
    return processor.batch_decode(new_tokens, skip_special_tokens=True)[0]


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


def _player_index(value):
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _record(event: dict, candidates: list[dict], answer: dict) -> dict:
    picked = _player_index(answer.get("player_index"))
    chosen = candidates[picked] if picked is not None and 0 <= picked < len(candidates) else None
    if chosen is None:
        point = [0.0, 0.0]
        code = "unknown"
        team = None
    else:
        point = _center(chosen["xyxy"])
        code = _jersey(answer.get("player_code"))
        team = _team(answer.get("team"))
    return {
        "index": event.get("index"),
        "clip": event.get("clip"),
        "label": event.get("label"),
        "position_ms": event.get("position_ms"),
        "frame": event.get("source_frame"),
        "player_code": code,
        "player_id": "unknown",
        "name_cn": _name_cn(team, str(event.get("label") or "")),
        "player_screen_xy": point,
        "action_screen_time_period": _clock(answer.get("action_screen_time_period")),
    }


def main() -> None:
    args = _parse_args()
    events = _events(args.detections)
    if not events:
        raise SystemExit("没有可判断的事件")
    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    stem = Path(events[0]["clips_index"]).parent.name if events[0].get("clips_index") else args.detections.stem
    output = args.output or (PROJECT_ROOT / "outputs" / "opensportslib" / f"{stem}.possession.json")
    print(f"模型 {args.model}，设备 {device}，事件 {len(events)} 个", flush=True)
    model, processor = _load_model(args.model, device)
    records = []
    for event in events:
        video_path = Path(str(event.get("boxed_video") or ""))
        if not video_path.is_file():
            raise SystemExit(f"找不到画框视频: {video_path}")
        offset = _event_offset(event)
        _ball, ball_xy = _ball_on(event["frames"], offset)
        candidates = _candidates(event["frames"][offset], ball_xy)
        pictures = _read_video(video_path)
        images = _images_for(event, pictures, candidates)
        raw = _ask(model, processor, images, _prompt(event, candidates, ball_xy))
        for image in images:
            image.close()
        for image in pictures:
            image.close()
        answer = _json_object(raw) or {}
        record = _record(event, candidates, answer)
        records.append(record)
        print(f"{int(event.get('index') or 0):03d} {event.get('label')} {json.dumps(record, ensure_ascii=False)}", flush=True)
    result = {
        "model": args.model,
        "detections": str(args.detections),
        "events": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"控球结果 {output}")


if __name__ == "__main__":
    main()
