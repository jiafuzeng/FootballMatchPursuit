"""把事件帧的全部检测框和画框视频交给 Qwen3.8-Max，让它指出控球人的框。

检测 JSON 原样列出这一帧的人员框和球。模型只返回框序号。
序号必须落在给出的框里。若已有 step4 结果，再对照是不是同一个人的脚底。
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_MODEL = "qwen3.8-max"
BASE_URL = "https://api.agicto.cn/v1"
API_KEY = "sk-G61fbkgJzoYPXjIoHFCmnDFzFzprPfOTNWej6GDP7NT1SGh5"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_VIDEO = "/home/gc/FootballMatchPursuit/outputs/opensportslib/clips/test_3min_boxed_2s/001_长传_10000ms.mp4"
DEFAULT_DETECTIONS = "/home/gc/FootballMatchPursuit/outputs/opensportslib/detections/test_3min/001_长传_10000ms.json"
DEFAULT_STEP4 = PROJECT_ROOT / "outputs" / "opensportslib" / "test_3min.step4.qwen38-max.json"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="测试模型能否从检测框里指出控球人。")
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO, help="送给模型的画框视频")
    parser.add_argument("--detections", type=Path, default=DEFAULT_DETECTIONS, help="step3 的 JSON 文件或目录")
    parser.add_argument("--step4", type=Path, default=DEFAULT_STEP4, help="已有的 step4 结果，用来对照脚底")
    parser.add_argument("--index", type=int, default=1, help="事件序号，默认只测第 1 条")
    parser.add_argument("--output", type=Path, default=None, help="默认 outputs/opensportslib/test_step4_控球框.json")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=BASE_URL)
    parser.add_argument("--api-key", default=API_KEY)
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


def _event_offset(event: dict) -> int:
    target = int(event["source_frame"])
    for index, frame in enumerate(event["frames"]):
        if int(frame["source_frame"]) == target:
            return index
    return len(event["frames"]) // 2


def _boxes(frame: dict) -> list[dict]:
    boxes = []
    for index, person in enumerate(frame.get("players") or []):
        boxes.append(
            {
                "box_index": index,
                "class": person.get("class"),
                "confidence": person.get("confidence"),
                "xyxy": person.get("xyxy"),
                "xy": person.get("xy"),
            }
        )
    return boxes


def _prompt(event: dict, boxes: list[dict], ball: dict | None) -> str:
    payload = {
        "frame": event.get("source_frame"),
        "label": event.get("label"),
        "boxes": boxes,
        "ball": None if not ball else {"xy": ball.get("xy"), "xyxy": ball.get("xyxy")},
    }
    return (
        "视频是 step3 画出的检测片段。绿色框是球员，蓝色框是门将，黄色框是裁判，红色框是球。\n"
        "下面是事件帧上的检测框，顺序和内容都来自检测 JSON。box_index 就是这个列表里的序号。\n"
        "只判断这一帧谁在控球，指出控球人的框。控球人是正在碰球或把球控制在身边的人。球在空中时，指出刚刚把球送出的人。\n"
        "只输出一个 JSON 对象，不要解释。键只有 box_index。它必须是下面某个 box_index；看不出写 null。不要新编框，不要写坐标。\n"
        f"{json.dumps(payload, ensure_ascii=False)}\n"
        "第一个字符必须是 {。不要分析，不要markdown。"
    )


def _api_video(source: Path) -> Path:
    """接口不收短于 2 秒的视频。已有放慢文件就直接用。"""
    destination = source.parent.parent / f"{source.parent.name}_2s" / source.name
    if destination.is_file() and destination.stat().st_mtime >= source.stat().st_mtime:
        return destination
    import av

    container = av.open(str(source))
    frames = [frame.to_ndarray(format="bgr24") for frame in container.decode(container.streams.video[0])]
    container.close()
    if not frames:
        raise SystemExit(f"检测视频没有画面: {source}")
    pictures = list(frames)
    rate = 8
    while len(pictures) / rate < 2.05:
        if rate > 1:
            rate -= 1
        else:
            pictures = pictures + pictures
    destination.parent.mkdir(parents=True, exist_ok=True)
    height, width = pictures[0].shape[:2]
    output = av.open(str(destination), mode="w")
    stream = output.add_stream("libx264", rate=rate)
    stream.width = width - width % 2
    stream.height = height - height % 2
    stream.pix_fmt = "yuv420p"
    stream.options = {"preset": "ultrafast", "crf": "23"}
    for index, image in enumerate(pictures):
        frame = av.VideoFrame.from_ndarray(image, format="bgr24")
        frame = frame.reformat(width=stream.width, height=stream.height, format="yuv420p")
        frame.pts = index
        for packet in stream.encode(frame):
            if packet.dts is None:
                packet.dts = packet.pts
            output.mux(packet)
    for packet in stream.encode():
        if packet.dts is None:
            packet.dts = packet.pts
        output.mux(packet)
    output.close()
    return destination


def _video_url(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:video/mp4;base64,{encoded}"


def _json_value(text: str):
    start = 0
    while True:
        bracket = text.find("{", start)
        if bracket < 0:
            return None
        depth = 0
        for end in range(bracket, len(text)):
            if text[end] == "{":
                depth += 1
            elif text[end] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[bracket : end + 1])
                    except json.JSONDecodeError:
                        break
        start = bracket + 1


def _ask(video: Path, prompt: str, model: str, base_url: str, api_key: str) -> str:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "video_url", "video_url": {"url": _video_url(video)}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        "temperature": 0,
        "max_tokens": 128,
        "enable_thinking": False,
        "response_format": {"type": "json_object"},
    }
    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"接口返回 {exc.code}: {detail[:800]}") from exc
    message = body["choices"][0]["message"]
    return str(message.get("content") or message.get("reasoning_content") or "")


def _box_index(value, count: int) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        index = int(value)
    except (TypeError, ValueError):
        return None
    if 0 <= index < count:
        return index
    return None


def _same_point(left, right) -> bool | None:
    if not left or not right or len(left) < 2 or len(right) < 2:
        return None
    return abs(float(left[0]) - float(right[0])) < 1.0 and abs(float(left[1]) - float(right[1])) < 1.0


def _step4_xy(path: Path, frame: int):
    if not path.is_file():
        return None
    for event in _load(path).get("events") or []:
        if int(event.get("frame") or -1) == frame:
            return event.get("player_screen_xy")
    return None


def _mark(video: Path, offset: int, box: dict | None, ball: dict | None, destination: Path) -> None:
    import av
    import cv2

    container = av.open(str(video))
    image = None
    for index, frame in enumerate(container.decode(container.streams.video[0])):
        if index == offset:
            image = frame.to_ndarray(format="bgr24")
            break
    container.close()
    if image is None or box is None:
        return
    left, top, right, bottom = (int(round(float(value))) for value in box["xyxy"])
    cv2.rectangle(image, (left, top), (right, bottom), (255, 0, 255), 4)
    cv2.putText(
        image,
        f"picked {box['box_index']}",
        (left, max(top - 8, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 0, 255),
        2,
        cv2.LINE_AA,
    )
    if ball and ball.get("xy"):
        x, y = (int(round(float(value))) for value in ball["xy"])
        cv2.circle(image, (x, y), 12, (0, 255, 255), 3)
    destination.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(destination), image)


def main() -> None:
    args = _parse_args()
    events = [event for event in _events(args.detections) if int(event.get("index") or 0) == args.index]
    if not events:
        raise SystemExit(f"没有事件序号 {args.index}")
    event = events[0]
    video = args.video.expanduser().resolve()
    if not video.is_file():
        raise SystemExit(f"找不到画框视频: {video}")
    offset = _event_offset(event)
    frame = event["frames"][offset]
    boxes = _boxes(frame)
    ball = frame.get("ball")
    prompt = _prompt(event, boxes, ball)
    clip = video
    print(
        f"{int(event.get('index') or 0):03d} {event.get('label')} 帧 {event.get('source_frame')} "
        f"框 {len(boxes)} 个，视频 {clip.name}",
        file=sys.stderr,
    )
    text = _ask(clip, prompt, args.model, args.base_url, args.api_key)
    print(text)
    answer = _json_value(text) or {}
    picked = _box_index(answer.get("box_index"), len(boxes))
    chosen = None if picked is None else boxes[picked]
    reference = _step4_xy(args.step4, int(event["source_frame"]))
    same = _same_point(None if chosen is None else chosen.get("xy"), reference)
    if picked is None:
        verdict = "不正确：没有返回给出的框序号"
    elif same is True:
        verdict = "序号有效，脚底和已有 step4 结果是同一个人"
    elif same is False:
        verdict = "序号有效，脚底和已有 step4 结果不是同一个人"
    else:
        verdict = "序号有效，没有 step4 结果可对照"
    picture = PROJECT_ROOT / "outputs" / "opensportslib" / f"test_step4_控球框_{int(event.get('index') or 0):03d}.jpg"
    _mark(video, offset, chosen, ball, picture)
    result = {
        "model": args.model,
        "index": event.get("index"),
        "label": event.get("label"),
        "frame": event.get("source_frame"),
        "video": str(clip),
        "box_count": len(boxes),
        "raw": text,
        "box_index": picked,
        "box": chosen,
        "ball_xy": None if not ball else ball.get("xy"),
        "step4_xy": reference,
        "same_as_step4": same,
        "verdict": verdict,
        "picture": str(picture),
    }
    output = args.output or (PROJECT_ROOT / "outputs" / "opensportslib" / "test_step4_控球框.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(verdict, file=sys.stderr)


if __name__ == "__main__":
    main()
