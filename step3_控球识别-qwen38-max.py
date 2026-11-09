"""根据 step2 的事件片段，用 Qwen3.8-Max 填写控球记录。

输入是 step2.json 和同目录的片段。片段要截自已经画好脚底追踪号、头顶中心坐标的视频。
模型只看画面：球衣号、脚底追踪号、头顶方括号里的中心坐标、衣色、比分牌时钟。
输出的 frame 用 step2 的 source_frame，也就是原视频帧号。
name_cn 由衣色和 step2 里的动作组成：红衣 Team A，白衣 Team B。
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_MODEL = "qwen3.8-max"
PROJECT_ROOT = Path(__file__).resolve().parent


def _load_env(path: Path) -> None:
    """读取项目根目录的 .env。已有环境变量优先，不覆盖。"""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text or text.startswith("#") or "=" not in text:
            continue
        key, value = text.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


_load_env(PROJECT_ROOT / ".env")
BASE_URL = os.environ.get("BASE_URL", "")
API_KEY = os.environ.get("API_KEY", "")
DEFAULT_STEP2 = PROJECT_ROOT / "outputs" / "opensportslib" / "clips" / "1-2-6.step0" / "step2.json"
JERSEY_RE = re.compile(r"^(?:[1-9]|[1-9][0-9])$")
CLOCK_RE = re.compile(r"^\d{1,2}:\d{2}(?:\.\d{1,2})?$")
NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="用 Qwen3.8-Max 根据 step2 片段填写控球记录。")
    parser.add_argument("--step2", type=Path, default=DEFAULT_STEP2, help="step2.json，或它所在的目录")
    parser.add_argument("--output", type=Path, default=None, help="默认 outputs/opensportslib/<视频名>.step3.qwen38-max.json")
    parser.add_argument("--index", type=int, default=None, help="只处理这一个事件序号")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=BASE_URL, help="OpenAI 兼容接口根地址，默认读 .env 的 BASE_URL")
    parser.add_argument("--api-key", default=API_KEY, help="接口密钥，默认读 .env 的 API_KEY")
    return parser.parse_args()


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _manifest(path: Path) -> tuple[Path, dict]:
    if path.is_dir():
        path = path / "step2.json"
    if not path.is_file():
        raise SystemExit(f"找不到 step2 清单: {path}")
    document = _load(path)
    if not document.get("events"):
        raise SystemExit(f"step2 清单里没有事件: {path}")
    return path, document


def _output_frame(event: dict, fps: float) -> int:
    """test.json 的 frame 是原视频帧号，对应 step2 的 source_frame。"""
    if event.get("source_frame") is not None:
        return int(event["source_frame"])
    return int(round(int(event["position_ms"]) / 1000.0 * fps))


def _offset(event: dict) -> int:
    start = int(event.get("start_frame") or 0)
    center = int(event.get("source_frame") or start)
    offset = max(center - start, 0)
    total = int(event.get("frames") or 0)
    if total > 0:
        return min(offset, total - 1)
    return offset


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


def _track_id(value) -> str:
    """脚底追踪号。去掉前导零，认不出就写 unknown。"""
    text = str(value or "").strip()
    if text.isdigit() and 1 <= len(text) <= 6 and int(text) > 0:
        return str(int(text))
    return "unknown"


def _clock(value) -> str:
    text = re.sub(r"\s+", "", str(value or "")).replace("：", ":")
    return text if CLOCK_RE.fullmatch(text) else "unknown"


def _team(value) -> str | None:
    text = str(value or "").strip().upper()
    return text if text in {"A", "B"} else None


def _xy(value) -> list[float]:
    """读画面上头顶方括号里的两个数。认不出就写 [0.0, 0.0]。"""
    if isinstance(value, str):
        value = NUMBER_RE.findall(value)
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        try:
            x = float(value[0])
            y = float(value[1])
        except (TypeError, ValueError):
            return [0.0, 0.0]
        if 0.0 <= x <= 10000.0 and 0.0 <= y <= 10000.0:
            return [x, y]
    return [0.0, 0.0]


def _prompt(event: dict) -> str:
    return f"""只判断事件帧的控球人。动作是{event.get("label")}。
附上的静图就是事件帧。视频只用来看谁碰到球，数字不要从视频的其他帧里读。
画面上每个球员和门将都有脚底圆环：圆环中间的数字是追踪号，头顶方括号里的两个数是这个人的中心坐标。足球和裁判没有这些标注。
控球人是这一帧碰到球、或正在把球传出去的球员或门将。

只输出一个 JSON 对象，不要解释，不要其他字段。键只有这五个：

- player_code: 控球人球衣上的号码。看不清写 unknown。
- player_id: 控球人脚底圆环里的追踪号，写成数字字符串。不要读球衣号。看不清写 unknown。
- player_screen_xy: 控球人头顶方括号里的两个数，按从左到右放进数组。看不清写 null。
- team: 红衣写 A，白衣写 B。看不出写 null。
- action_screen_time_period: 这一帧画面上方比分牌里的比赛时钟，不是用帧号换算的时间。时钟在比分正下方，分和秒之间用冒号。画面上如果还有小数，按画面原样保留。看不清写 unknown。
"""


def _api_video(source: Path) -> Path:
    """接口不收短于 2 秒的视频。把片段的原帧放慢，画面不改。"""
    import av

    container = av.open(str(source))
    frames = [frame.to_ndarray(format="bgr24") for frame in container.decode(container.streams.video[0])]
    container.close()
    if not frames:
        raise SystemExit(f"片段没有画面: {source}")
    pictures = list(frames)
    rate = 8
    while len(pictures) / rate < 2.05:
        if rate > 1:
            rate -= 1
        else:
            pictures = pictures + pictures
    destination = source.parent.parent / f"{source.parent.name}_2s" / source.name
    if destination.is_file() and destination.stat().st_mtime >= source.stat().st_mtime:
        return destination
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


def _event_image(video: Path, offset: int):
    import av

    container = av.open(str(video))
    image = None
    last = None
    for index, frame in enumerate(container.decode(container.streams.video[0])):
        last = frame
        if index == offset:
            image = frame.to_ndarray(format="bgr24")
            break
    container.close()
    if image is None and last is not None and offset > 0:
        image = last.to_ndarray(format="bgr24")
    if image is None:
        raise SystemExit(f"{video.name} 没有第 {offset} 帧")
    return image


def _jpeg_url(image) -> str:
    import cv2

    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    if not ok:
        raise SystemExit("事件帧无法编码成 JPEG")
    return "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")


def _video_url(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:video/mp4;base64,{encoded}"


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


def _answer_of(value) -> dict:
    if isinstance(value, dict):
        return value
    if isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                return item
    return {}


def _ask(image_url: str, video: Path, prompt: str, model: str, base_url: str, api_key: str) -> str:
    content = [
        {"type": "image_url", "image_url": {"url": image_url}},
        {"type": "video_url", "video_url": {"url": _video_url(video)}},
        {"type": "text", "text": prompt + "\n第一个字符必须是 {。不要分析，不要markdown。"},
    ]
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0,
        "max_tokens": 512,
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


def _record(event: dict, fps: float, answer: dict) -> dict:
    team = _team(answer.get("team"))
    return {
        "frame": _output_frame(event, fps),
        "player_code": _jersey(answer.get("player_code")),
        "player_id": _track_id(answer.get("player_id")),
        "name_cn": _name_cn(team, str(event.get("label") or "")),
        "player_screen_xy": _xy(answer.get("player_screen_xy")),
        "action_screen_time_period": _clock(answer.get("action_screen_time_period")),
    }


def _write(output: Path, records: list[dict]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = _parse_args()
    if not args.base_url or not args.api_key:
        raise SystemExit("请在 .env 中配置 BASE_URL 和 API_KEY")
    manifest_path, document = _manifest(args.step2.expanduser().resolve())
    events = list(document["events"])
    if args.index is not None:
        events = [event for event in events if int(event.get("index") or 0) == args.index]
    if not events:
        raise SystemExit("没有可判断的事件")
    fps = float(document.get("fps") or 30.0)
    stem = Path(str(document.get("video") or manifest_path.parent)).stem
    output = args.output or (PROJECT_ROOT / "outputs" / "opensportslib" / f"{stem}.step3.qwen38-max.json")
    records = []
    for event in events:
        clip = manifest_path.parent / str(event.get("clip") or "")
        if not clip.is_file():
            raise SystemExit(f"找不到 step2 片段: {clip}")
        offset = _offset(event)
        print(
            f"{int(event.get('index') or 0):03d} {event.get('label')} 帧 {_output_frame(event, fps)} "
            f"{clip.name} {clip.stat().st_size / 1024 / 1024:.1f}MB",
            file=sys.stderr,
        )
        image = _jpeg_url(_event_image(clip, offset))
        text = _ask(image, _api_video(clip), _prompt(event), args.model, args.base_url, args.api_key)
        print(text)
        record = _record(event, fps, _answer_of(_json_value(text)))
        records.append(record)
        print(json.dumps(record, ensure_ascii=False), file=sys.stderr)
        _write(output, records)
    print(f"已写入: {output}，合计 {len(records)} 条", file=sys.stderr)


if __name__ == "__main__":
    main()
