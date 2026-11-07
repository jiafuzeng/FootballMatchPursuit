"""把一段视频和一段提示词发给 Qwen3.8-Max，原样打印模型返回的文本。"""

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
# 实测请求体到 16MB 能进模型，24MB 会被网关拒绝。单段原片控制在 9MB，base64 后大约 12MB。
CHUNK_BYTES = 9 * 1024 * 1024
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_VIDEO = PROJECT_ROOT / "test_data" / "test_3min.mp4"
DEFAULT_PROMPT = """参考事件说明书，给出完整的视频事件 JSON。只输出 JSON 数组，不要解释。

## 事件名

`Team A` 或 `Team B` 与动作直接相连，中间无空格。

动作为：传球、停球（带球）、长传、抢断、射门、助攻、界外手抛球。

进球写 `Team AGoal`、`Team BGoal`。扑救写 `Team A_GKSave`、`Team B_GKSave`。角球不带队名，写 `角球`。

## 传球人和接球人

传球、长传的一条记录里同时写下两个人。

- `player_code`、`player_id`、`player_screen_xy` 是传球人。
- `receiver_code`、`receiver_id` 是接球人。球衣号可以是 `unknown`，跟踪号要写上。

接球人接下来那条记录的 `player_id` 等于这次传球的 `receiver_id`。他的画面位置写在那条记录里。

多数情况下，下一条是他的停球（带球）。停球只写这个接球人，`receiver_code` 和 `receiver_id` 都是 `unknown`。若他接球后直接再传、再做长传或射门，中间可以没有停球，下一脚的 `player_id` 仍然等于上一脚的 `receiver_id`。

同一个人停球之后再出球时，两行的 `player_id` 相同。

`action_screen_time_period` 是这一帧画面上方比分牌里的比赛时钟，不是帧号换算出来的时间。时钟在比分正下方的蓝色底上，白色数字，只有分和秒，形如 15:26。照抄成 `MM:SS`，不要补百分秒，也不要写成 15:26.00。不要用帧号除以 30，也不要把这段视频的开头写成 00:00。

样例只说明字段怎么排。样例里 6 号跟踪号把球交给 26 号跟踪号，26 号跟踪号再交给 7 号跟踪号，7 号跟踪号停球：

```json
{"frame": 522, "player_code": "26", "player_id": "6", "name_cn": "Team A传球", "receiver_code": "unknown", "receiver_id": "26", "player_screen_xy": [956.63, 340.66], "action_screen_time_period": "15:43"}
{"frame": 565, "player_code": "unknown", "player_id": "26", "name_cn": "Team A传球", "receiver_code": "55", "receiver_id": "7", "player_screen_xy": [1307.87, 330.29], "action_screen_time_period": "15:44"}
{"frame": 611, "player_code": "55", "player_id": "7", "name_cn": "Team A停球（带球）", "receiver_code": "unknown", "receiver_id": "unknown", "player_screen_xy": [569.59, 426.39], "action_screen_time_period": "15:46"}
```
"""


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="把视频和提示词发给 Qwen3.8-Max，输出模型原文。")
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO, help="输入视频，默认 test_data/test_3min.mp4")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="提示词，默认按事件说明书输出完整事件 JSON")
    parser.add_argument("--output", type=Path, default=None, help="可选，把模型原文写入这个文件")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=BASE_URL, help="OpenAI 兼容接口根地址")
    parser.add_argument("--api-key", default=API_KEY)
    return parser.parse_args()


def _clip_dir(source: Path) -> Path:
    return PROJECT_ROOT / "outputs" / "opensportslib" / "clips" / f"{source.stem}_segments"


def _duration(path: Path) -> float:
    import av

    container = av.open(str(path))
    stream = container.streams.video[0]
    if stream.duration is not None:
        seconds = float(stream.duration * stream.time_base)
    else:
        seconds = 0.0
    container.close()
    if seconds <= 0:
        raise SystemExit(f"读不到视频时长: {path}")
    return seconds


def _cut(source: Path, start: float, end: float, destination: Path) -> None:
    """按时间把原片切出一段，不重新编码。"""
    import av

    container = av.open(str(source))
    stream = container.streams.video[0]
    output = av.open(str(destination), mode="w")
    video_out = output.add_stream_from_template(stream)
    container.seek(int(start / stream.time_base), any_frame=False, stream=stream)
    for packet in container.demux(stream):
        if packet.dts is None or packet.pts is None:
            continue
        timestamp = float(packet.pts * stream.time_base)
        if timestamp < start:
            continue
        if timestamp >= end:
            break
        packet.stream = video_out
        output.mux(packet)
    output.close()
    container.close()


def _segments(source: Path) -> list[tuple[float, float, Path]]:
    """按原片体积切成若干段，片段保存在 outputs/opensportslib/clips。"""
    duration = _duration(source)
    if source.stat().st_size <= CHUNK_BYTES:
        return [(0.0, duration, source)]
    span = max(8.0, duration * (CHUNK_BYTES / source.stat().st_size))
    directory = _clip_dir(source)
    directory.mkdir(parents=True, exist_ok=True)
    pieces = []
    start = 0.0
    index = 1
    while start < duration - 0.05:
        end = min(duration, start + span)
        if duration - end < 4.0:
            end = duration
        destination = directory / f"{index:03d}_{start:.2f}-{end:.2f}s.mp4"
        if not destination.is_file() or destination.stat().st_mtime < source.stat().st_mtime:
            _cut(source, start, end, destination)
        pieces.append((start, end, destination))
        index += 1
        start = end
    return pieces


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
        if "frame" in value or "name_cn" in value:
            return [value]
    return []


def _source_frame(event: dict, start_frame: int) -> dict:
    """模型若从片段内的 0 开始编号，换算成源视频帧号。"""
    frame = event.get("frame")
    if isinstance(frame, bool) or not isinstance(frame, (int, float)):
        return event
    if int(frame) < start_frame:
        event = dict(event)
        event["frame"] = int(frame) + start_frame
    return event


def _summarize(parts: list[dict]) -> list[dict]:
    """按源视频帧号合成一份事件，去掉切段重叠处的重复。"""
    rows = []
    for part in parts:
        for event in part["events"]:
            rows.append(_source_frame(event, part["start_frame"]))
    rows.sort(key=lambda event: int(event.get("frame") or 0))
    kept = []
    for event in rows:
        frame = int(event.get("frame") or 0)
        name = str(event.get("name_cn") or "")
        if any(
            abs(frame - int(item.get("frame") or 0)) <= 15 and str(item.get("name_cn") or "") == name
            for item in kept
        ):
            continue
        kept.append(event)
    return kept


def _video_url(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:video/mp4;base64,{encoded}"


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
        "max_tokens": 8192,
        "enable_thinking": False,
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


def main() -> None:
    args = _parse_args()
    video = args.video.expanduser().resolve()
    if not video.is_file():
        raise SystemExit(f"找不到视频: {video}")
    pieces = _segments(video)
    parts = []
    for index, (start, end, piece) in enumerate(pieces, start=1):
        start_frame = int(round(start * 30))
        print(
            f"第 {index}/{len(pieces)} 段 {start:.1f}-{end:.1f}s，{piece.stat().st_size / 1024 / 1024:.1f}MB",
            file=sys.stderr,
        )
        prompt = (
            f"{args.prompt}\n\n"
            f"这段画面是源视频的 {start:.2f} 秒到 {end:.2f} 秒，"
            f"源帧号 {start_frame} 到 {int(round(end * 30))}。只写这段里发生的事件。"
        )
        text = _ask(piece, prompt, args.model, args.base_url, args.api_key)
        events = _events_of(_json_value(text))
        print(text)
        print(f"第 {index} 段解析出 {len(events)} 条", file=sys.stderr)
        parts.append({
            "start": start,
            "end": end,
            "start_frame": start_frame,
            "clip": str(piece),
            "raw": text,
            "events": events,
        })
    events = _summarize(parts)
    result = {
        "model": args.model,
        "video": str(video),
        "events": events,
        "segments": [{key: part[key] for key in ("start", "end", "clip", "raw")} for part in parts],
    }
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    output = args.output or (PROJECT_ROOT / "outputs" / "opensportslib" / f"{video.stem}.qwen38-max.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text + "\n", encoding="utf-8")
    print(f"已写入: {output}，合计 {len(events)} 条", file=sys.stderr)


if __name__ == "__main__":
    main()
