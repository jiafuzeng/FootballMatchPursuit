"""根据 step3 的控球记录，用 Qwen3.8-Max 从视频里找接球人。

只处理传球和长传。片段从带标注的视频上截取，出球前留 5 帧。
短传看出球后 3 秒，慢放成 3 秒；长传看出球后 5 秒，慢放成 5 秒。按 5 帧/秒抽帧。已经上传过的片段不覆盖。
读到的追踪号如果和传球人相同，视为没找到。结果写入 step4 JSON，不改 step3。
"""

from __future__ import annotations

import argparse
import base64
import json
import os
from fractions import Fraction
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_MODEL = "qwen3.8-max"
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_EVENTS = PROJECT_ROOT / "outputs" / "opensportslib" / "test.step0.step3.qwen38-max.json"
DEFAULT_VIDEO = PROJECT_ROOT / "outputs" / "opensportslib" / "test.step0.mp4"
DEFAULT_CLIPS = PROJECT_ROOT / "outputs" / "opensportslib" / "clips" / "step4"
LEAD_FRAMES = 5  # 出球前保留的源帧数
PASS_DELAY_SECONDS = 3  # 短传：出球后再看的秒数
LONG_DELAY_SECONDS = 5  # 长传：出球后再看的秒数
PASS_OUTPUT_SECONDS = 3  # 短传成片时长，单位秒
LONG_OUTPUT_SECONDS = 5  # 长传成片时长，单位秒
SAMPLE_FPS = 5  # 从源视频抽帧的帧率
JERSEY_RE = re.compile(r"^(?:[1-9]|[1-9][0-9])$")


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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="用 Qwen3.8-Max 从出球到下一次触球的视频里找接球人。")
    parser.add_argument("--events", type=Path, default=DEFAULT_EVENTS, help="step3 的控球 JSON")
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO, help="带脚底追踪号的视频")
    parser.add_argument("--output", type=Path, default=None, help="默认把 step3 文件名中的 step3 换成 step4")
    parser.add_argument("--clips", type=Path, default=DEFAULT_CLIPS, help="截出的片段目录，默认 outputs/opensportslib/clips/step4")
    parser.add_argument("--index", type=int, default=None, help="只处理这一个传球序号，从 1 开始")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=BASE_URL, help="默认读 .env 的 BASE_URL")
    parser.add_argument("--api-key", default=API_KEY, help="默认读 .env 的 API_KEY")
    return parser.parse_args()


def _load(path: Path) -> list[dict]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(document, dict):
        document = document.get("events") or []
    if not isinstance(document, list) or not document:
        raise SystemExit(f"控球记录里没有事件: {path}")
    return sorted(document, key=lambda item: int(item["frame"]))


def _jersey(value) -> str:
    text = str(value or "").strip()
    return text if JERSEY_RE.fullmatch(text) else "unknown"


def _track_id(value) -> str:
    text = str(value or "").strip()
    if text.isdigit() and 1 <= len(text) <= 6 and int(text) > 0:
        return str(int(text))
    return "unknown"


def _video_meta(video: Path) -> tuple[float, int, int, int]:
    import av

    container = av.open(str(video))
    stream = container.streams.video[0]
    fps = float(stream.average_rate) if stream.average_rate else 30.0
    frames = int(stream.frames or 0)
    if frames <= 0 and stream.duration is not None and fps > 0:
        frames = int(stream.duration * stream.time_base * fps)
    width = int(stream.codec_context.width)
    height = int(stream.codec_context.height)
    container.close()
    if fps <= 1.0 or frames <= 0:
        raise SystemExit(f"无法读取视频帧率或帧数: {video}")
    return fps, frames, width, height


def _action(name: str) -> str:
    text = str(name)
    for prefix in ("Team A", "Team B"):
        if text.startswith(prefix):
            text = text[len(prefix):]
            break
    return text


def _sample_indexes(frame: int, n_frames: int, delay_seconds: int, fps: float) -> tuple[int, int, set[int]]:
    """出球前留 LEAD_FRAMES，出球后再看 delay 秒，按 SAMPLE_FPS 抽帧。出球帧一定保留。"""
    step = max(1, int(round((fps if fps > 1 else 30.0) / SAMPLE_FPS)))
    start = max(0, frame - LEAD_FRAMES)
    end = min(n_frames - 1, frame + int(round(delay_seconds * (fps if fps > 1 else 30.0))))
    first = frame - ((frame - start) // step) * step
    if first < start:
        first += step
    if first > end:
        return end, end, {end}
    return first, end, set(range(first, end + 1, step))


def _playback_rate(picked: int, output_seconds: int) -> Fraction:
    """抽到的帧铺满成片时长，按这个帧率慢放。"""
    if picked <= 0 or output_seconds <= 0:
        return Fraction(SAMPLE_FPS)
    return Fraction(picked, output_seconds)


def _prompt() -> str:
    return """这段视频从出球前开始，到球被接下来之后结束。
画面上每个球员和门将都有脚底圆环：圆环中间的数字是追踪号。头顶方括号里的两个数是中心坐标。足球和裁判没有这些标注。
开头把球传出去的人是传球人。接球人是之后第一个接到球的另一个人。

只输出一个 JSON 对象，不要解释，不要其他字段。键只有这两个：

- receiver_code: 接球人球衣上的号码。看不清写 unknown。
- receiver_id: 接球人脚底圆环里的追踪号，写成数字字符串。不要读球衣号，不要读传球人的号。看不清或看不到接球人写 unknown。
"""


class _ClipWriter:
    def __init__(self, path: Path, width: int, height: int, rate: Fraction):
        import av

        path.parent.mkdir(parents=True, exist_ok=True)
        self.count = 0
        self.container = av.open(str(path), mode="w")
        self.stream = self.container.add_stream("libx264", rate=rate)
        self.stream.width = width - width % 2
        self.stream.height = height - height % 2
        self.stream.pix_fmt = "yuv420p"
        self.stream.options = {"preset": "ultrafast", "crf": "23"}

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


def _cut(video: Path, jobs: list[dict], fps: float, width: int, height: int) -> None:
    """按源帧号扫一遍视频，把每个传球窗口写成一个 mp4。"""
    import av

    if not jobs:
        return
    container = av.open(str(video))
    stream = container.streams.video[0]
    writers: dict[int, _ClipWriter] = {}
    pending = sorted(jobs, key=lambda item: item["start"])
    active: list[dict] = []
    cursor = 0
    for index, frame in enumerate(container.decode(stream)):
        while cursor < len(pending) and pending[cursor]["start"] <= index:
            job = pending[cursor]
            writers[job["index"]] = _ClipWriter(job["path"], width, height, job["rate"])
            active.append(job)
            cursor += 1
        still = []
        for job in active:
            if index <= job["end"]:
                if index in job["picks"]:
                    writers[job["index"]].write(frame)
                still.append(job)
            else:
                writers.pop(job["index"]).close()
        active = still
        if cursor >= len(pending) and not active:
            break
    for job in active:
        writers.pop(job["index"]).close()
    container.close()
    for job in jobs:
        if job["path"].is_file():
            job["frames"] = len(job["picks"])


def _api_video(source: Path, fps: float, frame_count: int) -> Path:
    """接口不收短于 2 秒的视频。原帧不够长时只放慢，画面不改。"""
    import av

    duration = frame_count / fps if frame_count and fps else 0.0
    if duration >= 2:
        return source
    container = av.open(str(source))
    pictures = [frame.to_ndarray(format="bgr24") for frame in container.decode(container.streams.video[0])]
    container.close()
    if not pictures:
        raise SystemExit(f"片段没有画面: {source}")
    rate = 8
    while len(pictures) / rate < 2.05:
        if rate > 1:
            rate -= 1
        else:
            pictures = pictures + pictures
    destination = source.parent.parent / f"{source.parent.name}_2s" / source.name
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


def _ask(video: Path, prompt: str, model: str, base_url: str, api_key: str) -> str:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "video_url", "video_url": {"url": _video_url(video)}},
                    {"type": "text", "text": prompt + "\n第一个字符必须是 {。不要分析，不要markdown。"},
                ],
            }
        ],
        "temperature": 0,
        "max_tokens": 256,
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


def _record(event: dict, receiver_code: str, receiver_id: str) -> dict:
    return {
        "frame": int(event["frame"]),
        "player_code": event.get("player_code", "unknown"),
        "player_id": event.get("player_id", "unknown"),
        "name_cn": event.get("name_cn", ""),
        "receiver_code": receiver_code,
        "receiver_id": receiver_id,
        "player_screen_xy": event.get("player_screen_xy") or [0.0, 0.0],
        "action_screen_time_period": event.get("action_screen_time_period", "unknown"),
    }


def _write(output: Path, records: list[dict]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _output_path(events_path: Path, chosen: Path | None) -> Path:
    if chosen is not None:
        return chosen
    name = events_path.name.replace(".step3.", ".step4.")
    if name == events_path.name:
        name = f"{events_path.stem}.step4.json"
    return events_path.with_name(name)


def main() -> None:
    args = _parse_args()
    if not args.base_url or not args.api_key:
        raise SystemExit("请在 .env 中配置 BASE_URL 和 API_KEY")
    events_path = args.events.expanduser().resolve()
    video = args.video.expanduser().resolve()
    if not video.is_file():
        raise SystemExit(f"找不到视频: {video}")
    events = _load(events_path)
    fps, n_frames, width, height = _video_meta(video)
    output = _output_path(events_path, args.output)
    done = {}
    if output.is_file():
        previous = json.loads(output.read_text(encoding="utf-8"))
        if isinstance(previous, list):
            done = {int(item["frame"]): item for item in previous if "frame" in item}
    jobs = []
    for index, event in enumerate(events, start=1):
        action = _action(str(event.get("name_cn") or ""))
        if action not in {"传球", "长传"}:
            continue
        if args.index is not None and index != args.index:
            continue
        frame = int(event["frame"])
        delay = LONG_DELAY_SECONDS if action == "长传" else PASS_DELAY_SECONDS
        output_seconds = LONG_OUTPUT_SECONDS if action == "长传" else PASS_OUTPUT_SECONDS
        start, end, picks = _sample_indexes(frame, n_frames, delay, fps)
        jobs.append(
            {
                "index": index,
                "event": event,
                "start": start,
                "end": end,
                "picks": picks,
                "rate": _playback_rate(len(picks), output_seconds),
                "path": args.clips / f"{index:03d}_{frame}_{output_seconds}s.mp4",
                "frames": len(picks),
                "keep": frame in done,
            }
        )
    _cut(video, [job for job in jobs if not job["keep"]], fps, width, height)
    records = []
    job_at = {job["index"]: job for job in jobs}
    for index, event in enumerate(events, start=1):
        kept = done.get(int(event["frame"]))
        if kept is not None:
            records.append(kept)
            continue
        job = job_at.get(index)
        if job is None:
            records.append(_record(event, "unknown", "unknown"))
            continue
        clip = job["path"]
        if not clip.is_file():
            raise SystemExit(f"没有截出片段: {clip}")
        print(
            f"{index:03d} {event.get('name_cn')} 帧 {event.get('frame')} "
            f"截 {job['start']}-{job['end']} {clip.name}",
            file=sys.stderr,
        )
        sent = _api_video(clip, job["rate"], job["frames"])
        text = _ask(sent, _prompt(), args.model, args.base_url, args.api_key)
        print(text)
        answer = _answer_of(_json_value(text))
        receiver_id = _track_id(answer.get("receiver_id"))
        receiver_code = _jersey(answer.get("receiver_code"))
        if receiver_id != "unknown" and receiver_id == str(event.get("player_id") or ""):
            receiver_id = "unknown"
            receiver_code = "unknown"
        record = _record(event, receiver_code, receiver_id)
        records.append(record)
        print(json.dumps(record, ensure_ascii=False), file=sys.stderr)
        _write(output, records)
    _write(output, records)
    print(f"已写入: {output}，合计 {len(records)} 条", file=sys.stderr)


if __name__ == "__main__":
    main()
