"""按 step1 的定位结果，为每个事件截取一段视频。

每个事件的 frame 原样沿用 step1 预测里的 frame，不改数值、不改名字。
片段中心仍按 position_ms 和源视频帧率计算：round(position_ms / 1000 × fps)。
每个事件单独一个 mp4，内容是发生帧的前 10 帧、这一帧和后 10 帧，按原帧率保存。
清单里的序号和文件名与事件一一对应，供后面的检测使用。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_PREDICTIONS = PROJECT_ROOT / "outputs" / "opensportslib" / "test_3min.step1.json"
DEFAULT_CLIPS = PROJECT_ROOT / "outputs" / "opensportslib" / "clips"
DEFAULT_RADIUS = 10


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="按定位时刻从源视频截取每个事件的视频片段。")
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument(
        "--video",
        type=Path,
        default=None,
        help="源视频。不传时用预测文件里的路径；那条路径不在本机时，改用 test_data 下的同名文件",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="片段目录，默认 outputs/opensportslib/clips/<视频名>",
    )
    parser.add_argument(
        "--radius",
        type=int,
        default=DEFAULT_RADIUS,
        help="发生帧前后各保留的帧数，默认 10，连同发生帧一共 21 帧",
    )
    return parser.parse_args()


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _events(payload: dict) -> list[dict]:
    events = []
    for sample in payload.get("data", []):
        events.extend(sample.get("events", []))
    events.sort(key=lambda event: int(event["position_ms"]))
    return events


def _recorded_video(payload: dict) -> Path | None:
    for sample in payload.get("data", []):
        for item in sample.get("inputs", []):
            if item.get("type") == "video" and item.get("path"):
                return Path(item["path"])
    return None


def _usable(path: Path) -> bool:
    try:
        return path.is_file()
    except OSError:
        return False


def _resolve_video(payload: dict, override: Path | None) -> Path:
    if override is not None:
        if not _usable(override):
            raise FileNotFoundError(f"找不到视频: {override}")
        return override
    recorded = _recorded_video(payload)
    if recorded is not None and _usable(recorded):
        return recorded
    if recorded is not None:
        local = PROJECT_ROOT / "test_data" / recorded.name
        if _usable(local):
            return local
    raise FileNotFoundError("预测文件里的视频路径在本机不存在，请用 --video 指定")


def _video_meta(video: Path) -> tuple[float, int]:
    import av

    container = av.open(str(video))
    stream = container.streams.video[0]
    fps = float(stream.average_rate) if stream.average_rate else 0.0
    frames = int(stream.frames or 0)
    if frames <= 0 and stream.duration is not None and fps > 0:
        frames = int(stream.duration * stream.time_base * fps)
    container.close()
    if fps <= 1.0 or frames <= 0:
        raise RuntimeError(f"无法读取视频帧率或帧数: {video}")
    return fps, frames


def _source_frame(position_ms: int, fps: float) -> int:
    return int(round(position_ms / 1000.0 * fps))


def _clip_name(index: int, event: dict) -> str:
    label = str(event.get("label", "event")).replace("/", "_")
    return f"{index:03d}_{label}_{int(event['position_ms'])}ms.mp4"


def _span(center: int, radius: int, n_frames: int) -> tuple[int, int]:
    start = max(0, center - radius)
    end = min(n_frames - 1, center + radius)
    return start, end


class _ClipWriter:
    def __init__(self, path: Path, width: int, height: int, rate: int):
        import av

        self.count = 0
        self.container = av.open(str(path), mode="w")
        self.stream = self.container.add_stream("libx264", rate=rate)
        self.stream.width = width
        self.stream.height = height
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


def _save_clips(video: Path, jobs: list[dict], directory: Path, fps: float) -> None:
    """按源帧号扫一遍视频。每个事件的连续帧写成一个 mp4。"""
    import av

    directory.mkdir(parents=True, exist_ok=True)
    for stale in directory.glob("*.mp4"):
        stale.unlink()

    order = sorted(jobs, key=lambda job: int(job["start_frame"]))
    container = av.open(str(video))
    stream = container.streams.video[0]
    rate = int(round(fps))
    width = stream.codec_context.width
    height = stream.codec_context.height
    writers: dict[int, _ClipWriter] = {}
    cursor = 0
    active: list[dict] = []
    index = 0
    for frame in container.decode(stream):
        while cursor < len(order) and order[cursor]["start_frame"] <= index:
            job = order[cursor]
            writers[job["index"]] = _ClipWriter(directory / job["clip"], width, height, rate)
            active.append(job)
            cursor += 1
        still_open = []
        for job in active:
            if index <= job["end_frame"]:
                writers[job["index"]].write(frame)
                job["frames"] += 1
                still_open.append(job)
            else:
                writers.pop(job["index"]).close()
        active = still_open
        index += 1
        if cursor >= len(order) and not active:
            break
    for job in active:
        writers.pop(job["index"]).close()
    container.close()
    short = [job for job in jobs if job["frames"] != job["end_frame"] - job["start_frame"] + 1]
    if short:
        detail = ", ".join(f"{job['clip']} 得到 {job['frames']} 帧" for job in short)
        raise RuntimeError(f"有事件的片段没有截全: {detail}")


def main() -> None:
    args = _parse_args()
    if args.radius < 0:
        raise SystemExit("--radius 不能小于 0")
    payload = _load(args.predictions)
    events = _events(payload)
    if not events:
        raise SystemExit(f"预测文件里没有事件: {args.predictions}")
    video = _resolve_video(payload, args.video)
    fps, n_frames = _video_meta(video)
    directory = args.output or (DEFAULT_CLIPS / video.stem)

    jobs = []
    for index, event in enumerate(events, start=1):
        position_ms = int(event["position_ms"])
        center = min(max(_source_frame(position_ms, fps), 0), n_frames - 1)
        start, end = _span(center, args.radius, n_frames)
        jobs.append(
            {
                "index": index,
                "clip": _clip_name(index, event),
                "label": event.get("label"),
                "position_ms": position_ms,
                "frame": event.get("frame"),
                "source_frame": center,
                "start_frame": start,
                "end_frame": end,
                "frames": 0,
                "confidence": event.get("confidence"),
            }
        )
    _save_clips(video, jobs, directory, fps)
    manifest = {
        "video": str(video),
        "fps": fps,
        "radius_frames": args.radius,
        "predictions": str(args.predictions),
        "events": jobs,
    }
    manifest_path = directory / "step2.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"视频 {video}，{fps:g} 帧/秒，前后各 {args.radius} 帧，{len(jobs)} 个事件")
    print(f"片段目录 {directory}")
    print(f"清单 {manifest_path}")
    for job in jobs:
        print(
            f"{job['index']:03d} {job['label']} {job['position_ms']}ms frame {job['frame']} "
            f"源帧 {job['start_frame']}-{job['end_frame']} 共 {job['frames']} 帧 -> {job['clip']}"
        )


if __name__ == "__main__":
    main()
