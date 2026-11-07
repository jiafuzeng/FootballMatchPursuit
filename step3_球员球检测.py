"""用 sports 官方 PLAYER_DETECTION，检测 step2 每个事件片段里的球员和球。

输入是 step2 的 step2.json 和同目录下的 mp4。片段按原帧率逐帧检测。
对应 examples/soccer/main.py 的 PLAYER_DETECTION：只加载 football-player-detection，imgsz 1280。
类别是 ball、goalkeeper、player、referee。这台机器用 cuda。
球员、门将、裁判的坐标取脚底中心，球取框中心。单位是像素，x 向右，y 向下。
同一帧有多个球框时保留置信度最高的一个。没有球时 ball 为 null。
默认把框画进单独的视频片段，并在每一帧标出球员、门将、裁判和球的数量。不改 step2 的原片。
每个片段单独写一份 JSON，文件名和视频一致。source_frame、start_frame、end_frame 原样沿用 step2。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_INDEX = PROJECT_ROOT / "outputs" / "opensportslib" / "clips" / "test_3min" / "step2.json"
PLAYER_MODEL = PROJECT_ROOT / "sports" / "examples" / "soccer" / "data" / "football-player-detection.pt"
BALL_CLASS_ID = 0
PERSON_CLASSES = {
    1: "goalkeeper",
    2: "player",
    3: "referee",
}
BOX_COLORS = {
    "player": (0, 220, 0),
    "goalkeeper": (255, 160, 0),
    "referee": (0, 220, 255),
    "ball": (0, 0, 255),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="用官方 PLAYER_DETECTION 检测 step2 视频片段里的球员和球。")
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX, help="step2 写出的 step2.json")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="检测 JSON 目录，默认 outputs/opensportslib/detections/<视频名>，每个片段一份",
    )
    parser.add_argument("--device", type=str, default=None, help="cuda 或 cpu，默认有显卡就用 cuda")
    parser.add_argument(
        "--no-draw",
        action="store_true",
        help="不把框画出来。默认会画成视频，存到 clips/<视频名>_boxed",
    )
    return parser.parse_args()


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _round_box(xyxy) -> list[float]:
    return [round(float(value), 2) for value in xyxy]


def _point(xyxy, bottom: bool) -> list[float]:
    left, top, right, bottom_edge = (float(value) for value in xyxy)
    x = (left + right) / 2.0
    y = bottom_edge if bottom else (top + bottom_edge) / 2.0
    return [round(x, 2), round(y, 2)]


def _split(detections) -> tuple[list[dict], dict | None]:
    """按官方模型的类别拆开。人取脚底，球取框中心。"""
    people = []
    balls = []
    if detections.class_id is None:
        return people, None
    for xyxy, confidence, class_id in zip(detections.xyxy, detections.confidence, detections.class_id):
        class_id = int(class_id)
        item = {
            "confidence": round(float(confidence), 4),
            "xy": _point(xyxy, bottom=class_id != BALL_CLASS_ID),
            "xyxy": _round_box(xyxy),
        }
        if class_id == BALL_CLASS_ID:
            balls.append(item)
        elif class_id in PERSON_CLASSES:
            item["class"] = PERSON_CLASSES[class_id]
            people.append(item)
    people.sort(key=lambda item: item["confidence"], reverse=True)
    ball = max(balls, key=lambda item: item["confidence"]) if balls else None
    return people, ball


def _stats(players: list[dict], ball: dict | None) -> dict:
    counts = {"player": 0, "goalkeeper": 0, "referee": 0, "ball": 0 if ball is None else 1}
    for person in players:
        name = person["class"]
        counts[name] = counts.get(name, 0) + 1
    return counts


def _annotate(frame, players: list[dict], ball: dict | None, source_frame: int):
    """在副本上画框，并写上这一帧的数量。原画面不改。"""
    import cv2

    canvas = frame.copy()
    marks = list(players)
    if ball is not None:
        marks.append({"class": "ball", **ball})
    for mark in marks:
        color = BOX_COLORS[mark["class"]]
        left, top, right, bottom = (int(round(value)) for value in mark["xyxy"])
        cv2.rectangle(canvas, (left, top), (right, bottom), color, 2)
        text = f"{mark['class']} {mark['confidence']:.2f}"
        origin = (left, max(top - 6, 16))
        cv2.putText(canvas, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    counts = _stats(players, ball)
    summary = (
        f"frame {source_frame}  player {counts['player']}  goalkeeper {counts['goalkeeper']}  "
        f"referee {counts['referee']}  ball {counts['ball']}"
    )
    height, width = canvas.shape[:2]
    cv2.rectangle(canvas, (0, height - 36), (width, height), (0, 0, 0), -1)
    cv2.putText(canvas, summary, (12, height - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


class _ClipWriter:
    def __init__(self, path: Path, width: int, height: int, rate: int):
        import av

        path.parent.mkdir(parents=True, exist_ok=True)
        self.count = 0
        self.container = av.open(str(path), mode="w")
        self.stream = self.container.add_stream("libx264", rate=rate)
        self.stream.width = width
        self.stream.height = height
        self.stream.pix_fmt = "yuv420p"
        self.stream.options = {"preset": "ultrafast", "crf": "23"}

    def write(self, image) -> None:
        import av

        frame = av.VideoFrame.from_ndarray(image, format="bgr24")
        frame = frame.reformat(format="yuv420p")
        frame.pts = self.count
        self.count += 1
        for packet in self.stream.encode(frame):
            if packet.dts is None:
                packet.dts = packet.pts
            self.container.mux(packet)

    def close(self) -> None:
        for packet in self.stream.encode():
            if packet.dts is None:
                packet.dts = packet.pts
            self.container.mux(packet)
        self.container.close()


def _prepare_boxed(directory: Path) -> None:
    """清掉上一版的画框照片，只留这次写出的视频。"""
    directory.mkdir(parents=True, exist_ok=True)
    for stale in directory.glob("*.mp4"):
        stale.unlink()
    for folder in directory.iterdir():
        if not folder.is_dir():
            continue
        for photo in folder.glob("*.jpg"):
            photo.unlink()
        folder.rmdir()


def _clip_frames(path: Path):
    import av

    container = av.open(str(path))
    stream = container.streams.video[0]
    for frame in container.decode(stream):
        yield frame.to_ndarray(format="bgr24")
    container.close()


def _prepare_json_dir(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for stale in directory.glob("*.json"):
        stale.unlink()


def _event_document(index_path: Path, video: str | None, boxed_root: Path | None, event: dict) -> dict:
    clip = event.get("clip")
    return {
        "mode": "PLAYER_DETECTION",
        "clips_index": str(index_path),
        "video": video,
        "player_model": str(PLAYER_MODEL),
        "boxed_video": None if boxed_root is None or not clip else str(boxed_root / clip),
        "index": event.get("index"),
        "clip": clip,
        "label": event.get("label"),
        "position_ms": event.get("position_ms"),
        "source_frame": event.get("source_frame"),
        "start_frame": event.get("start_frame"),
        "end_frame": event.get("end_frame"),
        "frames": event["frames"],
    }


def _write_event(directory: Path, document: dict) -> Path:
    name = Path(str(document["clip"])).with_suffix(".json").name
    path = directory / name
    path.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def _detect(index_path: Path, device: str, boxed_root: Path | None) -> dict:
    import supervision as sv
    from ultralytics import YOLO

    payload = _load(index_path)
    clips_root = index_path.parent
    rate = int(round(float(payload.get("fps") or 30)))
    player_model = YOLO(str(PLAYER_MODEL))
    if boxed_root is not None:
        _prepare_boxed(boxed_root)
    events = []
    for event in payload.get("events", []):
        clip_path = clips_root / event["clip"]
        if not clip_path.is_file():
            raise FileNotFoundError(f"找不到片段: {clip_path}")
        start_frame = int(event["start_frame"])
        writer = None
        frames = []
        for offset, frame in enumerate(_clip_frames(clip_path)):
            source_frame = start_frame + offset
            result = player_model(frame, imgsz=1280, verbose=False, device=device)[0]
            players, ball = _split(sv.Detections.from_ultralytics(result))
            counts = _stats(players, ball)
            if boxed_root is not None:
                if writer is None:
                    height, width = frame.shape[:2]
                    writer = _ClipWriter(boxed_root / event["clip"], width, height, rate)
                writer.write(_annotate(frame, players, ball, source_frame))
            frames.append(
                {
                    "source_frame": source_frame,
                    "stats": counts,
                    "players": players,
                    "ball": ball,
                }
            )
        if writer is not None:
            writer.close()
        events.append(
            {
                "index": event.get("index"),
                "clip": event.get("clip"),
                "label": event.get("label"),
                "position_ms": event.get("position_ms"),
                "source_frame": event.get("source_frame"),
                "start_frame": event.get("start_frame"),
                "end_frame": event.get("end_frame"),
                "frames": frames,
            }
        )
        ball_count = sum(frame["ball"] is not None for frame in frames)
        player_count = sum(len(frame["players"]) for frame in frames)
        print(
            f"{int(event.get('index') or 0):03d} {event.get('label')} {event.get('position_ms')}ms "
            f"人员框 {player_count} 有球 {ball_count}/{len(frames)}"
        )
    return {
        "video": payload.get("video"),
        "boxed_root": boxed_root,
        "events": events,
    }


def main() -> None:
    args = _parse_args()
    if not args.index.is_file():
        raise SystemExit(f"找不到 step2 清单: {args.index}")
    if not PLAYER_MODEL.is_file():
        raise SystemExit(f"找不到球员检测权重: {PLAYER_MODEL}")
    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    stem = args.index.parent.name
    output = args.output or (PROJECT_ROOT / "outputs" / "opensportslib" / "detections" / stem)
    boxed_root = None if args.no_draw else args.index.parent.parent / f"{stem}_boxed"
    payload = _detect(args.index, device, boxed_root)
    _prepare_json_dir(output)
    for event in payload["events"]:
        _write_event(output, _event_document(args.index, payload["video"], boxed_root, event))
    frames = sum(len(event["frames"]) for event in payload["events"])
    print(f"设备 {device}，{len(payload['events'])} 个片段，{frames} 帧")
    print(f"检测结果 {output}")
    if boxed_root is not None:
        print(f"画框片段 {boxed_root}")


if __name__ == "__main__":
    main()
