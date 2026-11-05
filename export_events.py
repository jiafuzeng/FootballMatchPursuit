"""把 OpenSportsLib 的定位结果写成 test.json 那种事件记录。

输入是 outputs/opensportslib/test.predictions.json。每条预测只有动作类别和
发生时间。球员、接球人、球队和画面坐标不在这个模型的输出里，对应字段写
unknown，坐标写 [0.0, 0.0]。没有对应中文事件名的类别不写入结果。

帧号和画面时间按视频帧率计算。test.mp4 是 30 帧/秒，与 test.json 的约定一致：
frame = round(秒 × 帧率)，action_screen_time_period 为 MM:SS.cc。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_PREDICTIONS = PROJECT_ROOT / "outputs" / "opensportslib" / "test.predictions.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "opensportslib" / "test.events.json"

# 只映射事件说明书里有对应名称的类别。头球、出界、传中、任意球、封堵没有同名事件。
LABEL_TO_ACTION = {
    "PASS": "传球",
    "DRIVE": "停球（带球）",
    "HIGH PASS": "长传",
    "THROW IN": "界外手抛球",
    "SHOT": "射门",
    "PLAYER SUCCESSFUL TACKLE": "抢断",
    "GOAL": "进球",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="把 OpenSportsLib 定位结果转成 test.json 格式。")
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--team",
        choices=("A", "B"),
        default=None,
        help="定位模型没有球队。只有确认整段视频的行动方后才填写，否则 name_cn 只写动作名",
    )
    parser.add_argument("--fps", type=float, default=None, help="覆盖视频帧率，默认从预测里的视频读取")
    return parser.parse_args()


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


def _events(payload: dict, fps: float, team: str | None) -> tuple[list[dict], Counter]:
    skipped: Counter = Counter()
    records = []
    for sample in payload.get("data", []):
        for event in sample.get("events", []):
            label = str(event.get("label", ""))
            action = LABEL_TO_ACTION.get(label)
            if action is None:
                skipped[label] += 1
                continue
            position_ms = int(event["position_ms"])
            records.append(
                {
                    "frame": _frame(position_ms, fps),
                    "player_code": "unknown",
                    "player_id": "unknown",
                    "name_cn": _name_cn(action, team),
                    "receiver_code": "unknown",
                    "receiver_id": "unknown",
                    "player_screen_xy": [0.0, 0.0],
                    "action_screen_time_period": _screen_time(position_ms),
                }
            )
    records.sort(key=lambda item: (item["frame"], item["action_screen_time_period"], item["name_cn"]))
    return records, skipped


def main() -> int:
    args = _parse_args()
    predictions = args.predictions.expanduser().resolve()
    if not predictions.is_file():
        print(f"找不到预测文件: {predictions}", file=sys.stderr)
        return 1
    payload = json.loads(predictions.read_text(encoding="utf-8"))
    if args.fps is not None:
        fps = args.fps
    else:
        video = _prediction_video(payload)
        if video is None or not video.is_file():
            print("预测里没有可用视频，请用 --fps 指定帧率", file=sys.stderr)
            return 1
        fps = _video_fps(video)
        print(f"视频帧率: {fps:g}  ({video})")

    records, skipped = _events(payload, fps, args.team)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"写入 {len(records)} 条事件: {output}")
    counts = Counter(item["name_cn"] for item in records)
    for name, count in sorted(counts.items()):
        print(f"  {name}: {count}")
    if skipped:
        print("未写入（事件说明书没有对应名称）:")
        for label, count in sorted(skipped.items()):
            print(f"  {label}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
