"""用 sports 的球员跟踪处理原视频，写出带轨迹和检测框坐标的视频。

输入是一段原始比赛视频，默认 test_data/test_30s.mp4。
调用 sports/examples/soccer/main.py 的 PLAYER_TRACKING：
不改原画面内容。足球和裁判不跟踪。只在拷贝上画脚底圆环，追踪号写在脚底，人物中心点 [x,y] 写在头顶。
不改原片。这台机器用 cuda。
视频用 H.264、yuv420p 写出，并把索引放到文件头，保证常见播放器能打开。
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SPORTS_ROOT = PROJECT_ROOT / "sports"
SOCCER_DIR = SPORTS_ROOT / "examples" / "soccer"
DEFAULT_VIDEO = PROJECT_ROOT / "test_data" / "1-2-6.mp4"
sys.path.insert(0, str(SPORTS_ROOT))
sys.path.insert(0, str(SOCCER_DIR))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="调用 sports 球员跟踪，输出轨迹和坐标视频。")
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO, help="原始比赛视频")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="默认 outputs/opensportslib/<视频名>.step_src.mp4",
    )
    parser.add_argument("--device", default=None, help="cuda 或 cpu，默认有显卡就用 cuda")
    parser.add_argument("--max-seconds", type=float, default=None, help="只处理开头若干秒，便于试跑")
    return parser.parse_args()


def _ffmpeg() -> str | None:
    found = shutil.which("ffmpeg")
    if found:
        return found
    conda = Path.home() / "miniconda3" / "bin" / "ffmpeg"
    if conda.is_file():
        return str(conda)
    return None


class _PlayableWriter:
    """用 H.264 写 mp4。MPEG-4 很多播放器拒绝打开。"""

    def __init__(self, path: Path, width: int, height: int, fps: float):
        import av

        self.path = path
        self.count = 0
        self.container = av.open(str(path), mode="w")
        rate = int(round(fps)) or 30
        self.stream = self.container.add_stream("libx264", rate=rate)
        self.stream.width = width - width % 2
        self.stream.height = height - height % 2
        self.stream.pix_fmt = "yuv420p"
        self.stream.options = {"preset": "veryfast", "crf": "23"}

    def write(self, image) -> None:
        import av

        frame = av.VideoFrame.from_ndarray(image, format="bgr24")
        frame = frame.reformat(width=self.stream.width, height=self.stream.height, format="yuv420p")
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
        _faststart(self.path)


def _faststart(path: Path) -> None:
    """把 moov 移到文件头，播放器不用读完整文件就能开头。"""
    ffmpeg = _ffmpeg()
    if ffmpeg is None:
        return
    temporary = path.with_suffix(".faststart.mp4")
    result = subprocess.run(
        [ffmpeg, "-y", "-i", str(path), "-c", "copy", "-movflags", "+faststart", str(temporary)],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0 and temporary.is_file():
        temporary.replace(path)
    elif temporary.is_file():
        temporary.unlink()


def main() -> None:
    args = _parse_args()
    video = args.video.expanduser().resolve()
    if not video.is_file():
        raise SystemExit(f"找不到原视频: {video}")
    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    output = args.output or (PROJECT_ROOT / "outputs" / "opensportslib" / f"{video.stem}.step0.mp4")
    output.parent.mkdir(parents=True, exist_ok=True)

    import supervision as sv
    from main import run_player_tracking

    info = sv.VideoInfo.from_video_path(str(video))
    limit = None if args.max_seconds is None else max(1, int(args.max_seconds * info.fps))
    total = info.total_frames if limit is None else min(limit, info.total_frames or limit)
    print(f"设备 {device}，{video.name}，{info.fps:.2f} 帧/秒，写出 {total} 帧 -> {output}")
    written = 0
    writer = _PlayableWriter(output, info.width, info.height, info.fps)
    try:
        for frame in run_player_tracking(str(video), device):
            writer.write(frame)
            written += 1
            if written % 100 == 0:
                print(f"已写 {written} 帧", flush=True)
            if limit is not None and written >= limit:
                break
    finally:
        writer.close()
    print(f"轨迹视频 {output}，{written} 帧")


if __name__ == "__main__":
    main()
