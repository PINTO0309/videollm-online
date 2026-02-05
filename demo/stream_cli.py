import argparse
import os
import sys
import time

import cv2
import torch
import transformers

from .inference import LiveInfer

logger = transformers.logging.get_logger("streaming")


def _parse_cli_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--source", choices=["video", "webcam"], default="video")
    parser.add_argument("--video_path", default="demo/assets/cooking.mp4")
    parser.add_argument("--webcam_index", type=int, default=0)
    parser.add_argument("--input_fps", type=float, default=None)
    parser.add_argument("--output_interval", type=float, default=3.0)
    parser.add_argument("--query", default="Please narrate the video in real time.")
    parser.add_argument("--frame_token_interval_threshold", type=float, default=None)
    parser.add_argument("--pad_color", default="0,0,0")
    return parser.parse_known_args()


def _parse_pad_color(value: str):
    parts = value.split(",")
    if len(parts) != 3:
        raise ValueError(f"Invalid --pad_color format: {value}. Expected 'R,G,B'.")
    return tuple(int(p) for p in parts)


def _resize_and_pad(frame_bgr, resolution: int, pad_color_rgb):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    height, width = frame_rgb.shape[:2]
    if height == 0 or width == 0:
        return None
    scale = resolution / max(height, width)
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(frame_rgb, (new_w, new_h), interpolation=interp)
    pad_w = resolution - new_w
    pad_h = resolution - new_h
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    if pad_w or pad_h:
        resized = cv2.copyMakeBorder(
            resized,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_CONSTANT,
            value=pad_color_rgb,
        )
    return resized


def _frame_to_tensor(frame_rgb):
    return torch.from_numpy(frame_rgb).permute(2, 0, 1).contiguous()


def _process_queue(liveinfer: LiveInfer, buffer: list[str], last_response_content: str | None):
    while True:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        query, response = liveinfer.step()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        if query is None and response is None:
            break
        print(f"[Inference] step_time={elapsed * 1000:.1f} ms", flush=True)
        if query:
            buffer.append(str(query))
        if response:
            response_text = str(response)
            content = response_text
            if "Assistant:" in response_text:
                content = response_text.split("Assistant:", 1)[1].strip()
            if content != last_response_content:
                buffer.append(response_text)
                last_response_content = content
    return last_response_content


def _maybe_emit(buffer: list[str], last_output_time: float, now: float, output_interval: float):
    if now - last_output_time < output_interval:
        return last_output_time
    if buffer:
        print("\n".join(buffer), flush=True)
        buffer.clear()
    return now


def _open_capture(source: str, video_path: str, webcam_index: int):
    if source == "video":
        if not os.path.exists(video_path):
            raise FileNotFoundError(video_path)
        cap = cv2.VideoCapture(video_path)
    else:
        cap = cv2.VideoCapture(webcam_index)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open {source} source.")
    return cap


def main():
    cli_args, remaining = _parse_cli_args()
    sys.argv = [sys.argv[0]] + remaining
    liveinfer = LiveInfer()
    if cli_args.frame_token_interval_threshold is not None:
        liveinfer.frame_token_interval_threshold = cli_args.frame_token_interval_threshold

    pad_color_rgb = _parse_pad_color(cli_args.pad_color)
    cap = _open_capture(cli_args.source, cli_args.video_path, cli_args.webcam_index)
    try:
        source_fps = cap.get(cv2.CAP_PROP_FPS) or 0
        if cli_args.source == "video" and source_fps <= 0:
            source_fps = liveinfer.frame_fps
        input_fps = cli_args.input_fps or liveinfer.frame_fps
        if cli_args.source == "video" and source_fps > 0:
            input_fps = min(input_fps, source_fps)
        if input_fps <= 0:
            input_fps = liveinfer.frame_fps

        liveinfer.frame_fps = input_fps
        liveinfer.frame_interval = 1 / input_fps
        logger.warning(f"Input FPS = {input_fps:.2f}, Output Interval = {cli_args.output_interval:.2f}s")

        if cli_args.query:
            liveinfer.input_query_stream(cli_args.query, video_time=0.0)

        buffer = []
        last_response_content = None
        last_output_time = 0.0
        next_sample_time = 0.0
        input_interval = 1 / input_fps

        with torch.no_grad():
            if cli_args.source == "video":
                frame_idx = 0
                source_interval = 1 / max(source_fps, 1e-6)
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    video_time = frame_idx * source_interval
                    frame_idx += 1
                    if video_time + 1e-6 < next_sample_time:
                        continue
                    next_sample_time += input_interval
                    resized = _resize_and_pad(frame, liveinfer.frame_resolution, pad_color_rgb)
                    if resized is None:
                        continue
                    frame_tensor = _frame_to_tensor(resized)
                    liveinfer.input_frame_tensor(frame_tensor, video_time)
                    last_response_content = _process_queue(liveinfer, buffer, last_response_content)
                    last_output_time = _maybe_emit(buffer, last_output_time, video_time, cli_args.output_interval)
            else:
                start_time = time.time()
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    now = time.time() - start_time
                    if now + 1e-6 < next_sample_time:
                        continue
                    next_sample_time += input_interval
                    resized = _resize_and_pad(frame, liveinfer.frame_resolution, pad_color_rgb)
                    if resized is None:
                        continue
                    frame_tensor = _frame_to_tensor(resized)
                    liveinfer.input_frame_tensor(frame_tensor, now)
                    last_response_content = _process_queue(liveinfer, buffer, last_response_content)
                    last_output_time = _maybe_emit(buffer, last_output_time, now, cli_args.output_interval)
                    sleep_for = next_sample_time - (time.time() - start_time)
                    if sleep_for > 0:
                        time.sleep(sleep_for)
        if buffer:
            print("\n".join(buffer), flush=True)
            buffer.clear()
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
