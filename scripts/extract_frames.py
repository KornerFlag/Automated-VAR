#!/usr/bin/env python3
"""
Frame Extraction Script

Extract frames from video for annotation and training.
"""

import argparse
import sys
from pathlib import Path
import cv2
from loguru import logger


def extract_frames(
    video_path: str,
    output_dir: str,
    fps: int = 5,
    max_frames: int = None,
    start_time: float = 0,
    end_time: float = None,
    resize: tuple = None,
    format: str = "jpg",
    quality: int = 95
):
    """
    Extract frames from video.
    
    Args:
        video_path: Path to input video
        output_dir: Output directory for frames
        fps: Frames per second to extract
        max_frames: Maximum number of frames
        start_time: Start time in seconds
        end_time: End time in seconds
        resize: Optional resize (width, height)
        format: Output format (jpg/png)
        quality: JPEG quality (1-100)
    """
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Could not open video: {video_path}")
        return 0
    
    # Get video properties
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / video_fps if video_fps > 0 else 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    logger.info(f"Video: {video_path}")
    logger.info(f"  Resolution: {width}x{height}")
    logger.info(f"  FPS: {video_fps:.2f}")
    logger.info(f"  Duration: {duration:.2f}s")
    logger.info(f"  Total Frames: {total_frames}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Calculate frame interval
    frame_interval = max(1, int(video_fps / fps))
    
    # Set start frame
    start_frame = int(start_time * video_fps)
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    # Set end frame
    end_frame = total_frames
    if end_time:
        end_frame = min(total_frames, int(end_time * video_fps))
    
    logger.info(f"Extracting at {fps} fps (every {frame_interval} frames)")
    logger.info(f"Frame range: {start_frame} to {end_frame}")
    
    # Extract frames
    frame_number = start_frame
    extracted_count = 0
    video_name = Path(video_path).stem
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_number >= end_frame:
            break
        
        if max_frames and extracted_count >= max_frames:
            break
        
        if (frame_number - start_frame) % frame_interval == 0:
            # Resize if specified
            if resize:
                frame = cv2.resize(frame, resize)
            
            # Save frame
            filename = f"{video_name}_frame_{frame_number:06d}.{format}"
            filepath = output_path / filename
            
            if format == "jpg":
                cv2.imwrite(
                    str(filepath), 
                    frame, 
                    [cv2.IMWRITE_JPEG_QUALITY, quality]
                )
            else:
                cv2.imwrite(str(filepath), frame)
            
            extracted_count += 1
            
            if extracted_count % 50 == 0:
                logger.info(f"  Extracted {extracted_count} frames...")
        
        frame_number += 1
    
    cap.release()
    
    logger.info(f"Extraction complete: {extracted_count} frames saved to {output_path}")
    return extracted_count


def main():
    parser = argparse.ArgumentParser(
        description="Extract frames from video for training"
    )
    
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--fps", type=int, default=5, help="Frames per second")
    parser.add_argument("--max-frames", type=int, help="Maximum frames to extract")
    parser.add_argument("--start", type=float, default=0, help="Start time (seconds)")
    parser.add_argument("--end", type=float, help="End time (seconds)")
    parser.add_argument("--resize", type=int, nargs=2, help="Resize to width height")
    parser.add_argument("--format", default="jpg", choices=["jpg", "png"])
    parser.add_argument("--quality", type=int, default=95, help="JPEG quality")
    
    args = parser.parse_args()
    
    resize = tuple(args.resize) if args.resize else None
    
    extract_frames(
        video_path=args.video,
        output_dir=args.output,
        fps=args.fps,
        max_frames=args.max_frames,
        start_time=args.start,
        end_time=args.end,
        resize=resize,
        format=args.format,
        quality=args.quality
    )


if __name__ == "__main__":
    main()
