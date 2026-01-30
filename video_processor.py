"""
Video Processing Module for Real-Time Object Detection
Handles webcam capture and video file processing
"""
import cv2
import time
import numpy as np
from pathlib import Path
from typing import Optional, Generator, Tuple
from datetime import datetime

from config import DetectionConfig


class VideoProcessor:
    """Handles video input from webcam or video files."""
    
    def __init__(self, config: DetectionConfig):
        """Initialize video processor with configuration."""
        self.config = config
        self.cap: Optional[cv2.VideoCapture] = None
        self.writer: Optional[cv2.VideoWriter] = None
        self.frame_count = 0
        self.start_time = None
        self.is_webcam = False
        
    def open(self) -> bool:
        """
        Open the video source.
        
        Returns:
            True if successfully opened, False otherwise
        """
        source = self.config.input_source
        
        # Determine if source is webcam or file
        if source.isdigit():
            source = int(source)
            self.is_webcam = True
            print(f"📷 Opening webcam {source}...")
        else:
            self.is_webcam = False
            if not Path(source).exists():
                print(f"❌ Video file not found: {source}")
                return False
            print(f"🎬 Opening video file: {source}")
        
        # Open video capture - try multiple backends for best compatibility
        if self.is_webcam:
            # Try default backend first (usually works best)
            self.cap = cv2.VideoCapture(source)
            if not self.cap.isOpened():
                print("⚠️ Default backend failed, trying MSMF...")
                self.cap = cv2.VideoCapture(source, cv2.CAP_MSMF)
        else:
            self.cap = cv2.VideoCapture(source)
        
        if not self.cap.isOpened():
            print("❌ Failed to open video source")
            return False
        
        # Set resolution for webcam
        if self.is_webcam:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.frame_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.frame_height)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer for real-time
            
            # Warmup: read a few frames to let camera stabilize
            print("⏳ Warming up camera...")
            for _ in range(5):
                self.cap.read()
        
        # Get actual properties
        self.actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.actual_fps = self.cap.get(cv2.CAP_PROP_FPS) or self.config.video_fps
        
        print(f"✅ Video source opened: {self.actual_width}x{self.actual_height} @ {self.actual_fps:.1f} FPS")
        
        self.start_time = time.time()
        return True
    
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read a single frame from the video source.
        
        Returns:
            Tuple of (success, frame)
        """
        if self.cap is None:
            return False, None
        
        ret, frame = self.cap.read()
        
        if ret:
            self.frame_count += 1
        
        return ret, frame
    
    def frames(self) -> Generator[np.ndarray, None, None]:
        """
        Generator that yields frames from the video source.
        
        Yields:
            BGR frames as numpy arrays
        """
        while True:
            ret, frame = self.read_frame()
            if not ret:
                break
            yield frame
    
    def get_fps(self) -> float:
        """
        Calculate current processing FPS.
        
        Returns:
            Frames per second
        """
        if self.start_time is None or self.frame_count == 0:
            return 0.0
        
        elapsed = time.time() - self.start_time
        return self.frame_count / elapsed if elapsed > 0 else 0.0
    
    def setup_writer(self) -> bool:
        """
        Setup video writer for saving annotated output.
        
        Returns:
            True if writer was set up successfully
        """
        if not self.config.save_video:
            return False
        
        # Generate output filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = self.config.output_dir / f"detection_{timestamp}.mp4"
        
        # Define codec and create writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(
            str(output_path),
            fourcc,
            self.config.video_fps,
            (self.actual_width, self.actual_height)
        )
        
        if self.writer.isOpened():
            print(f"📹 Saving output to: {output_path}")
            return True
        else:
            print("⚠️ Failed to setup video writer")
            self.writer = None
            return False
    
    def write_frame(self, frame: np.ndarray):
        """
        Write a frame to the output video.
        
        Args:
            frame: BGR frame to write
        """
        if self.writer is not None:
            self.writer.write(frame)
    
    def display_frame(self, frame: np.ndarray, window_name: str = "Object Detection") -> int:
        """
        Display a frame in a window.
        
        Args:
            frame: BGR frame to display
            window_name: Name of the display window
            
        Returns:
            Key code pressed (or -1 if no key)
        """
        if not self.config.show_display:
            return -1
        
        cv2.imshow(window_name, frame)
        return cv2.waitKey(1) & 0xFF
    
    def get_progress(self) -> Optional[float]:
        """
        Get video processing progress (for video files only).
        
        Returns:
            Progress as percentage (0-100) or None for webcam
        """
        if self.is_webcam or self.cap is None:
            return None
        
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames > 0:
            return (self.frame_count / total_frames) * 100
        return None
    
    def get_video_info(self) -> dict:
        """
        Get information about the video source.
        
        Returns:
            Dictionary with video information
        """
        info = {
            "source": self.config.input_source,
            "is_webcam": self.is_webcam,
            "width": self.actual_width,
            "height": self.actual_height,
            "fps": self.actual_fps,
            "frames_processed": self.frame_count
        }
        
        if not self.is_webcam and self.cap is not None:
            info["total_frames"] = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            info["duration_seconds"] = info["total_frames"] / self.actual_fps
        
        return info
    
    def release(self):
        """Release all video resources."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        
        if self.writer is not None:
            self.writer.release()
            self.writer = None
        
        cv2.destroyAllWindows()
        
        elapsed = time.time() - self.start_time if self.start_time else 0
        print(f"\n📊 Processing complete:")
        print(f"   Frames processed: {self.frame_count}")
        print(f"   Total time: {elapsed:.1f}s")
        print(f"   Average FPS: {self.frame_count/elapsed:.1f}" if elapsed > 0 else "")


class FPSTracker:
    """Utility class for tracking FPS with smoothing."""
    
    def __init__(self, avg_frames: int = 30):
        """
        Initialize FPS tracker.
        
        Args:
            avg_frames: Number of frames to average for smooth FPS
        """
        self.avg_frames = avg_frames
        self.frame_times = []
        self.last_time = None
    
    def update(self) -> float:
        """
        Update FPS calculation with current frame.
        
        Returns:
            Current smoothed FPS
        """
        current_time = time.time()
        
        if self.last_time is not None:
            frame_time = current_time - self.last_time
            self.frame_times.append(frame_time)
            
            # Keep only recent frames
            if len(self.frame_times) > self.avg_frames:
                self.frame_times.pop(0)
        
        self.last_time = current_time
        
        return self.get_fps()
    
    def get_fps(self) -> float:
        """
        Get current smoothed FPS.
        
        Returns:
            Frames per second
        """
        if not self.frame_times:
            return 0.0
        
        avg_time = sum(self.frame_times) / len(self.frame_times)
        return 1.0 / avg_time if avg_time > 0 else 0.0
