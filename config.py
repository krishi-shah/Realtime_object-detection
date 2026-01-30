"""
Configuration settings for Real-Time Object Detection System
"""
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DetectionConfig:
    """Configuration for the object detection system."""
    
    # Model settings
    model_name: str = "yolov8m.pt"  # Options: yolov8n, yolov8s, yolov8m, yolov8l, yolov8x (m = balanced accuracy)
    confidence_threshold: float = 0.35  # Lower threshold catches more objects
    iou_threshold: float = 0.5  # Higher IOU for better duplicate filtering
    
    # Input settings
    input_source: str = "0"  # "0" for webcam, or path to video file
    frame_width: int = 1280
    frame_height: int = 720
    
    # Output settings
    output_dir: Path = field(default_factory=lambda: Path("output"))
    save_video: bool = True
    save_csv: bool = True
    video_fps: int = 30
    
    # Display settings
    show_display: bool = True
    show_fps: bool = True
    show_counts: bool = True
    box_thickness: int = 2
    font_scale: float = 0.6
    
    # Classes to detect (None = all classes)
    # Common COCO classes: person, car, truck, bus, motorcycle, bicycle, dog, cat, bird, etc.
    target_classes: Optional[List[str]] = None
    
    # Tracking settings
    enable_tracking: bool = True
    track_history_length: int = 30
    
    # Statistics settings
    stats_window_seconds: int = 60  # Rolling window for statistics
    
    def __post_init__(self):
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)


# Predefined configurations for specific use cases
class PresetConfigs:
    """Preset configurations for common detection scenarios."""
    
    @staticmethod
    def person_counter() -> DetectionConfig:
        """Configuration for crowd/person counting."""
        return DetectionConfig(
            model_name="yolov8n.pt",
            confidence_threshold=0.4,
            target_classes=["person"],
            enable_tracking=True,
        )
    
    @staticmethod
    def vehicle_tracker() -> DetectionConfig:
        """Configuration for traffic/vehicle monitoring."""
        return DetectionConfig(
            model_name="yolov8s.pt",
            confidence_threshold=0.5,
            target_classes=["car", "truck", "bus", "motorcycle", "bicycle"],
            enable_tracking=True,
        )
    
    @staticmethod
    def animal_detector() -> DetectionConfig:
        """Configuration for wildlife detection."""
        return DetectionConfig(
            model_name="yolov8m.pt",
            confidence_threshold=0.45,
            target_classes=["bird", "cat", "dog", "horse", "sheep", "cow", 
                          "elephant", "bear", "zebra", "giraffe"],
            enable_tracking=True,
        )
    
    @staticmethod
    def general_detector() -> DetectionConfig:
        """General-purpose object detection."""
        return DetectionConfig(
            model_name="yolov8n.pt",
            confidence_threshold=0.5,
            target_classes=None,  # Detect all classes
            enable_tracking=True,
        )


# COCO class names (80 classes that YOLOv8 can detect)
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush"
]

# Color palette for visualization (BGR format)
DETECTION_COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
    (0, 255, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0),
    (128, 0, 128), (0, 128, 128), (255, 128, 0), (255, 0, 128), (128, 255, 0),
    (0, 255, 128), (128, 0, 255), (0, 128, 255), (255, 128, 128), (128, 255, 128),
]
