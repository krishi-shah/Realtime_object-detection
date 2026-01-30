"""
YOLOv8-based Object Detector
"""
import cv2
import numpy as np
from ultralytics import YOLO
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict

from config import DetectionConfig, DETECTION_COLORS, COCO_CLASSES


@dataclass
class Detection:
    """Represents a single object detection."""
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    track_id: Optional[int] = None


class ObjectDetector:
    """Real-time object detector using YOLOv8."""
    
    def __init__(self, config: DetectionConfig):
        """Initialize the detector with given configuration."""
        self.config = config
        self.model = self._load_model()
        self.track_history: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
        self.class_colors = self._assign_colors()
        
        # Filter class indices if specific classes are targeted
        self.target_class_ids = None
        if config.target_classes:
            self.target_class_ids = [
                COCO_CLASSES.index(cls) for cls in config.target_classes 
                if cls in COCO_CLASSES
            ]
    
    def _load_model(self) -> YOLO:
        """Load the YOLOv8 model."""
        print(f"🔄 Loading YOLOv8 model: {self.config.model_name}")
        model = YOLO(self.config.model_name)
        print(f"✅ Model loaded successfully!")
        return model
    
    def _assign_colors(self) -> Dict[int, Tuple[int, int, int]]:
        """Assign unique colors to each class."""
        colors = {}
        for i in range(len(COCO_CLASSES)):
            colors[i] = DETECTION_COLORS[i % len(DETECTION_COLORS)]
        return colors
    
    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess frame for better detection accuracy.
        
        Args:
            frame: BGR image as numpy array
            
        Returns:
            Enhanced frame
        """
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) for better contrast
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # Light denoising (preserves edges while reducing noise)
        enhanced = cv2.bilateralFilter(enhanced, 5, 50, 50)
        
        return enhanced
    
    def detect(self, frame: np.ndarray, enhance: bool = False) -> List[Detection]:
        """
        Perform object detection on a single frame.
        
        Args:
            frame: BGR image as numpy array
            enhance: Whether to apply image enhancement (can slow down processing)
            
        Returns:
            List of Detection objects
        """
        detections = []
        
        # Optionally preprocess frame for better detection in low-light conditions
        processed_frame = self._preprocess_frame(frame) if enhance else frame
        
        # Run inference with optimized parameters for better accuracy
        if self.config.enable_tracking:
            results = self.model.track(
                processed_frame,
                persist=True,
                conf=self.config.confidence_threshold,
                iou=self.config.iou_threshold,
                imgsz=640,  # Optimal input size for accuracy
                verbose=False
            )
        else:
            results = self.model(
                processed_frame,
                conf=self.config.confidence_threshold,
                iou=self.config.iou_threshold,
                imgsz=640,  # Optimal input size for accuracy
                verbose=False
            )
        
        # Process results
        for result in results:
            boxes = result.boxes
            
            if boxes is None or len(boxes) == 0:
                continue
            
            for i, box in enumerate(boxes):
                class_id = int(box.cls[0])
                
                # Skip if not in target classes
                if self.target_class_ids and class_id not in self.target_class_ids:
                    continue
                
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Get track ID if tracking is enabled
                track_id = None
                if self.config.enable_tracking and box.id is not None:
                    track_id = int(box.id[0])
                    
                    # Update track history
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    self.track_history[track_id].append((center_x, center_y))
                    
                    # Limit history length
                    if len(self.track_history[track_id]) > self.config.track_history_length:
                        self.track_history[track_id].pop(0)
                
                detection = Detection(
                    class_id=class_id,
                    class_name=COCO_CLASSES[class_id],
                    confidence=confidence,
                    bbox=(x1, y1, x2, y2),
                    track_id=track_id
                )
                detections.append(detection)
        
        return detections
    
    def draw_detections(self, frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
        """
        Draw detection boxes and labels on the frame.
        
        Args:
            frame: BGR image as numpy array
            detections: List of Detection objects
            
        Returns:
            Annotated frame
        """
        annotated = frame.copy()
        
        for det in detections:
            color = self.class_colors[det.class_id]
            x1, y1, x2, y2 = det.bbox
            
            # Draw bounding box
            cv2.rectangle(
                annotated, (x1, y1), (x2, y2), 
                color, self.config.box_thickness
            )
            
            # Prepare label
            label = f"{det.class_name}"
            if det.track_id is not None:
                label = f"ID:{det.track_id} {label}"
            label += f" {det.confidence:.2f}"
            
            # Draw label background
            (label_w, label_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 
                self.config.font_scale, 2
            )
            cv2.rectangle(
                annotated, 
                (x1, y1 - label_h - 10), 
                (x1 + label_w + 5, y1),
                color, -1
            )
            
            # Draw label text
            cv2.putText(
                annotated, label,
                (x1 + 2, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.config.font_scale,
                (255, 255, 255), 2
            )
            
            # Draw tracking trail
            if det.track_id is not None and det.track_id in self.track_history:
                track = self.track_history[det.track_id]
                if len(track) > 1:
                    points = np.array(track, dtype=np.int32).reshape((-1, 1, 2))
                    cv2.polylines(
                        annotated, [points], False, color, 2
                    )
        
        return annotated
    
    def draw_stats(self, frame: np.ndarray, detections: List[Detection], 
                   fps: float, class_counts: Dict[str, int]) -> np.ndarray:
        """
        Draw statistics overlay on the frame.
        
        Args:
            frame: BGR image
            detections: Current frame detections
            fps: Current FPS
            class_counts: Cumulative class counts
            
        Returns:
            Frame with stats overlay
        """
        annotated = frame.copy()
        h, w = annotated.shape[:2]
        
        # Stats panel background
        panel_height = 40 + len(class_counts) * 25
        overlay = annotated.copy()
        cv2.rectangle(overlay, (10, 10), (280, panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, annotated, 0.3, 0, annotated)
        
        # Draw FPS
        if self.config.show_fps:
            cv2.putText(
                annotated, f"FPS: {fps:.1f}",
                (20, 35), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0), 2
            )
        
        # Draw object counts
        if self.config.show_counts:
            y_offset = 65
            for class_name, count in sorted(class_counts.items()):
                text = f"{class_name}: {count}"
                cv2.putText(
                    annotated, text,
                    (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2
                )
                y_offset += 25
        
        # Current frame detection count
        current_count = len(detections)
        cv2.putText(
            annotated, f"Objects in frame: {current_count}",
            (w - 250, 35), cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 255, 255), 2
        )
        
        return annotated
    
    def cleanup(self):
        """Cleanup resources."""
        self.track_history.clear()
