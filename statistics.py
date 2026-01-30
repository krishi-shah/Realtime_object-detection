"""
Statistics Tracking and CSV Logging Module
"""
import csv
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from collections import defaultdict
from dataclasses import dataclass, field

from config import DetectionConfig


@dataclass
class FrameStats:
    """Statistics for a single frame."""
    timestamp: datetime
    frame_number: int
    detections: List[dict]
    total_objects: int
    class_counts: Dict[str, int]


class StatisticsTracker:
    """Tracks detection statistics over time."""
    
    def __init__(self, config: DetectionConfig):
        """Initialize statistics tracker."""
        self.config = config
        self.start_time = datetime.now()
        
        # Counters
        self.total_detections = 0
        self.unique_tracks: set = set()
        self.class_counts: Dict[str, int] = defaultdict(int)
        self.frame_stats: List[FrameStats] = []
        
        # Time-series data
        self.detections_per_second: List[tuple] = []  # (timestamp, count)
        
    def update(self, frame_number: int, detections: list) -> Dict[str, int]:
        """
        Update statistics with new detections.
        
        Args:
            frame_number: Current frame number
            detections: List of Detection objects
            
        Returns:
            Current class counts
        """
        current_time = datetime.now()
        frame_class_counts = defaultdict(int)
        frame_detections = []
        
        for det in detections:
            # Update class counts
            frame_class_counts[det.class_name] += 1
            self.class_counts[det.class_name] += 1
            self.total_detections += 1
            
            # Track unique objects
            if det.track_id is not None:
                self.unique_tracks.add(det.track_id)
            
            # Store detection data
            frame_detections.append({
                "class_name": det.class_name,
                "confidence": det.confidence,
                "bbox": det.bbox,
                "track_id": det.track_id
            })
        
        # Store frame statistics
        stats = FrameStats(
            timestamp=current_time,
            frame_number=frame_number,
            detections=frame_detections,
            total_objects=len(detections),
            class_counts=dict(frame_class_counts)
        )
        self.frame_stats.append(stats)
        
        # Track detections per second
        self.detections_per_second.append((current_time, len(detections)))
        
        return dict(frame_class_counts)
    
    def get_current_counts(self) -> Dict[str, int]:
        """Get current class counts from recent frames."""
        if not self.frame_stats:
            return {}
        
        # Return counts from last frame
        return self.frame_stats[-1].class_counts
    
    def get_cumulative_counts(self) -> Dict[str, int]:
        """Get cumulative class counts."""
        return dict(self.class_counts)
    
    def get_unique_object_count(self) -> int:
        """Get number of unique tracked objects."""
        return len(self.unique_tracks)
    
    def get_summary(self) -> dict:
        """
        Generate summary statistics.
        
        Returns:
            Dictionary with summary statistics
        """
        elapsed = (datetime.now() - self.start_time).total_seconds()
        
        return {
            "start_time": self.start_time.isoformat(),
            "duration_seconds": elapsed,
            "total_frames": len(self.frame_stats),
            "total_detections": self.total_detections,
            "unique_tracked_objects": len(self.unique_tracks),
            "class_counts": dict(self.class_counts),
            "avg_detections_per_frame": (
                self.total_detections / len(self.frame_stats) 
                if self.frame_stats else 0
            ),
            "avg_detections_per_second": (
                self.total_detections / elapsed if elapsed > 0 else 0
            )
        }
    
    def get_class_distribution(self) -> pd.DataFrame:
        """
        Get class distribution as DataFrame.
        
        Returns:
            DataFrame with class statistics
        """
        data = []
        total = sum(self.class_counts.values())
        
        for class_name, count in sorted(
            self.class_counts.items(), 
            key=lambda x: x[1], 
            reverse=True
        ):
            percentage = (count / total * 100) if total > 0 else 0
            data.append({
                "class": class_name,
                "count": count,
                "percentage": round(percentage, 2)
            })
        
        return pd.DataFrame(data)


class CSVLogger:
    """Logs detection data to CSV files."""
    
    def __init__(self, config: DetectionConfig):
        """Initialize CSV logger."""
        self.config = config
        self.csv_path: Optional[Path] = None
        self.csv_file = None
        self.csv_writer = None
        self.row_count = 0
        
    def setup(self) -> bool:
        """
        Setup CSV logging.
        
        Returns:
            True if setup successful
        """
        if not self.config.save_csv:
            return False
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = self.config.output_dir / f"detections_{timestamp}.csv"
        
        try:
            self.csv_file = open(self.csv_path, 'w', newline='', encoding='utf-8')
            self.csv_writer = csv.writer(self.csv_file)
            
            # Write header
            self.csv_writer.writerow([
                "timestamp",
                "frame_number",
                "class_name",
                "confidence",
                "track_id",
                "x1", "y1", "x2", "y2",
                "bbox_width", "bbox_height"
            ])
            
            print(f"📝 Logging detections to: {self.csv_path}")
            return True
            
        except Exception as e:
            print(f"⚠️ Failed to setup CSV logging: {e}")
            return False
    
    def log_detections(self, frame_number: int, detections: list):
        """
        Log detections for a frame.
        
        Args:
            frame_number: Current frame number
            detections: List of Detection objects
        """
        if self.csv_writer is None:
            return
        
        timestamp = datetime.now().isoformat()
        
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            bbox_width = x2 - x1
            bbox_height = y2 - y1
            
            self.csv_writer.writerow([
                timestamp,
                frame_number,
                det.class_name,
                round(det.confidence, 4),
                det.track_id if det.track_id is not None else "",
                x1, y1, x2, y2,
                bbox_width, bbox_height
            ])
            self.row_count += 1
        
        # Flush periodically
        if self.row_count % 100 == 0:
            self.csv_file.flush()
    
    def close(self):
        """Close CSV file."""
        if self.csv_file is not None:
            self.csv_file.close()
            self.csv_file = None
            self.csv_writer = None
            print(f"📊 CSV log saved: {self.row_count} detections recorded")


class SummaryReporter:
    """Generates summary reports."""
    
    def __init__(self, config: DetectionConfig):
        """Initialize summary reporter."""
        self.config = config
    
    def generate_report(self, stats: StatisticsTracker, video_info: dict) -> str:
        """
        Generate a text summary report.
        
        Args:
            stats: StatisticsTracker instance
            video_info: Video information dictionary
            
        Returns:
            Report as string
        """
        summary = stats.get_summary()
        
        report = []
        report.append("=" * 60)
        report.append("        OBJECT DETECTION SUMMARY REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Video info
        report.append("📹 VIDEO INFORMATION")
        report.append("-" * 40)
        report.append(f"  Source: {video_info.get('source', 'N/A')}")
        report.append(f"  Resolution: {video_info.get('width', 0)}x{video_info.get('height', 0)}")
        report.append(f"  FPS: {video_info.get('fps', 0):.1f}")
        report.append(f"  Frames processed: {video_info.get('frames_processed', 0)}")
        report.append("")
        
        # Detection stats
        report.append("🎯 DETECTION STATISTICS")
        report.append("-" * 40)
        report.append(f"  Total detections: {summary['total_detections']}")
        report.append(f"  Unique tracked objects: {summary['unique_tracked_objects']}")
        report.append(f"  Avg detections/frame: {summary['avg_detections_per_frame']:.2f}")
        report.append(f"  Avg detections/second: {summary['avg_detections_per_second']:.2f}")
        report.append(f"  Processing time: {summary['duration_seconds']:.1f}s")
        report.append("")
        
        # Class breakdown
        report.append("📊 CLASS BREAKDOWN")
        report.append("-" * 40)
        
        class_df = stats.get_class_distribution()
        if not class_df.empty:
            for _, row in class_df.iterrows():
                report.append(
                    f"  {row['class']:20s} {row['count']:6d} ({row['percentage']:5.1f}%)"
                )
        else:
            report.append("  No detections recorded")
        
        report.append("")
        report.append("=" * 60)
        
        return "\n".join(report)
    
    def save_report(self, stats: StatisticsTracker, video_info: dict):
        """
        Save summary report to file.
        
        Args:
            stats: StatisticsTracker instance
            video_info: Video information dictionary
        """
        report = self.generate_report(stats, video_info)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.config.output_dir / f"report_{timestamp}.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"📄 Report saved to: {report_path}")
        
        # Also save as JSON
        import json
        json_path = self.config.output_dir / f"stats_{timestamp}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(stats.get_summary(), f, indent=2, default=str)
        
        print(f"📋 JSON stats saved to: {json_path}")
        
        return report
