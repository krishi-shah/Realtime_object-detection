"""
Real-Time Object Detection System
Main entry point with CLI interface

Usage:
    python main.py                          # Webcam with default settings
    python main.py --source video.mp4       # Process video file
    python main.py --preset vehicle         # Use vehicle tracking preset
    python main.py --classes person car     # Detect only specific classes
"""
import argparse
import sys
from pathlib import Path

from config import DetectionConfig, PresetConfigs, COCO_CLASSES
from detector import ObjectDetector
from video_processor import VideoProcessor, FPSTracker
from statistics import StatisticsTracker, CSVLogger, SummaryReporter


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Real-Time Object Detection using YOLOv8",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                              # Webcam detection
  python main.py --source 1                   # Use webcam index 1
  python main.py --source video.mp4           # Process video file
  python main.py --preset person              # Person counting mode
  python main.py --preset vehicle             # Vehicle tracking mode
  python main.py --preset animal              # Animal detection mode
  python main.py --classes person car truck   # Custom classes
  python main.py --model yolov8s.pt           # Use larger model
  python main.py --conf 0.6                   # Higher confidence threshold
  python main.py --no-display                 # Headless mode (no window)
  python main.py --no-save                    # Don't save output video
        """
    )
    
    # Input/Output
    parser.add_argument(
        "--source", "-s",
        type=str,
        default="0",
        help="Video source: webcam index (0, 1, ...) or video file path"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="output",
        help="Output directory for saved files"
    )
    
    # Model settings
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="yolov8m.pt",
        choices=["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"],
        help="YOLOv8 model to use (n=nano, s=small, m=medium, l=large, x=extra-large)"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.35,
        help="Confidence threshold (0.0-1.0) - lower = more detections"
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.5,
        help="IOU threshold for NMS (0.0-1.0)"
    )
    
    # Preset configurations
    parser.add_argument(
        "--preset", "-p",
        type=str,
        choices=["person", "vehicle", "animal", "general"],
        help="Use preset configuration for specific detection scenarios"
    )
    
    # Class filtering
    parser.add_argument(
        "--classes", "-c",
        type=str,
        nargs="+",
        help="Classes to detect (e.g., --classes person car truck)"
    )
    parser.add_argument(
        "--list-classes",
        action="store_true",
        help="List all available classes and exit"
    )
    
    # Display options
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Run without display window (headless mode)"
    )
    parser.add_argument(
        "--no-tracking",
        action="store_true",
        help="Disable object tracking"
    )
    
    # Save options
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save output video"
    )
    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="Don't save detection CSV log"
    )
    
    # Resolution
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Frame width (for webcam)"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=720,
        help="Frame height (for webcam)"
    )
    
    return parser.parse_args()


def create_config(args) -> DetectionConfig:
    """Create configuration from arguments."""
    
    # Start with preset if specified
    if args.preset:
        preset_map = {
            "person": PresetConfigs.person_counter,
            "vehicle": PresetConfigs.vehicle_tracker,
            "animal": PresetConfigs.animal_detector,
            "general": PresetConfigs.general_detector,
        }
        config = preset_map[args.preset]()
        print(f"📋 Using preset configuration: {args.preset}")
    else:
        config = DetectionConfig()
    
    # Override with command line arguments
    config.input_source = args.source
    config.output_dir = Path(args.output)
    config.model_name = args.model
    config.confidence_threshold = args.conf
    config.iou_threshold = args.iou
    config.show_display = not args.no_display
    config.enable_tracking = not args.no_tracking
    config.save_video = not args.no_save
    config.save_csv = not args.no_csv
    config.frame_width = args.width
    config.frame_height = args.height
    
    # Override classes if specified
    if args.classes:
        valid_classes = [c for c in args.classes if c in COCO_CLASSES]
        invalid_classes = [c for c in args.classes if c not in COCO_CLASSES]
        
        if invalid_classes:
            print(f"⚠️ Unknown classes ignored: {invalid_classes}")
        
        if valid_classes:
            config.target_classes = valid_classes
            print(f"🎯 Detecting classes: {valid_classes}")
    
    return config


def run_detection(config: DetectionConfig):
    """
    Run the object detection pipeline.
    
    Args:
        config: Detection configuration
    """
    print("\n" + "="*60)
    print("     🎯 REAL-TIME OBJECT DETECTION SYSTEM")
    print("="*60 + "\n")
    
    # Initialize components
    detector = ObjectDetector(config)
    video_proc = VideoProcessor(config)
    stats = StatisticsTracker(config)
    csv_logger = CSVLogger(config)
    fps_tracker = FPSTracker()
    reporter = SummaryReporter(config)
    
    # Open video source
    if not video_proc.open():
        print("❌ Failed to open video source. Exiting.")
        return
    
    # Setup outputs
    video_proc.setup_writer()
    csv_logger.setup()
    
    print("\n🚀 Starting detection... (Press 'q' to quit)\n")
    
    try:
        for frame in video_proc.frames():
            # Perform detection
            detections = detector.detect(frame)
            
            # Update statistics
            frame_counts = stats.update(video_proc.frame_count, detections)
            
            # Log to CSV
            csv_logger.log_detections(video_proc.frame_count, detections)
            
            # Update FPS
            current_fps = fps_tracker.update()
            
            # Draw annotations
            annotated = detector.draw_detections(frame, detections)
            annotated = detector.draw_stats(
                annotated, detections, current_fps, frame_counts
            )
            
            # Add progress bar for video files
            progress = video_proc.get_progress()
            if progress is not None:
                h, w = annotated.shape[:2]
                bar_width = int(w * progress / 100)
                import cv2
                cv2.rectangle(annotated, (0, h-5), (bar_width, h), (0, 255, 0), -1)
                cv2.rectangle(annotated, (0, h-5), (w, h), (100, 100, 100), 1)
            
            # Save frame
            video_proc.write_frame(annotated)
            
            # Display
            key = video_proc.display_frame(annotated)
            
            # Handle keypresses
            if key == ord('q') or key == 27:  # q or ESC
                print("\n⏹️ Detection stopped by user")
                break
            elif key == ord('s'):  # Screenshot
                screenshot_path = config.output_dir / f"screenshot_{video_proc.frame_count}.jpg"
                import cv2
                cv2.imwrite(str(screenshot_path), annotated)
                print(f"📸 Screenshot saved: {screenshot_path}")
            elif key == ord('p'):  # Pause
                print("⏸️ Paused. Press any key to continue...")
                import cv2
                cv2.waitKey(0)
    
    except KeyboardInterrupt:
        print("\n⏹️ Detection interrupted")
    
    finally:
        # Cleanup
        video_proc.release()
        csv_logger.close()
        detector.cleanup()
        
        # Generate report
        video_info = video_proc.get_video_info()
        report = reporter.save_report(stats, video_info)
        
        # Print summary
        print("\n" + report)


def list_available_classes():
    """Print all available COCO classes."""
    print("\n📋 Available COCO Classes (80 classes):")
    print("=" * 50)
    
    for i, cls in enumerate(COCO_CLASSES):
        print(f"  {i+1:2d}. {cls}")
    
    print("\n💡 Usage example: python main.py --classes person car truck")


def main():
    """Main entry point."""
    args = parse_args()
    
    # Handle list-classes flag
    if args.list_classes:
        list_available_classes()
        return
    
    # Create configuration
    config = create_config(args)
    
    # Print configuration summary
    print("\n📝 Configuration:")
    print(f"   Model: {config.model_name}")
    print(f"   Source: {config.input_source}")
    print(f"   Confidence: {config.confidence_threshold}")
    print(f"   Tracking: {'Enabled' if config.enable_tracking else 'Disabled'}")
    print(f"   Display: {'Enabled' if config.show_display else 'Disabled'}")
    print(f"   Save video: {'Yes' if config.save_video else 'No'}")
    print(f"   Save CSV: {'Yes' if config.save_csv else 'No'}")
    
    if config.target_classes:
        print(f"   Target classes: {config.target_classes}")
    else:
        print(f"   Target classes: All (80 COCO classes)")
    
    # Run detection
    run_detection(config)


if __name__ == "__main__":
    main()
