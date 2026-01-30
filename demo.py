"""
Demo Script: Shows how to use the Object Detection System programmatically

Run this script to see the system in action with your webcam:
    python demo.py
"""
import cv2
import time
from config import DetectionConfig, PresetConfigs
from detector import ObjectDetector
from video_processor import VideoProcessor, FPSTracker
from statistics import StatisticsTracker


def demo_webcam_detection():
    """
    Demo: Real-time webcam object detection
    """
    print("🎬 Demo: Real-time Webcam Object Detection")
    print("=" * 50)
    
    # Create configuration
    config = DetectionConfig(
        model_name="yolov8n.pt",  # Fast model for real-time
        confidence_threshold=0.5,
        input_source="0",  # Webcam
        show_display=True,
        save_video=False,  # Don't save for demo
        save_csv=False,
        enable_tracking=True
    )
    
    # Initialize components
    detector = ObjectDetector(config)
    video_proc = VideoProcessor(config)
    fps_tracker = FPSTracker()
    stats = StatisticsTracker(config)
    
    # Open webcam
    if not video_proc.open():
        print("❌ Could not open webcam")
        return
    
    print("\n🎥 Detection running... Press 'q' to quit\n")
    
    try:
        for frame in video_proc.frames():
            # Detect objects
            detections = detector.detect(frame)
            
            # Update stats
            frame_counts = stats.update(video_proc.frame_count, detections)
            
            # Calculate FPS
            current_fps = fps_tracker.update()
            
            # Draw on frame
            annotated = detector.draw_detections(frame, detections)
            annotated = detector.draw_stats(annotated, detections, current_fps, frame_counts)
            
            # Show frame
            key = video_proc.display_frame(annotated, "Object Detection Demo")
            
            # Quit on 'q'
            if key == ord('q'):
                break
    
    finally:
        video_proc.release()
        detector.cleanup()
        
        # Print summary
        summary = stats.get_summary()
        print("\n📊 Session Summary:")
        print(f"   Total detections: {summary['total_detections']}")
        print(f"   Unique objects tracked: {summary['unique_tracked_objects']}")
        print(f"   Classes detected: {list(summary['class_counts'].keys())}")


def demo_person_counting():
    """
    Demo: Person counting mode
    """
    print("🎬 Demo: Person Counter")
    print("=" * 50)
    
    # Use person counter preset
    config = PresetConfigs.person_counter()
    config.save_video = False
    config.save_csv = False
    
    detector = ObjectDetector(config)
    video_proc = VideoProcessor(config)
    fps_tracker = FPSTracker()
    
    if not video_proc.open():
        print("❌ Could not open webcam")
        return
    
    print("\n🎥 Counting people... Press 'q' to quit\n")
    unique_persons = set()
    
    try:
        for frame in video_proc.frames():
            detections = detector.detect(frame)
            
            # Count unique persons
            for det in detections:
                if det.track_id is not None:
                    unique_persons.add(det.track_id)
            
            current_fps = fps_tracker.update()
            
            # Draw
            annotated = detector.draw_detections(frame, detections)
            
            # Custom overlay for person count
            cv2.rectangle(annotated, (10, 10), (300, 80), (0, 0, 0), -1)
            cv2.putText(
                annotated, f"People in frame: {len(detections)}",
                (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
            )
            cv2.putText(
                annotated, f"Total unique: {len(unique_persons)}",
                (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )
            
            key = video_proc.display_frame(annotated, "Person Counter Demo")
            if key == ord('q'):
                break
    
    finally:
        video_proc.release()
        detector.cleanup()
        print(f"\n📊 Total unique persons detected: {len(unique_persons)}")


def demo_custom_classes():
    """
    Demo: Detect only specific classes
    """
    print("🎬 Demo: Custom Class Detection (Electronics)")
    print("=" * 50)
    
    config = DetectionConfig(
        model_name="yolov8n.pt",
        confidence_threshold=0.4,
        target_classes=["cell phone", "laptop", "keyboard", "mouse", "remote", "tv"],
        save_video=False,
        save_csv=False
    )
    
    detector = ObjectDetector(config)
    video_proc = VideoProcessor(config)
    fps_tracker = FPSTracker()
    
    if not video_proc.open():
        print("❌ Could not open webcam")
        return
    
    print(f"\n🎯 Looking for: {config.target_classes}")
    print("🎥 Point camera at electronics... Press 'q' to quit\n")
    
    try:
        for frame in video_proc.frames():
            detections = detector.detect(frame)
            current_fps = fps_tracker.update()
            
            annotated = detector.draw_detections(frame, detections)
            
            # Show FPS and detection count
            cv2.putText(
                annotated, f"FPS: {current_fps:.1f} | Found: {len(detections)} electronics",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )
            
            key = video_proc.display_frame(annotated, "Electronics Detector Demo")
            if key == ord('q'):
                break
    
    finally:
        video_proc.release()
        detector.cleanup()


def main():
    """Main demo menu."""
    print("\n" + "="*60)
    print("       🎯 OBJECT DETECTION SYSTEM - DEMOS")
    print("="*60)
    print("\nSelect a demo to run:")
    print("  1. General object detection (webcam)")
    print("  2. Person counter")
    print("  3. Electronics detector")
    print("  q. Quit")
    print()
    
    while True:
        choice = input("Enter choice (1/2/3/q): ").strip().lower()
        
        if choice == '1':
            demo_webcam_detection()
        elif choice == '2':
            demo_person_counting()
        elif choice == '3':
            demo_custom_classes()
        elif choice == 'q':
            print("👋 Goodbye!")
            break
        else:
            print("Invalid choice. Try again.")
        
        print()


if __name__ == "__main__":
    main()
