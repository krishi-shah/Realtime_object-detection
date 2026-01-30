# 🎯 Real-Time Object Detection System

A powerful, free, and open-source real-time object detection system using **YOLOv8**, **OpenCV**, and **pandas**. Detect, track, and analyze objects from your webcam or video files with beautiful visualizations and comprehensive statistics.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-green.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-red.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## ✨ Features

- **🔴 Real-Time Detection**: Process webcam feed or video files with live object detection
- **📊 Object Tracking**: Track objects across frames with unique IDs and movement trails
- **📈 Statistics**: Real-time counts, class distributions, and detection analytics
- **💾 Data Export**: CSV logs and JSON statistics for further analysis
- **🎬 Video Output**: Save annotated videos with detection overlays
- **🎯 Class Filtering**: Detect specific object classes (person, car, etc.)
- **⚙️ Preset Modes**: Pre-configured settings for common use cases

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or navigate to the project
cd real-object-detection

# Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 2. Basic Usage

```bash
# Webcam detection (default)
python main.py

# Process a video file
python main.py --source path/to/video.mp4

# Use a specific preset
python main.py --preset person    # Person counting
python main.py --preset vehicle   # Vehicle tracking
python main.py --preset animal    # Animal detection
```

### 3. Controls

| Key | Action |
|-----|--------|
| `q` / `ESC` | Quit detection |
| `s` | Save screenshot |
| `p` | Pause/Resume |

## 📖 Usage Examples

### Person Counter (Crowd Monitoring)
```bash
python main.py --preset person --source crowd_video.mp4
```

### Vehicle Tracker (Traffic Monitoring)
```bash
python main.py --preset vehicle --source traffic.mp4
```

### Custom Class Detection
```bash
# Detect only specific classes
python main.py --classes person car truck bicycle

# List all available classes
python main.py --list-classes
```

### High Accuracy Mode
```bash
# Use larger model with higher confidence threshold
python main.py --model yolov8m.pt --conf 0.6
```

### Headless Mode (Server/No Display)
```bash
python main.py --source video.mp4 --no-display
```

### Custom Output Directory
```bash
python main.py --output my_results/
```

## ⚙️ Command Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--source`, `-s` | `0` | Video source (webcam index or file path) |
| `--output`, `-o` | `output/` | Output directory for saved files |
| `--model`, `-m` | `yolov8n.pt` | YOLOv8 model variant |
| `--conf` | `0.5` | Confidence threshold (0.0-1.0) |
| `--iou` | `0.45` | IOU threshold for NMS |
| `--preset`, `-p` | - | Preset config: person, vehicle, animal, general |
| `--classes`, `-c` | - | Specific classes to detect |
| `--width` | `1280` | Frame width (webcam) |
| `--height` | `720` | Frame height (webcam) |
| `--no-display` | - | Run without display window |
| `--no-tracking` | - | Disable object tracking |
| `--no-save` | - | Don't save output video |
| `--no-csv` | - | Don't save CSV log |
| `--list-classes` | - | List all 80 COCO classes |

## 🧠 YOLOv8 Models

| Model | Size | Speed | Accuracy | Use Case |
|-------|------|-------|----------|----------|
| `yolov8n.pt` | 6 MB | ⚡⚡⚡⚡⚡ | ★★☆☆☆ | Real-time, edge devices |
| `yolov8s.pt` | 22 MB | ⚡⚡⚡⚡ | ★★★☆☆ | Balanced performance |
| `yolov8m.pt` | 50 MB | ⚡⚡⚡ | ★★★★☆ | Higher accuracy |
| `yolov8l.pt` | 84 MB | ⚡⚡ | ★★★★★ | High accuracy |
| `yolov8x.pt` | 131 MB | ⚡ | ★★★★★ | Maximum accuracy |

## 📊 Output Files

The system generates several output files in the output directory:

```
output/
├── detection_20260121_143022.mp4    # Annotated video
├── detections_20260121_143022.csv   # Detection log
├── report_20260121_143022.txt       # Summary report
└── stats_20260121_143022.json       # Statistics JSON
```

### CSV Log Format
```csv
timestamp,frame_number,class_name,confidence,track_id,x1,y1,x2,y2,bbox_width,bbox_height
2026-01-21T14:30:22,1,person,0.8523,1,100,200,300,400,200,200
```

### JSON Statistics
```json
{
  "total_detections": 1523,
  "unique_tracked_objects": 47,
  "class_counts": {"person": 892, "car": 631},
  "avg_detections_per_frame": 5.2
}
```

## 🎯 Available Classes (COCO Dataset)

The system can detect 80 object classes:

**People & Animals**: person, bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe

**Vehicles**: bicycle, car, motorcycle, airplane, bus, train, truck, boat

**Outdoor**: traffic light, fire hydrant, stop sign, parking meter, bench

**Sports**: frisbee, skis, snowboard, sports ball, kite, baseball bat, baseball glove, skateboard, surfboard, tennis racket

**Kitchen**: bottle, wine glass, cup, fork, knife, spoon, bowl

**Food**: banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake

**Furniture**: chair, couch, potted plant, bed, dining table

**Electronics**: tv, laptop, mouse, remote, keyboard, cell phone

**Appliances**: microwave, oven, toaster, sink, refrigerator

**Other**: book, clock, vase, scissors, teddy bear, hair drier, toothbrush, toilet, backpack, umbrella, handbag, tie, suitcase

## 🔧 Custom Training (Optional)

For specialized detection (e.g., pothole detection), you can train a custom model:

```python
from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n.pt')

# Train on custom dataset (e.g., Kaggle pothole dataset)
results = model.train(
    data='path/to/data.yaml',
    epochs=100,
    imgsz=640
)

# Use your trained model
# python main.py --model path/to/best.pt
```

### Free Datasets on Kaggle

- [Pothole Detection Dataset](https://www.kaggle.com/datasets/atulyakumar98/pothole-detection-dataset)
- [COCO 2017 Dataset](https://www.kaggle.com/datasets/awsaf49/coco-2017-dataset)
- [Self Driving Car Dataset](https://www.kaggle.com/datasets/sshikamaru/udacity-self-driving-car-dataset)

## 🖥️ System Requirements

- **Python**: 3.8+
- **RAM**: 4GB minimum, 8GB+ recommended
- **GPU**: Optional but recommended for better performance
- **Webcam**: For live detection (optional)

### GPU Support (Optional)

For NVIDIA GPU acceleration:

```bash
# Install PyTorch with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## 🐛 Troubleshooting

### Webcam not detected
```bash
# Try different camera index
python main.py --source 1
python main.py --source 2
```

### Low FPS
- Use a smaller model: `--model yolov8n.pt`
- Lower resolution: `--width 640 --height 480`
- Disable tracking: `--no-tracking`

### Out of memory
- Use smaller model
- Process video in segments
- Reduce frame resolution

## 📄 License

This project uses free and open-source components:
- **YOLOv8**: AGPL-3.0 License (Ultralytics)
- **OpenCV**: Apache 2.0 License
- **PyTorch**: BSD License

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest features
- Submit pull requests

## 📚 Resources

- [Ultralytics YOLOv8 Documentation](https://docs.ultralytics.com/)
- [OpenCV Documentation](https://docs.opencv.org/)
- [COCO Dataset](https://cocodataset.org/)

---
