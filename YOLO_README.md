# YOLO Action Detection

This project implements action detection (standing, sitting, lying, throwing) using YOLOv8 and the Ultralytics library.

## Setup

1. Install the required packages:
   ```
   pip install ultralytics opencv-python matplotlib torch wandb
   ```

2. Convert the dataset to YOLO format:
   ```
   python convert_to_yolo.py
   ```

3. Train the YOLO model:
   ```
   python train_yolo.py --model yolov8n.pt --epochs 50
   ```

4. Evaluate the trained model:
   ```
   python generate_json_labels.py
   ```
