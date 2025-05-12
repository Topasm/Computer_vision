#!/usr/bin/env python
# filepath: /home/ahrilab/Desktop/CV/Computer_vision/generate_json_labels.py
import os
import json
import argparse
from ultralytics import YOLO
import cv2
import numpy as np
from tqdm import tqdm


def create_json_label(image_path, predictions, confidence_threshold=0.25):
    """
    Create a JSON label file in the format required by the dataset

    Args:
        image_path: Path to the image
        predictions: YOLO model predictions
        confidence_threshold: Confidence threshold for predictions

    Returns:
        JSON data structure
    """
    # Read image to get dimensions
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not read image: {image_path}")
        return None

    image_height, image_width = img.shape[:2]
    image_name = os.path.basename(image_path)

    # Class mapping
    idx_to_class = {
        0: "standing",
        1: "sitting",
        2: "lying",
        3: "throwing"
    }

    # Create basic JSON structure
    label_data = {
        "version": "5.2.1",
        "flags": {},
        "shapes": [],
        "imagePath": image_name,
        "imageData": None,
        "imageHeight": image_height,
        "imageWidth": image_width
    }

    # Process predictions
    if len(predictions) > 0 and hasattr(predictions[0].boxes, 'xyxy'):
        boxes = predictions[0].boxes.xyxy.cpu().numpy()
        confs = predictions[0].boxes.conf.cpu().numpy()
        cls_ids = predictions[0].boxes.cls.cpu().numpy().astype(int)

        # Add each detection as a shape
        for box, conf, cls_id in zip(boxes, confs, cls_ids):
            if conf < confidence_threshold:
                continue

            # Get coordinates
            x1, y1, x2, y2 = box

            # Get class name
            class_name = idx_to_class.get(cls_id, f"class_{cls_id}")

            # Create shape entry
            shape = {
                "label": class_name,
                "points": [[float(x1), float(y1)], [float(x2), float(y2)]],
                "group_id": None,
                "description": f"confidence: {float(conf):.2f}",
                "shape_type": "rectangle",
                "flags": {}
            }

            label_data["shapes"].append(shape)

    return label_data


def generate_labels(model_path, image_dir, output_dir, confidence_threshold=0.25, device=''):
    """Generate JSON label files for all images in a directory using YOLO predictions"""
    # Load the model
    try:
        model = YOLO(model_path)
        print(f"Successfully loaded model from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get all images
    image_extensions = ['.jpg', '.jpeg', '.png']
    image_files = [
        os.path.join(image_dir, f) for f in os.listdir(image_dir)
        if any(f.lower().endswith(ext) for ext in image_extensions)
    ]

    if not image_files:
        print(f"No images found in {image_dir}")
        return False

    print(f"Found {len(image_files)} images")

    # Process each image
    for image_path in tqdm(image_files, desc="Generating labels"):
        # Get predictions
        try:
            results = model.predict(
                image_path, conf=confidence_threshold, device=device, verbose=False
            )

            # Create JSON label
            label_data = create_json_label(
                image_path, results, confidence_threshold)

            if label_data:
                # Save JSON label
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                output_path = os.path.join(output_dir, f"{base_name}.json")

                with open(output_path, 'w') as f:
                    json.dump(label_data, f, indent=4)

        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            continue

    print(
        f"Label generation complete. {len(os.listdir(output_dir))} files created in {output_dir}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Generate JSON label files from YOLO predictions")
    parser.add_argument("--model", type=str, default="yolo_runs/run12/weights/best.pt",
                        help="Path to YOLO model")
    parser.add_argument("--image-dir", type=str, default="CV_Test/Images",
                        help="Directory containing images")
    parser.add_argument("--output-dir", type=str, default="CV_Test/Labels",
                        help="Directory to save JSON label files")
    parser.add_argument("--confidence", type=float, default=0.25,
                        help="Confidence threshold for predictions")
    parser.add_argument("--device", type=str, default='',
                        help="Device to use for inference (empty for auto)")

    args = parser.parse_args()

    # Check if model exists
    if not os.path.exists(args.model):
        print(f"Model not found at: {args.model}")
        return

    # Check if image directory exists
    if not os.path.exists(args.image_dir):
        print(f"Image directory not found: {args.image_dir}")
        return

    # Generate labels
    print(f"Generating JSON labels using model: {args.model}")
    print(f"Image directory: {args.image_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Confidence threshold: {args.confidence}")
    print(f"Device: {args.device or 'auto'}")

    success = generate_labels(
        model_path=args.model,
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        confidence_threshold=args.confidence,
        device=args.device
    )

    if success:
        print("\nLabel generation complete!")
        print(f"Labels saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
