import os
import argparse
from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from pathlib import Path
import torch


def draw_predictions(image, results, confidence_threshold=0.25):
    """Draw bounding boxes and labels on the image"""
    # Make a copy of the image
    img_with_boxes = image.copy()

    # Define colors for each class (standing, sitting, lying, throwing)
    colors = [
        (0, 255, 0),    # Green for standing
        (0, 0, 255),    # Blue for sitting
        (255, 0, 0),    # Red for lying
        (255, 255, 0),  # Yellow for throwing
    ]

    # Class names
    class_names = ["standing", "sitting", "lying", "throwing"]

    # Get boxes, confidence scores, and class IDs
    boxes = results[0].boxes.xyxy.cpu().numpy()
    confs = results[0].boxes.conf.cpu().numpy()
    cls_ids = results[0].boxes.cls.cpu().numpy().astype(int)

    # Draw each box if above threshold
    for box, conf, cls_id in zip(boxes, confs, cls_ids):
        if conf < confidence_threshold:
            continue

        # Get box coordinates
        x1, y1, x2, y2 = map(int, box)

        # Get class color and name
        color = colors[cls_id % len(colors)]
        class_name = class_names[cls_id] if cls_id < len(
            class_names) else f"Class {cls_id}"

        # Draw rectangle and label
        cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), color, 2)
        label = f"{class_name} {conf:.2f}"

        # Calculate text size for better positioning
        (label_width, label_height), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )

        # Draw label background
        cv2.rectangle(
            img_with_boxes,
            (x1, y1 - label_height - 5),
            (x1 + label_width, y1),
            color,
            -1  # Fill
        )

        # Draw label text
        cv2.putText(
            img_with_boxes,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

    return img_with_boxes


def evaluate_model(model_path, test_dir, output_dir, confidence_threshold=0.25, device=''):
    """Evaluate the YOLO model on test images"""
    # Load the model
    model = YOLO(model_path)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get all test images
    image_extensions = ['.jpg', '.jpeg', '.png']
    test_images = [
        os.path.join(test_dir, f) for f in os.listdir(test_dir)
        if any(f.lower().endswith(ext) for ext in image_extensions)
    ]

    # Track performance metrics
    inference_times = []

    # Process each test image
    for img_path in test_images:
        # Read image
        img = cv2.imread(img_path)
        if img is None:
            print(f"Could not read image: {img_path}")
            continue

        # RGB conversion for display
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Measure inference time
        start_time = time.time()
        results = model.predict(
            img_rgb, conf=confidence_threshold, device=device)
        inference_time = (time.time() - start_time) * 1000  # ms
        inference_times.append(inference_time)

        # Draw predictions on image
        img_with_boxes = draw_predictions(img, results, confidence_threshold)

        # Create output path
        base_name = os.path.basename(img_path)
        output_path = os.path.join(output_dir, f"pred_{base_name}")

        # Save output image
        cv2.imwrite(output_path, img_with_boxes)

    # Calculate and print statistics
    if inference_times:
        avg_inference_time = np.mean(inference_times)
        print(f"Evaluated {len(test_images)} images")
        print(f"Average inference time: {avg_inference_time:.2f} ms")

        # Plot inference time histogram
        plt.figure(figsize=(10, 6))
        plt.hist(inference_times, bins=20)
        plt.axvline(x=50, color='r', linestyle='--', label='50ms threshold')
        plt.xlabel('Inference Time (ms)')
        plt.ylabel('Frequency')
        plt.title(
            f'Inference Time Distribution (Avg: {avg_inference_time:.2f} ms)')
        plt.legend()
        plt.tight_layout()

        # Save histogram
        plt.savefig(os.path.join(output_dir, 'inference_time_hist.png'))

        # Return metrics
        return {
            'avg_inference_time': avg_inference_time,
            'num_images': len(test_images)
        }
    else:
        print("No images were successfully processed")
        return None


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained YOLO model")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to trained YOLO model")
    parser.add_argument("--test-dir", type=str, default="CV_Test/Images",
                        help="Directory containing test images")
    parser.add_argument("--output-dir", type=str, default="test_outputs/yolo",
                        help="Directory to save outputs")
    parser.add_argument("--confidence", type=float, default=0.25,
                        help="Confidence threshold")
    parser.add_argument("--device", type=str, default='',
                        help="Device to use for inference (empty for auto)")

    args = parser.parse_args()

    # Check if model exists
    if not os.path.exists(args.model):
        print(f"Model not found at: {args.model}")
        return

    # Check if test directory exists
    if not os.path.exists(args.test_dir):
        print(f"Test directory not found: {args.test_dir}")
        return

    # Run evaluation
    print(f"Starting evaluation on {args.test_dir}...")
    metrics = evaluate_model(
        args.model,
        args.test_dir,
        args.output_dir,
        args.confidence,
        args.device
    )

    if metrics:
        print("Evaluation complete!")
        print(f"Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
