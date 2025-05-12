import os
import json
import shutil
from pathlib import Path
import random

# Class names
CLASSES = ["standing", "sitting", "lying", "throwing"]


def create_yolo_directories(base_dir):
    """Create YOLO dataset directories structure"""
    yolo_dir = os.path.join(base_dir, "yolo_dataset")

    # Create main directories
    os.makedirs(yolo_dir, exist_ok=True)
    os.makedirs(os.path.join(yolo_dir, "images", "train"), exist_ok=True)
    os.makedirs(os.path.join(yolo_dir, "images", "val"), exist_ok=True)
    os.makedirs(os.path.join(yolo_dir, "labels", "train"), exist_ok=True)
    os.makedirs(os.path.join(yolo_dir, "labels", "val"), exist_ok=True)

    # Create dataset.yaml file for YOLO
    yaml_content = f"""
# YOLO dataset configuration
path: {os.path.abspath(yolo_dir)}
train: images/train
val: images/val

# Classes
names:
  0: standing
  1: sitting
  2: lying
  3: throwing

# Number of classes
nc: {len(CLASSES)}
"""

    with open(os.path.join(yolo_dir, "dataset.yaml"), "w") as f:
        f.write(yaml_content)

    return yolo_dir


def convert_to_yolo_format(json_path, image_width, image_height):
    """
    Convert JSON annotation to YOLO format.

    YOLO format: <class_id> <x_center> <y_center> <width> <height>
    All values normalized to [0, 1]
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    yolo_annotations = []

    # Process each shape (bounding box)
    for shape in data.get("shapes", []):
        label = shape.get("label")
        points = shape.get("points")

        # Skip if not a supported class or not rectangle
        if label not in CLASSES or shape.get("shape_type") != "rectangle":
            continue

        # Get class ID
        class_id = CLASSES.index(label)

        # Extract box coordinates (top-left and bottom-right corners)
        x1, y1 = points[0]
        x2, y2 = points[1]

        # Calculate YOLO format (normalized center x, center y, width, height)
        box_width = abs(x2 - x1)
        box_height = abs(y2 - y1)
        x_center = (x1 + x2) / 2
        y_center = (y1 + y2) / 2

        # Normalize coordinates
        x_center /= image_width
        y_center /= image_height
        box_width /= image_width
        box_height /= image_height

        # Create YOLO annotation line
        yolo_annotation = f"{class_id} {x_center} {y_center} {box_width} {box_height}"
        yolo_annotations.append(yolo_annotation)

    return "\n".join(yolo_annotations)


def process_dataset(source_img_dir, source_label_dir, yolo_base_dir, val_split=0.0):
    """
    Process dataset and convert to YOLO format.

    Args:
        source_img_dir (str): Directory containing images
        source_label_dir (str): Directory containing JSON labels
        yolo_base_dir (str): Base directory for YOLO dataset
        val_split (float): Validation set ratio (0.0 to 1.0), defaults to 0 to use all data for training
    """
    # Get all image files
    image_files = sorted([f for f in os.listdir(source_img_dir)
                          if f.lower().endswith((".jpg", ".jpeg", ".png"))])

    # Use all images for training (no validation split)
    train_files = image_files
    val_files = []  # Empty validation set

    print(f"Total images: {len(image_files)}")
    print(f"Training images: {len(train_files)}")
    print(f"Validation images: {len(val_files)}")

    # Define paths
    train_img_dir = os.path.join(yolo_base_dir, "images", "train")
    train_label_dir = os.path.join(yolo_base_dir, "labels", "train")
    val_img_dir = os.path.join(yolo_base_dir, "images", "val")
    val_label_dir = os.path.join(yolo_base_dir, "labels", "val")

    # Process training files
    for img_file in train_files:
        process_file(img_file, source_img_dir, source_label_dir,
                     train_img_dir, train_label_dir)

    # Process validation files
    for img_file in val_files:
        process_file(img_file, source_img_dir, source_label_dir,
                     val_img_dir, val_label_dir)


def process_file(img_file, source_img_dir, source_label_dir, target_img_dir, target_label_dir):
    """Process a single file, convert label, and copy to target directories"""
    base_name = os.path.splitext(img_file)[0]
    json_file = f"{base_name}.json"

    # Source paths
    img_path = os.path.join(source_img_dir, img_file)
    json_path = os.path.join(source_label_dir, json_file)

    # Target paths
    target_img_path = os.path.join(target_img_dir, img_file)
    target_label_path = os.path.join(target_label_dir, f"{base_name}.txt")

    # Skip if JSON file doesn't exist
    if not os.path.exists(json_path):
        print(f"Warning: Label file not found for {img_file}, skipping.")
        return

    # Copy image
    shutil.copy(img_path, target_img_path)

    # Read image dimensions from JSON
    with open(json_path, "r") as f:
        data = json.load(f)
        img_width = data.get("imageWidth", 640)  # Default if not specified
        img_height = data.get("imageHeight", 345)  # Default if not specified

    # Convert annotation to YOLO format
    yolo_content = convert_to_yolo_format(json_path, img_width, img_height)

    # Save YOLO format annotation
    with open(target_label_path, "w") as f:
        f.write(yolo_content)


def main():
    # Set base directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Base directory: {base_dir}")

    # Find the source data directories
    cv_train_img_dir = os.path.join(base_dir, "CV_Train", "Images")
    cv_train_label_dir = os.path.join(base_dir, "CV_Train", "Labels")
    cv_test_img_dir = os.path.join(base_dir, "CV_Test", "Images")
    cv_test_label_dir = os.path.join(base_dir, "CV_Test", "Labels")

    print(f"Looking for train images in: {cv_train_img_dir}")
    print(f"Looking for train labels in: {cv_train_label_dir}")

    # Check if directories exist
    if os.path.exists(cv_train_img_dir):
        print(
            f"Train image directory exists with {len(os.listdir(cv_train_img_dir))} files")
    else:
        print(f"Train image directory NOT FOUND: {cv_train_img_dir}")

    if os.path.exists(cv_train_label_dir):
        print(
            f"Train label directory exists with {len(os.listdir(cv_train_label_dir))} files")
    else:
        print(f"Train label directory NOT FOUND: {cv_train_label_dir}")

    # Create YOLO directories
    yolo_dir = create_yolo_directories(base_dir)

    # Check if CV_Train and CV_Test exist
    if os.path.exists(cv_train_img_dir) and os.path.exists(cv_train_label_dir):
        print("Processing training dataset...")
        # Use all training dataset for training (no validation split)
        process_dataset(cv_train_img_dir, cv_train_label_dir,
                        yolo_dir, val_split=0.0)
    else:
        print(
            f"Training directories not found: {cv_train_img_dir}, {cv_train_label_dir}")

    print("\nDataset conversion completed successfully!")
    print(f"YOLO dataset created at: {yolo_dir}")
    print(
        f"To train: yolo train model=yolov8n.pt data={os.path.join(yolo_dir, 'dataset.yaml')}")


if __name__ == "__main__":
    main()
