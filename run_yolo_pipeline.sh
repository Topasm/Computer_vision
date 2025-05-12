#!/bin/bash
# filepath: /home/ahrilab/Desktop/CV/Computer_vision/run_yolo_pipeline.sh

# Exit on error
set -e

# Define paths and settings
DATASET_DIR="$(pwd)/yolo_dataset"
MODEL_TYPE="yolov8n.pt"  # yolov8n.pt, yolov8s.pt, yolov8m.pt, etc.
EPOCHS=50
BATCH_SIZE=16
IMG_SIZE=640
PATIENCE=10
PROJECT_NAME="yolo_runs"
RUN_NAME="run_$(date +%Y%m%d_%H%M%S)"
CONFIDENCE=0.25

# Check if dataset exists
if [ ! -f "${DATASET_DIR}/dataset.yaml" ]; then
    echo "Dataset not found. Converting data to YOLO format..."
    python convert_to_yolo.py
    
    if [ ! -f "${DATASET_DIR}/dataset.yaml" ]; then
        echo "Failed to create dataset. Check convert_to_yolo.py output."
        exit 1
    fi
fi

# Display configuration
echo "--------------------------------------"
echo "YOLO Training and Evaluation Pipeline"
echo "--------------------------------------"
echo "Dataset: ${DATASET_DIR}"
echo "Model: ${MODEL_TYPE}"
echo "Epochs: ${EPOCHS}"
echo "Batch Size: ${BATCH_SIZE}"
echo "Image Size: ${IMG_SIZE}"
echo "Run Name: ${RUN_NAME}"
echo "--------------------------------------"

# Train model
echo "Starting training..."
python train_yolo.py \
    --model "${MODEL_TYPE}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --img-size "${IMG_SIZE}" \
    --patience "${PATIENCE}" \
    --project-name "${PROJECT_NAME}" \
    --run-name "${RUN_NAME}" \
    --use-wandb

# Get path to best trained model
BEST_MODEL="${PROJECT_NAME}/${RUN_NAME}/weights/best.pt"

if [ ! -f "${BEST_MODEL}" ]; then
    echo "Training did not produce a best model. Using last model."
    BEST_MODEL="${PROJECT_NAME}/${RUN_NAME}/weights/last.pt"
    
    if [ ! -f "${BEST_MODEL}" ]; then
        echo "No model found after training. Exiting."
        exit 1
    fi
fi

# Evaluate the model
echo "Evaluating model on test dataset..."
python evaluate_yolo.py \
    --model "${BEST_MODEL}" \
    --test-dir "CV_Test/Images" \
    --output-dir "test_outputs/yolo_${RUN_NAME}" \
    --confidence "${CONFIDENCE}"

echo "Pipeline complete. Check test_outputs/yolo_${RUN_NAME} for results."
