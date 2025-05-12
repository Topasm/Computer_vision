#!/bin/bash

# This script runs YOLO training with data augmentation for better performance

RUN_NAME="run_augmented"  # Change this to a unique name for each training run
MODEL="yolov8n.pt"        # Starting model
EPOCHS=100                # Number of epochs 
BATCH_SIZE=16             # Batch size
IMG_SIZE=640              # Image size

# Run training with augmentations
python train_yolo.py \
  --model $MODEL \
  --epochs $EPOCHS \
  --batch-size $BATCH_SIZE \
  --img-size $IMG_SIZE \
  --run-name $RUN_NAME \
  --device cuda:0 \
  --degrees 10.0 \
  --translate 0.1 \
  --scale 0.5 \
  --shear 2.0 \
  --perspective 0.001 \
  --flipud 0.0 \
  --fliplr 0.5 \
  --mosaic 1.0 \
  --mixup 0.1 \
  --copy-paste 0.1

echo "Training complete! Model saved to yolo_runs/$RUN_NAME/"
