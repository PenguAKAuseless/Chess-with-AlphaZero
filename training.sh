#!/bin/bash

# Script to run scGPT training in SSH background
# This script will start the training process and keep it running even after logging out

# Activate Python environment if needed
# source /path/to/your/env/bin/activate  # Uncomment and modify if needed

# Set up directories - modify these as needed
INPUT_DIR="/nfsshared/dung/filtered_mapped"
CHECKPOINT_DIR="/nfsshared/dung/pengu/checkpoints"
DEVICE=1
MODEL_SAVE_DIR="/nfsshared/dung/pengu/models"
LOG_DIR="logs"
MAPPING_DIR="/nfsshared/dung/pengu/mapping"
D_MODEL=512
N_HEADS=8
N_LAYERS=6
D_FF=2048
DROPOUT=0.1
MAX_SEQ_LEN=512

# Create directories if they don't exist with error checking
echo "Creating directories at $(date)"
mkdir -p "$CHECKPOINT_DIR" || { echo "Failed to create $CHECKPOINT_DIR at $(date)"; exit 1; }
mkdir -p "$MODEL_SAVE_DIR" || { echo "Failed to create $MODEL_SAVE_DIR at $(date)"; exit 1; }
mkdir -p "$LOG_DIR" || { echo "Failed to create $LOG_DIR at $(date)"; exit 1; }
mkdir -p "$MAPPING_DIR" || { echo "Failed to create $MAPPING_DIR at $(date)"; exit 1; }

# Set timestamp for run ID
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/training_$TIMESTAMP.log"

echo "Starting training process at $(date)"
echo "Timestamp for this run: $TIMESTAMP"
echo "Input directory: $INPUT_DIR"
echo "Checkpoint directory: $CHECKPOINT_DIR"
echo "Device: $DEVICE"
echo "Model save directory: $MODEL_SAVE_DIR"
echo "Log directory: $LOG_DIR"
echo "Mapping directory: $MAPPING_DIR"
echo "Log file: $LOG_FILE"

# Run training with nohup and capture PID
echo "Launching training script with nohup at $(date)..."
nohup python trainer.py \
  --input_dir "$INPUT_DIR" \
  --checkpoint_dir "$CHECKPOINT_DIR" \
  --device "$DEVICE" \
  --model_save_dir "$MODEL_SAVE_DIR" \
  --log_dir "$LOG_DIR" \
  --mapping_dir "$MAPPING_DIR" \
  --d_model "$D_MODEL" \
  --n_heads "$N_HEADS" \
  --n_layers "$N_LAYERS" \
  --d_ff "$D_FF" \
  --dropout "$DROPOUT" \
  --max_seq_len "$MAX_SEQ_LEN" \
  > "$LOG_DIR/stdout_$TIMESTAMP.log" 2> "$LOG_DIR/stderr_$TIMESTAMP.log" &

# Save the process ID to kill it later if needed
PID=$!
echo $PID > "$LOG_DIR/training_pid_$TIMESTAMP.txt"
echo "Training process started with PID: $! at $(date)"
echo "Logs will be saved to: $LOG_FILE"
echo "Standard output redirected to: $LOG_DIR/stdout_$TIMESTAMP.log"
echo "Standard error redirected to: $LOG_DIR/stderr_$TIMESTAMP.log"
echo "To monitor logs in real-time: tail -f $LOG_FILE"
echo "To stop the process: kill -9 \$(cat $LOG_DIR/training_pid_$TIMESTAMP.txt)"
echo "Training process launched successfully at $(date)"