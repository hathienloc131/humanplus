#!/bin/bash

# Real-time inference deployment script for HIT

# Parse command line arguments
POLICY_PATH=""
NORM_STATS_PATH=""
CAMERA_NAMES=("cam_high" "cam_low")
OBSERVATION_NAMES=("qpos")
WIDTH=224
HEIGHT=224
DEVICE="cuda"
VISUALIZATION=true

print_usage() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  --policy-path PATH         Path to the trained policy checkpoint (required)"
    echo "  --norm-stats-path PATH     Path to the normalization statistics (required)"
    echo "  --camera-names NAME1 NAME2 Names of cameras to use (default: cam_high cam_low)"
    echo "  --observation-names NAME1  Names of observation fields to use (default: qpos)"
    echo "  --width WIDTH              Width to resize images to (default: 224)"
    echo "  --height HEIGHT            Height to resize images to (default: 224)"
    echo "  --device DEVICE            Device to run inference on (default: cuda)"
    echo "  --no-visualization         Disable visualization"
    echo "  --help                     Show this help message"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --policy-path)
            POLICY_PATH="$2"
            shift 2
            ;;
        --norm-stats-path)
            NORM_STATS_PATH="$2"
            shift 2
            ;;
        --camera-names)
            CAMERA_NAMES=()
            shift
            while [[ $# -gt 0 && ! $1 =~ ^-- ]]; do
                CAMERA_NAMES+=("$1")
                shift
            done
            ;;
        --observation-names)
            OBSERVATION_NAMES=()
            shift
            while [[ $# -gt 0 && ! $1 =~ ^-- ]]; do
                OBSERVATION_NAMES+=("$1")
                shift
            done
            ;;
        --width)
            WIDTH="$2"
            shift 2
            ;;
        --height)
            HEIGHT="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --no-visualization)
            VISUALIZATION=false
            shift
            ;;
        --help)
            print_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            print_usage
            exit 1
            ;;
    esac
done

# Check required arguments
if [ -z "$POLICY_PATH" ] || [ -z "$NORM_STATS_PATH" ]; then
    echo "Error: --policy-path and --norm-stats-path are required"
    print_usage
    exit 1
fi

# Build camera and observation name arguments
CAMERA_NAMES_ARG=""
for name in "${CAMERA_NAMES[@]}"; do
    CAMERA_NAMES_ARG="$CAMERA_NAMES_ARG $name"
done

OBSERVATION_NAMES_ARG=""
for name in "${OBSERVATION_NAMES[@]}"; do
    OBSERVATION_NAMES_ARG="$OBSERVATION_NAMES_ARG $name"
done

# Run the appropriate script based on visualization setting
if [ "$VISUALIZATION" = true ]; then
    echo "Starting real-time inference with visualization..."
    python -m HIT.visualize_inference \
        --policy_path "$POLICY_PATH" \
        --norm_stats_path "$NORM_STATS_PATH" \
        --camera_names $CAMERA_NAMES_ARG \
        --observation_names $OBSERVATION_NAMES_ARG \
        --width "$WIDTH" \
        --height "$HEIGHT" \
        --device "$DEVICE"
else
    echo "Starting real-time inference without visualization..."
    python -m HIT.example_realtime_inference \
        --policy_path "$POLICY_PATH" \
        --norm_stats_path "$NORM_STATS_PATH" \
        --camera_names $CAMERA_NAMES_ARG \
        --observation_names $OBSERVATION_NAMES_ARG \
        --width "$WIDTH" \
        --height "$HEIGHT" \
        --device "$DEVICE"
fi
