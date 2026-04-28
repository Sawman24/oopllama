#!/bin/bash

# NOVA Model Downloader
# This script downloads a lightweight, highly capable model directly into the models/ directory.
# We are starting with Microsoft's Phi-3-mini-4k-instruct (3.8B parameters) or TinyLlama 
# to ensure it comfortably fits in your V100's 16GB VRAM at FP16 precision without quantization.

MODEL_DIR="./models"
MODEL_FILE="model.safetensors"
# Using TinyLlama 1.1B Chat for rapid testing. It's fully open and requires no API tokens.
MODEL_URL="https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/resolve/main/model.safetensors"

# Ensure the models directory exists
mkdir -p "$MODEL_DIR"

echo "========================================================="
echo "Initializing NOVA Brain Download..."
echo "Target: $MODEL_DIR/$MODEL_FILE"
echo "Downloading from HuggingFace (This may take a few minutes depending on connection speed)..."
echo "========================================================="

# Download the model using curl, following redirects (-L) and showing a progress bar (-#)
curl -L -# "$MODEL_URL" -o "$MODEL_DIR/$MODEL_FILE"

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================================="
    echo "Download Complete! 🧠"
    echo "NOVA's core reasoning engine has been saved to $MODEL_DIR/$MODEL_FILE."
    echo "You can now restart your Oopllama Docker container."
    echo "========================================================="
else
    echo ""
    echo "========================================================="
    echo "❌ Download Failed. Please check your internet connection and try again."
    echo "========================================================="
    exit 1
fi
