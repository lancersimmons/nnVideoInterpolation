#!/bin/bash
# Demo script to exercise other scripts in order.

echo "Hello, $USER."
echo "Place the .mp4 file you would like to use in the same directory as the scripts."
echo "Enter your .mp4 file name and hit [ENTER]:"
read videoFileName
echo "$videoFileName"
echo "Starting training pipeline."
echo "Call python script"
python frameExtractor.py $videoFileName
sleep 1s
echo "Starting CNN training and testing."
th cnn_gpu.lua
sleep 1s