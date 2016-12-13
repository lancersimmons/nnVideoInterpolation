#!/bin/bash
# Demo script to generate videos

cd prediction
ffmpeg -f image2 -r 30 -i PredictedSeq%05d.jpg -vcodec mpeg4 -y interpolated.mp4
mv interpolated.mp4 ..
cd ..
cd original
ffmpeg -f image2 -r 30 -i OriginalSeq%05d.jpg -vcodec mpeg4 -y original.mp4
mv original.mp4 ..