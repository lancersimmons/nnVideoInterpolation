# nnVideoInterpolation

for combining images into mp4:

ffmpeg -f image2 -r FRAMERATE -i frame%05d.jpg -vcodec mpeg4 -y FILENAME.mp4
