# nnVideoInterpolation

This project interpolates video files by predicting additional frames. These predicted frames are spliced with the original footage to create a more natural slow-mo effect.




Note - for combining images into mp4:
ffmpeg -f image2 -r FRAMERATE -i frame%05d.jpg -vcodec mpeg4 -y FILENAME.mp4
