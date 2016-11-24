#!/usr/bin/python

# IMPORTS
import sys
import cv2
import time

if (len(sys.argv) == 1):
    print "Error: Invalid execution. Expected an .mp4 file."
    exit()

filename = sys.argv[1]
print "Converting file: " + filename


try:
    cap = cv2.VideoCapture(filename)
    count = 0
    while cap.isOpened():
        ret,frame = cap.read()
        resizedImage = cv2.resize(frame, (360, 240))

        ## preview frames
        # cv2.imshow('window-name',resizedImage)
        # time.sleep(0.035)

        cv2.imwrite("frame%d.jpg" % count, resizedImage)
        count = count + 1
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cap.destroyAllWindows()
except:
    pass

