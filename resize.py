#!/usr/bin/python

# IMPORTS
import sys
import os
import cv2
import time
import shutil
import errno

def generateFrames(filename):
    # generate all the frames
    try:
        cap = cv2.VideoCapture(filename)
        count = 0
        while cap.isOpened():
            ret,frame = cap.read()
            resizedImage = cv2.resize(frame, (360, 240))

            ## preview frames
            # cv2.imshow('window-name',resizedImage)
            # time.sleep(0.035)
            formattedString = format(count, '05d')
            # print(formattedString)
            cv2.imwrite("frame%s.jpg" % formattedString, resizedImage)
            count = count + 1
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cap.destroyAllWindows()
    except:
       pass

def create_directory_safely(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

def delete_directory_and_contents(path):
    try:
        shutil.rmtree(path)
    except OSError:
        pass


def main():

    ## Check user input
    if (len(sys.argv) == 1):
        print "Error: Invalid execution. Expected an .mp4 file."
        exit()

    filename = sys.argv[1]
    print "Converting file: " + filename

    # Generate frames
    generateFrames(filename)

    # Create directory, if needed
    print "Move split frames to frames directory"
    delete_directory_and_contents("frames")
    create_directory_safely("frames")

    # Copy all the frames into a directory
    cwd = os.getcwd()
    sourceDirFiles = os.listdir(cwd)
    destinationDir = cwd + "/frames"

    for fil in sourceDirFiles:
        if fil.endswith(".jpg"):
            shutil.move(fil,destinationDir)

if __name__ == "__main__":
    main()