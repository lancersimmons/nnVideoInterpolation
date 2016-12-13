#!/usr/bin/python
#
# Python 2 compatible
# Converts an .mp4 file to a directory of .jpg frames.


# IMPORTS
import cv2
import errno
import os
import shutil
import sys
import time

## PARAMETERS

# Dimensions of Output Frames
frameWidth = 160
frameHeight = 96
# This is used to specifiy how many frames are trained on as a group
# within the Neural Net. (3 for triplets, 5 for quintuplets, etc.)
frameGroupSize = 3


## FUNCTIONS

# Generate all frames from a video file. 
# This function is slightly bugged;
# It will throw an OpenCV error after finishing converting all the frames of the video
# I'm probably failing to do some sort of cleanup correctly in this function              
def generateFrames(filename):
    print("Ignore the following OpenCV Error:")
    try:
        cap = cv2.VideoCapture(filename)
        count = 0
        while cap.isOpened():
            ret,frame = cap.read()
            resizedImage = cv2.resize(frame, (frameWidth, frameHeight))

            ## preview frames in a popup window
            # cv2.imshow('window-name',resizedImage)
            # time.sleep(0.035)

            formattedString = format(count, '05d')
            cv2.imwrite("frame%s.jpg" % formattedString, resizedImage)
            count = count + 1
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cap.destroyAllWindows()
    except:
       pass

# this allows us to safely create a directory,
# avoiding potential race condition
def create_directory_safely(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

# recursively delete a directory and its contents
def delete_directory_and_contents(path):
    try:
        shutil.rmtree(path)
    except OSError:
        print("ERROR: Failed to delete path: " + str(path))

## MAIN

def main():

    # Validate user input
    if (len(sys.argv) == 1):
        print "Error: Invalid execution. Expected an .mp4 file."
        exit()

    filename = sys.argv[1]
    print "Converting file: " + filename

    # Generate frames
    generateFrames(filename)

    # Delete old frames directory, if needed,
    # Then create frames directory
    print "Moving split frames to frames directory."
    delete_directory_and_contents("frames")
    create_directory_safely("frames")

    # Copy all the frames into a directory
    # cwd is the current working directory this script is ran from
    cwd = os.getcwd()

    # Get all filenames in this directory
    tempDirFiles = os.listdir(cwd)
    destinationDir = cwd + "/frames"

    # Get all the frame files
    sourceDirFiles = []
    for entry in tempDirFiles:
        if ("frame" not in entry) or (".jpg" not in entry):
            # print("Not a frame: " + entry)
            pass
        else:
            sourceDirFiles.append(entry)

    # Alphabetize frame list so we can drop frames sfrom the end
    sourceDirFiles.sort()

    # Drop frames from the end until total frames is a multiple of frameGroupSize
    # This is so we can get triplets, quintuplets, etc.
    numberFrames = len(sourceDirFiles)
    while numberFrames % frameGroupSize != 0:
        print("Dropping frame to hit proper modulus: " + str(sourceDirFiles[-1]))        
        # Delete file
        os.remove(sourceDirFiles[-1])
        # Forget filename
        del sourceDirFiles[-1]
        # Check number of frames now
        numberFrames = len(sourceDirFiles)
    # Move all files in our working directory into the newly created frames directory
    for fil in sourceDirFiles:
        if fil.endswith(".jpg"):
            shutil.move(fil,destinationDir)

if __name__ == "__main__":
    main()