#!/usr/bin/python
#
# Converts .mp4 to frames.
#
# FIXME:
# NOTE: This script might need to be ran multiple times in its current state.
# It will rarely, randomly drop frames during the conversion


# IMPORTS
import sys
import os
import cv2
import time
import shutil
import errno
import time

# PARAMETERS
frameWidth = 160 #360
frameHeight = 96 #240
frameGroupSize = 3


# FUNCTIONS
def generateFrames(filename):
    # generate all the frames
    try:
        cap = cv2.VideoCapture(filename)
        count = 0
        while cap.isOpened():
            ret,frame = cap.read()
            resizedImage = cv2.resize(frame, (frameWidth, frameHeight))

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
    print "Ignore the following OpenCV Error:"

    # Generate frames
    generateFrames(filename)

    # Create directory, if needed
    print "Moving split frames to frames directory."
    delete_directory_and_contents("frames")
    create_directory_safely("frames")

    # Copy all the frames into a directory
    # cwd is current working directory
    cwd = os.getcwd()
    # get all filenames in this directory
    tempDirFiles = os.listdir(cwd)
    destinationDir = cwd + "/frames"

    # get all the frame files
    sourceDirFiles = []
    for entry in tempDirFiles:
        if ("frame" not in entry) or (".jpg" not in entry):
            # print("Not a frame: " + entry)
            pass
        else:
            sourceDirFiles.append(entry)
    # alphabetize frame list so we can drop from the end
    sourceDirFiles.sort()


    # Drop frames from the end until total frames is a multiple of frameGroupSize
    # This is so we can get triplets, quintuplets, etc.
    numberFrames = len(sourceDirFiles)
    while numberFrames % frameGroupSize != 0:
        print("Dropping frame to hit proper modulus." + str(sourceDirFiles[-1]))
        # delete file
        os.remove(sourceDirFiles[-1])
        # forget filename
        del sourceDirFiles[-1]
        # check number of frames now
        numberFrames = len(sourceDirFiles)

    for fil in sourceDirFiles:
        if fil.endswith(".jpg"):
            shutil.move(fil,destinationDir)




if __name__ == "__main__":
    main()