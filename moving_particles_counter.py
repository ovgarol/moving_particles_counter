# SPDX-FileCopyrightText: 2025 Helmholtz-Zentrum hereon GmbH
# SPDX-License-Identifier: CC0-1.0
# SPDX-FileContributor Ovidio Garcia-Oliva <ovidio.garcia@hereon.de>

## based on:
## https://www.educative.io/answers/background-subtraction-opencv
## https://www.geeksforgeeks.org/find-the-solidity-and-equivalent-diameter-of-an-image-object-using-opencv-python/
## https://pyimagesearch.com/2016/03/28/measuring-size-of-objects-in-an-image-with-opencv/
## https://www.geeksforgeeks.org/python-opencv-background-subtraction/

import cv2
import numpy as np
import pandas as pd
import argparse

show_it = False
approx_it = False
edge_it = False
min_diameter = 0

columns = ['Frame', 'ParticleID', 'Sol', 'ESD', 'Area']
data = pd.DataFrame(columns=columns)

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
    help="path to the input video")
ap.add_argument("-x", "--index", type=int, required=True,
    help="index of the output file")
args = vars(ap.parse_args())

print(args)

# function to find real Area
def find_area(count):
    x,y,w,h = cv2.boundingRect(count)
    return w*h

# function to find Solidity 
def find_solidity(count): 
    contourArea = cv2.contourArea(count) 
    convexHull = cv2.convexHull(count) 
    contour_hull_area = cv2.contourArea(convexHull) + 0.01
    solidity = float(contourArea)/contour_hull_area 
    return solidity 

# function to calculate the equivalent diameter 
def find_equi_diameter(count): 
    area = cv2.contourArea(count) 
    equi_diameter = 2*np.sqrt(area/np.pi) 
    return equi_diameter 

# function to get all particles 
def find_particles(gray,idx): 
    data_tmp = pd.DataFrame(columns=columns)

    gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY) 
    # apply thresholding to convert grayscale image to the binary image 
    # and find the contours 

    # perform edge detection, then perform a dilation + erosion to
    # close gaps in between object edges
    if edge_it:
        edged = cv2.Canny(gray, 50, 100)
        edged = cv2.dilate(edged, None, iterations=1)
        edged = cv2.erode(edged, None, iterations=1)
        gray = edged

    contours, hierarchy = cv2.findContours(gray, 1, cv2.CHAIN_APPROX_SIMPLE ) 
    #print("Number of objects detected:", len(contours)) 

    # iterate over the list 'contours' to get 
    # solidity and Equivalent Diameter of each object 
    for i, cnt in enumerate(contours): 
        Solidity = find_solidity(cnt) 
        #Solidity = round(Solidity, 2) 
        equi_diameter = find_equi_diameter(cnt) 
        #equi_diameter = round(equi_diameter, 2) 
        area = cv2.contourArea(cnt)
        #area = round(area, 2) 
        #print(f"Solidity of object {i+1}: ", Solidity) 
        #print(f"Equivalent Diameter of object {i+1}: ", equi_diameter) 
        if(equi_diameter>min_diameter):
            #print(i,Solidity,equi_diameter)
            # Store the particle coordinates in the DataFrame
            cnt_data = pd.DataFrame({'Frame': idx, 'ParticleID': i, 'Sol': Solidity, 'ESD': equi_diameter, 'Area':area}, index=[0])
            data_tmp=pd.concat([data_tmp,cnt_data],ignore_index = True)
    return data_tmp, contours



# Path to the input video file
#inputvid = 'test4.mp4'
inputvid = args["input"]

# Create a VideoCapture object to read from the video file
vidcap = cv2.VideoCapture(inputvid)

# Get the video's properties
fps = vidcap.get(cv2.CAP_PROP_FPS)
width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create a VideoWriter object to save the output video
output_path = 'output_video-'+str(args["index"])+'.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Create a background subtractor object
bgsub = cv2.createBackgroundSubtractorMOG2(detectShadows = False,varThreshold = 32)
#bgsub =  cv2.bgsegm.createBackgroundSubtractorMOG()
#bgsub =  cv2.bgsegm.createBackgroundSubtractorGMG()

idx = 0

while True:
    # Read the video frame
    suc, vidframe = vidcap.read()

    # If there are no more frames to show, break the loop
    if not suc:
        break

    # Apply the background subtractor to the frame
    foremask = bgsub.apply(vidframe)

    # Convert the mask to 3 channels
    foremask = cv2.cvtColor(foremask, cv2.COLOR_GRAY2BGR)

    xx, contours = find_particles(foremask,idx) 
    data = pd.concat([data,xx],ignore_index = True)

    # Add the overlay to the current frame
    output_frame = cv2.addWeighted(vidframe, 1, foremask, 0.15, 0)

    if show_it:
        # Resize the frame and mask to a smaller size
        scale_percent = 50  # Adjust this value to control the scaling
        winwidth = int(vidframe.shape[1] * scale_percent / 100)
        winheight = int(vidframe.shape[0] * scale_percent / 100)
        small_vidframe = cv2.resize(vidframe, (winwidth, winheight))
        small_vidmask = cv2.resize(foremask, (winwidth, winheight))

        # Stack the resized frame and mask horizontally
        hstacked_frames = np.vstack((small_vidframe, small_vidmask))
        cv2.imshow("Original and Masked Video", hstacked_frames)

    for c in contours:
        the_area = cv2.contourArea(c)
        if  the_area > 1000: cv2.drawContours(output_frame, [c], -1, (0,0,0), 1)
        elif  the_area > 300: cv2.drawContours(output_frame, [c], -1, (0,0,255), 1)
        elif the_area > 100: cv2.drawContours(output_frame, [c], -1, (255,0,0), 1)
        elif  the_area > 30: cv2.drawContours(output_frame, [c], -1, (0,255,255), 1)
        elif the_area > 10: cv2.drawContours(output_frame, [c], -1, (255,255,0), 1)
        elif the_area > 3: cv2.drawContours(output_frame, [c], -1, (0,255,0), 1)
        elif the_area > 1: cv2.drawContours(output_frame, [c], -1, (255,0,255), 1)

        # polygonal approximation
        if approx_it: 
            for eps in [0.01]:#np.linspace(0.001, 0.05, 10):
                # approximate the contour
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, eps * peri, True)
                cv2.drawContours(output_frame, [approx], -1, (0, 0, 0), 1)

    # Write the frame to the output video file
    output.write(output_frame) # overlayed
    #output.write(foremask) # just masked

    idx = idx + 1

    # If the 'q' key is pressed, stop the loop
    if cv2.waitKey(1) == ord("q"):
        break

# Release the video capture object
vidcap.release()
cv2.destroyAllWindows()

# Save the particle coordinates to a CSV file
output_csv = 'particle_coordinates-'+str(args["index"])+'.csv'
data.to_csv(output_csv, index=False)
