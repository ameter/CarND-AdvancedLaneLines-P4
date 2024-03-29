'''
Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
'''

import numpy as np
import cv2
import glob
import pickle

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*9, 3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.

# Make a list of calibration images
images = glob.glob('./camera_cal/calibration*.jpg')

# Step through the list and search for chessboard corners
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (9, 6),None)

    # If found, add object points, image points
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (9, 6), corners, ret)
        cv2.imshow('img', img)
        cv2.waitKey(200)

cv2.destroyAllWindows()

# Get image shape
img = cv2.imread(images[0])
img_size = (img.shape[1], img.shape[0])

# Compute the camera calibration matrix and distortion coefficients
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

# Output corrected image example
# img = cv2.imread("./camera_cal/calibration1.jpg")
# cv2.imwrite("./output_images/chessboard_input.jpg", img)
# img = cv2.undistort(img, mtx, dist, None, mtx)
# cv2.imwrite("./output_images/chessboard_undistorted.jpg", img)

print ("\nCalibration Matrix:")
print (mtx)
print ("\nDistortion Coefficients:")
print (dist)

calibration_data = {
    'mtx': mtx,
    'dist': dist
}

print(calibration_data)

pickle.dump(calibration_data, open("./calibration.p", "wb"))
print("\nCalibration data stored.")