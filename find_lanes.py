'''
Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
Apply a distortion correction to raw images.
Use color transforms, gradients, etc., to create a thresholded binary image.
Apply a perspective transform to rectify binary image ("birds-eye view").
Detect lane pixels and fit to find the lane boundary.
Determine the curvature of the lane and vehicle position with respect to center.
Warp the detected lane boundaries back onto the original image.
Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.
'''

import numpy as np
import cv2
from glob import glob
import pickle
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip

ym_per_pix = 30 / 720  # meters per pixel in y dimension
xm_per_pix = 3.7 / 700  # meters per pixel in x dimension
DEBUG = False

lines = []
frame = 0
ploty = []

# Define a class to receive the characteristics of each line detection
class Line:
    def __init__(self, side, smoothing_factor=1):
        # which side of the lane is this line
        if side != "left" and side != "right":
            raise RuntimeError("Line.side must be either 'left' or 'right'")
        self.side = side
        # number of line samples to average over for smoothing
        self.smoothing_factor = smoothing_factor
        # number of bad detections (dropped frames)
        self.dropped_frames = 0
        # was the line detected in the last n iterations?
        self.detected = False

        # x values of the last n fits of the line
        self.current_xfit = []
        self.recent_xfitted = []
        #average x values of the fitted line over the last n iterations
        self.bestx = None

        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        #polynomial coefficients of last n fits of the line
        self.recent_fits = []
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0, 0, 0], dtype='float')

        #radius of curvature of the line in some meters
        self.radius_of_curvature = None
        self.previous_curvature = None

        #distance in meters of vehicle center from the line
        self.line_base_pos = None

        #x values for detected line pixels
        self.allx = None
        #y values for detected line pixels
        self.ally = None


# Thresholds the linear gradient (x direction by default to detect lines in more vertical direction
def abs_sobel_thresh(img, thresh_min=0, thresh_max=255, orient='x'):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Get absolute value of x or y gradient
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
    return binary_output

# Thresholds the S-channel of HLS
def hls_s_thresh(img, thresh_min=0, thresh_max=255):
    # Convert to HLS color space and separate the S channel
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    s_channel = hls[:,:,2]
    binary_output = np.zeros_like(s_channel)
    binary_output[(s_channel > thresh_min) & (s_channel <= thresh_max)] = 1
    return binary_output

# Shows two images
def show2(img1, img2, title1='Image 1', title2='Image 2'):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    ax1.set_title(title1)
    ax1.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    ax2.set_title(title2)
    ax2.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    plt.show()

#Apply a perspective transform to rectify binary image ("birds-eye view").
def perspective_transform(img):
    img_size = (img.shape[1], img.shape[0])

    # src = np.float32([[577, 464], [707, 464],[269, 675], [1036, 675]])
    # dst = np.float32([[423, 464], [861, 464], [423, 675], [861, 675]])

    src = np.float32([[220, 719], [1220, 719], [750, 480], [550, 480]])
    dst = np.float32([[240, 719], [1040, 719], [1040, 300], [240, 300]])

    # src = np.float32([(200, 720), (580, 480), (720, 480), (1050, 700)])
    # dst = np.float32([(280, 720), (400, 190), (920, 190), (960, 720)])

    # Compute the perspective transform, M, given source and destination points
    M = cv2.getPerspectiveTransform(src, dst)

    # Compute the inverse perspective transform
    Minv = cv2.getPerspectiveTransform(dst, src)

    # Warp an image using the perspective transform, M
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    return warped, Minv

# Detect lane pixels and fit to find the lane boundary without using previous lane info.
def find_line(binary_warped, line):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0] // 3:, :], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] / 2)
    if line.side == "left":
        x_base = np.argmax(histogram[:midpoint])
    elif line.side == "right":
        x_base = np.argmax(histogram[midpoint:]) + midpoint
    else:
        raise RuntimeError("Line.side must be either 'left' or 'right'")

    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0] / nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    x_current = x_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 250
    # Create empty lists to receive left and right lane pixel indices
    lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_x_low = x_current - margin
        win_x_high = x_current + margin

        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_x_low, win_y_low), (win_x_high, win_y_high),
                      (0, 255, 0), 2)

        # Identify the nonzero pixels in x and y within the window
        good_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_x_low) & (nonzerox < win_x_high)).nonzero()[0]

        # Append these indices to the lists
        lane_inds.append(good_inds)

        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_inds) > minpix:
            x_current = np.int(np.mean(nonzerox[good_inds]))
            print(frame, "rebasing", len(good_inds))

    # Concatenate the arrays of indices
    lane_inds = np.concatenate(lane_inds)

    # Extract line pixel positions
    allx = nonzerox[lane_inds]
    ally = nonzeroy[lane_inds]

    # Update line objects
    set_line_attributes(line, allx, ally, binary_warped)

    if DEBUG:
        out_img[line.ally, line.allx] = [255, 0, 0]
        plt.imshow(out_img)
        plt.plot(line.current_xfit, ploty, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)
        # plt.show()
        plt.savefig("./output_images/lines" + str(frame) + "-" + line.side)
        plt.close()


# Detect lane pixels and fit to find the lane boundary using previous lane info.
def update_line(binary_warped, line):
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 75
    lane_inds = ((nonzerox > (line.current_fit[0] * (nonzeroy ** 2) + line.current_fit[1] * nonzeroy + line.current_fit[2] - margin)) & (nonzerox < (line.current_fit[0] * (nonzeroy ** 2) + line.current_fit[1] * nonzeroy + line.current_fit[2] + margin)))

    # Again, extract line pixel positions
    allx = nonzerox[lane_inds]
    ally = nonzeroy[lane_inds]

    # Update line objects
    set_line_attributes(line, allx, ally, binary_warped)

    if DEBUG:
        # Create an image to draw on and an image to show the selection window
        out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
        window_img = np.zeros_like(out_img)

        # Color in left and right line pixels
        out_img[line.ally, line.allx] = [255, 0, 0]

        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        line_window1 = np.array([np.transpose(np.vstack([line.current_xfit - margin, ploty]))])
        line_window2 = np.array([np.flipud(np.transpose(np.vstack([line.current_xfit + margin, ploty])))])
        line_pts = np.hstack((line_window1, line_window2))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([line_pts]), (0, 255, 0))
        result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
        plt.imshow(result)
        plt.plot(line.current_xfit, ploty, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)
        # plt.show()
        plt.savefig("./output_images/lines" + str(frame) + "-" + line.side)
        plt.close()

def set_line_attributes(line, allx, ally, binary_warped):
    # Set line pixel positions
    line.allx = allx
    line.ally = ally

    # Fit a second order polynomial to each
    line.current_fit = np.polyfit(line.ally, line.allx, 2)

    if len(line.recent_fits) > 0:
        line.diffs = line.current_fit - line.recent_fits[-1]

    line.recent_fits.append(line.current_fit)
    if len(line.recent_fits) > line.smoothing_factor:
        line.recent_fits.pop(0)

    line.best_fit = np.mean(line.recent_fits, 0)

    # Generate fit x values and add to recent x fitted (for plotting)
    line.current_xfit = line.current_fit[0] * ploty ** 2 + line.current_fit[1] * ploty + line.current_fit[2]
    line.recent_xfitted.append(line.current_xfit)
    if len(line.recent_xfitted) > line.smoothing_factor:
        line.recent_xfitted.pop(0)

    line.bestx = np.mean(line.recent_xfitted, 0, np.int)
    line.line_base_pos = (line.current_xfit[-1] - binary_warped.shape[1] / 2) * xm_per_pix

    # Get curvature
    y_eval = np.max(ploty)
    # Fit new polynomials to x,y in world space
    fit_cr = np.polyfit(ploty * ym_per_pix, line.current_xfit * xm_per_pix, 2)
    # Calculate the new radii of curvature
    line.previous_curvature = line.radius_of_curvature
    line.radius_of_curvature = ((1 + (2 * fit_cr[0] * y_eval * ym_per_pix + fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * fit_cr[0])

    line.detected = True


def get_vehicle_position(lines):
    return lines[1].line_base_pos + lines[0].line_base_pos

def check_sanity(lines, binary_warped):
    # diff_prev_curv_left = lines[0].previous_curvature - lines[0].radius_of_curvature
    #print()
    # get difference between bottoms of lines
    dx_bottom = lines[0].current_xfit[-1] - lines[1].current_xfit[-1]
    # get a copy of left line superimposed over right line
    overlay = lines[0].current_xfit - dx_bottom
    # Get mean squared error of overlay and right line
    mse = ((overlay - lines[1].current_xfit) ** 2).mean()

    if mse > 10000:
        # The lines are not parallel enough.  Drop the detection
        # Note: We only reset the values in the frame that matter for output and the next detection.
        # If we use other values in the future, we will need to reset those here as well.
        print()
        print("Dropping", frame)
        print()
        for line in lines:
            line.current_fit = line.recent_fits.pop()
            line.best_fit =  np.mean(line.recent_fits, 0)
            line.current_xfit = line.recent_xfitted.pop()
            line.bestx = np.mean(line.recent_xfitted, 0, np.int)
            line.radius_of_curvature = line.previous_curvature
            line.dropped_frames += 1
            if line.dropped_frames > 10:
                line.detected = False
    else:
        for line in lines:
            line.dropped_frames = 0


    #print(frame, mse)
    #for line in lines:
        #print(frame, line.bestx[0], line.bestx[-1])
    # try:
    #     for line in lines:
    #         #curv_change = abs(line.previous_curvature - line.radius_of_curvature)
    #         # if curv_change > 1000:
    #         #     print()
    #         #     print(frame, "dropping", curv_change)
    #
    #         if np.sum(np.abs(line.diffs)) > 50:
    #             print()
    #             print(frame, "dropping", np.sum(np.abs(line.diffs)))
    #             # print(frame, abs(line.previous_curvature - line.radius_of_curvature))
    #             # print(frame, (line.previous_curvature - line.radius_of_curvature) / 2)
    #             #print(frame, line.current_fit.deriv())
    #             print(frame, "poly", np.polyder(line.current_fit))
    #             print(frame, "poly", np.polyder(line.current_fit,3))
    # except:
    #     pass
    # Create an image to draw on and an image to show the selection window
    if DEBUG:
        out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
        window_img = np.zeros_like(out_img)

        # Color in left and right line pixels
        out_img[lines[0].ally, lines[0].allx] = [255, 0, 0]
        out_img[lines[1].ally, lines[1].allx] = [0, 0, 255]

        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        #line_window1 = np.array([np.transpose(np.vstack([line.current_xfit - margin, ploty]))])
        #line_window2 = np.array([np.flipud(np.transpose(np.vstack([line.current_xfit + margin, ploty])))])
        #line_pts = np.hstack((line_window1, line_window2))

        # Draw the lane onto the warped blank image
        #cv2.fillPoly(window_img, np.int_([line_pts]), (0, 255, 0))
        #result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
        plt.imshow(out_img)
        plt.plot(lines[0].current_xfit, ploty, color='yellow')
        plt.plot(lines[1].current_xfit, ploty, color='yellow')
        plt.plot(overlay, ploty, color='green')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)
        plt.title("MSE: " + str(int(mse)))
        # plt.show()
        plt.savefig("./output_images/mse" + str(frame))
        plt.close()



def generate_output(lines, Minv, binary_warped, img):
    vehicle_position = ((img.shape[1] // 2) - ((lines[0].current_xfit[-1] + lines[1].current_xfit[-1]) // 2) ) * xm_per_pix
    side = "left"
    if vehicle_position > 0:
        side = "right"

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([lines[0].bestx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([lines[1].bestx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    pts = np.transpose(np.vstack([lines[0].bestx, ploty])).reshape((-1,1,2)).astype(np.int32)
    cv2.drawContours(color_warp, pts, -1, (255,0,0), thickness=50)
    pts = np.transpose(np.vstack([lines[1].bestx, ploty])).reshape((-1, 1, 2)).astype(np.int32)
    cv2.drawContours(color_warp, pts, -1, (0, 0, 255), thickness=50)
    #plt.plot(line.current_xfit, ploty, color='yellow')

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(img, 1, newwarp, 0.3, 0)

    # Write text output
    mean_curvature = (lines[0].radius_of_curvature + lines[1].radius_of_curvature) / 2
    cv2.putText(result, "Radius of Curvature = {}(m)".format(int(mean_curvature)), (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)
    cv2.putText(result, "Vehicle is {:.2f}m {} of center".format(abs(vehicle_position.item()), side), (50, 140), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)

    # plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    # print(vehicle_position)
    # plt.show()
    return result

# Accepts RGB image and returns image with lane identified
def process_image(input_img):
    # referece global primatives
    global ploty, frame

    # track which frame we're on.
    frame += 1

    # Convert input image to BGR since we will be processing it with OpenCV
    input_img = cv2.cvtColor(input_img, cv2.COLOR_RGB2BGR)
    # if DEBUG:
    #     cv2.imwrite("./writeup_images/input.jpg", input_img)

    # Apply a distortion correction to raw images.
    img = cv2.undistort(input_img, mtx, dist, None, mtx)
    # if DEBUG:
    #     cv2.imwrite("./writeup_images/undistorted.jpg", img)

    # Threshold x gradient
    sobel_x_binary = abs_sobel_thresh(img, 20, 100)
    # if DEBUG:
    #     plt.imshow(sobel_x_binary, cmap="gray")
    #     plt.savefig("./writeup_images/sobel_x.jpg")
    #     plt.close()

    # Threshold s channel
    hls_s_binary = hls_s_thresh(img, 170, 255)
    # if DEBUG:
    #     plt.imshow(hls_s_binary, cmap="gray")
    #     plt.savefig("./writeup_images/hls_s_binary.jpg")
    #     plt.close()

    # Stack each channel to view their individual contributions in green and blue respectively
    # This returns a stack of the two binary images, whose components you can see as different colors
    # color_binary = np.dstack((np.zeros_like(sobel_x_binary), sobel_x_binary, hls_s_binary)) * 255

    # Combine the two binary thresholds
    combined_binary = np.zeros_like(sobel_x_binary)
    combined_binary[(sobel_x_binary == 1) | (hls_s_binary == 1)] = 1
    # if DEBUG:
    #     plt.imshow(combined_binary, cmap="gray")
    #     plt.savefig("./writeup_images/combined_binary.jpg")
    #     plt.close()

    binary_warped, Minv = perspective_transform(combined_binary)
    # if DEBUG:
    #     plt.imshow(binary_warped, cmap="gray")
    #     plt.savefig("./writeup_images/binary_warped.jpg")
    #     plt.close()

    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])

    for line in lines:
        if line.detected:
            update_line(binary_warped, line)
        else:
            find_line(binary_warped, line)

    check_sanity(lines, binary_warped)

    return cv2.cvtColor(generate_output(lines, Minv, binary_warped, img), cv2.COLOR_BGR2RGB)

def test_images():
    global lines
    lines = [Line("left", 1), Line("right", 1)]

    # Get image filenames
    img_filenames = glob("./test_images/test*.jpg")

    for img_filename in img_filenames:
        # Convert image to RGB before passing because that's what process_image expects
        result = process_image(cv2.cvtColor(cv2.imread(img_filename), cv2.COLOR_BGR2RGB))
        # if DEBUG:
        #     cv2.imwrite("./writeup_images/result.jpg", cv2.cvtColor(result, cv2.COLOR_RGB2BGR))

        cv2.imwrite("./output_images/" + img_filename.split("/")[-1], cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
        # cv2.imshow("image", result)
        #cv2.waitKey(800)

def test_video():
    global lines
    lines = [Line("left", 15), Line("right", 15)]
    ## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
    ## To do so add .subclip(start_second,end_second) to the end of the line below
    ## Where start_second and end_second are integer values representing the start and end of the subclip
    ## You may also uncomment the following line for a subclip of the first 5 seconds
    #clip = VideoFileClip("./project_video.mp4").subclip(39, 42)
    clip = VideoFileClip("./project_video.mp4")

    # Process the video
    result = clip.fl_image(process_image)  # NOTE: this function expects color images!!

    # Save the processed video
    result.write_videofile("./output_images/output_video.mp4", audio=False)



# Read in saved calibration data
calibration_data = pickle.load(open("./calibration.p", "rb"))
mtx = calibration_data["mtx"]
dist = calibration_data["dist"]

test_images()
test_video()



