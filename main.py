import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#---------------------------------------Calibration---------------------------------------------------------
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.
chessboards = []

# Make a list of calibration images
images_path = glob.glob('camera_cal/calibration*.jpg')
nx = 9
ny = 6
# Step through the list and search for chessboard corners

for fname in images_path:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (nx,ny),None)

    # If found, add object points, image points
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (9,6), corners, ret)
        chessboards.append(img)
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)


#---------------------------------------Binary#---------------------------------------#
def to_binary(image):

    # Grayscale image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Sobel x
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    abs_sobelx = np.absolute(sobelx)
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
    
    # Threshold x gradient
    thresh_min = 30
    thresh_max = 150
    sxbinary = np.zeros_like(scaled_sobel)
    retval, sxthresh = cv2.threshold(scaled_sobel, thresh_min, thresh_max, cv2.THRESH_BINARY)
    sxbinary[(sxthresh >= thresh_min) & (sxthresh <= thresh_max)] = 1
    
    # Convert to HLS color space and separate the S channel
    hls = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2HLS)
    s_channel = hls[:, :, 2]
    
    # Convert to HSV color space and get mask 
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    yellow = cv2.inRange(hsv,np.array((0,100,100)),np.array((80,255,255)))
    white = cv2.inRange(image,np.array((200,200,200)),np.array((255,255,255)))
    mask = cv2.bitwise_or(white,yellow)
    
    # Threshold color channel
    s_thresh_min = 175
    s_thresh_max = 255
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1
    
    # Combine binary thresholds
    combined_binary = np.zeros_like(s_binary)
    combined_binary[(sxbinary == 1)] = 1
    combined_binary[(mask == 255)] = 1

    return combined_binary

#---------------------------------------Perspective Tranformation(Eye Bird view)#---------------------------------------#
def perspective_transform(img):

    img_size = img.shape

    src = np.array([[575. / 1280. * img_size[1], 460. / 720. * img_size[0]],
                    [705. / 1280. * img_size[1], 460. / 720. * img_size[0]],
                    [1127. / 1280. * img_size[1], 720. / 720. * img_size[0]],
                    [203. / 1280. * img_size[1], 720. / 720. * img_size[0]]], np.float32)

    dst = np.array([[320. / 1280. * img_size[1], 100. / 720. * img_size[0]],
                    [960. / 1280. * img_size[1], 100. / 720. * img_size[0]],
                    [960. / 1280. * img_size[1], 720. / 720. * img_size[0]],
                    [320. / 1280. * img_size[1], 720. / 720. * img_size[0]]], np.float32)
    #Compute the perspective transform, M, given source and destination points:
    M = cv2.getPerspectiveTransform(src, dst)
    #Compute the inverse perspective transform:
    M_inv = cv2.getPerspectiveTransform(dst, src)
    #Warp an image using the perspective transform, M:
    perspective_img = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)
    
    unperspective_img = cv2.warpPerspective(perspective_img, M_inv, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)
   
    return perspective_img,unperspective_img, src,dst, M_inv
    
#-----------------------------------------Draw lanes-----------------------------------------#
def draw_lane(perspective_img, undist_img, Minv,left_lane_xfitted, right_lane_xfitted,curvature_center):
    ploty = np.linspace(0, perspective_img.shape[0] - 1, perspective_img.shape[0])
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(perspective_img).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_lane_xfitted, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_lane_xfitted, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
   
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    
    newwarp = cv2.warpPerspective(color_warp, Minv, (color_warp.shape[1], color_warp.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(undist_img, 1, newwarp, 0.3, 0)

    
    xm_per_pix = 3.7 / 700

    vehicle_pose = perspective_img.shape[1] // 2

    dx = (vehicle_pose * xm_per_pix - curvature_center)  # Positive if on right, Negative on left

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(result, ('Curavature center = '+str(curvature_center) + ' m'),
                 (10, 100), font, 1, (255, 255, 255), 2)
    
    cv2.putText(result, ('vehicle position in lane = '+str(dx)+' m'),
                 (10, 200), font, 1, (255, 255, 255), 2)

    return result

#-----------------------------------------Finding lanes-----------------------------------------#
def find_lanes(binary_warped):
    # left_lane.recent_xfitted
    histogram = np.sum(binary_warped[int(binary_warped.shape[0] / 2):, :], axis=0)
    #plt.plot(histogram)
 
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Set the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0] / nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (
        nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (
        nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    left_lane_x = nonzerox[left_lane_inds]
    left_lane_y = nonzeroy[left_lane_inds]
    right_lane_x = nonzerox[right_lane_inds]
    right_lane_y = nonzeroy[right_lane_inds]
    
    # Fit a second order polynomial to each
    left_lane_fit = np.polyfit(left_lane_y, left_lane_x, 2)
    right_lane_fit = np.polyfit(right_lane_y, right_lane_x, 2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])

    left_lane_xfitted = left_lane_fit[0] * ploty ** 2 + left_lane_fit[1] * ploty + left_lane_fit[2]
    right_lane_xfitted = right_lane_fit[0] * ploty ** 2 + right_lane_fit[1] * ploty + right_lane_fit[2]

    
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    
    #plot
    #plt.imshow(out_img)
    #plt.plot(left_lane_xfitted, ploty, color='yellow')
    #plt.plot(right_lane_xfitted, ploty, color='yellow')
    #plt.xlim(0, 1280)
    #plt.ylim(720, 0)
    #plt.show()
    
    return  left_lane_fit, right_lane_fit, ploty, left_lane_xfitted, right_lane_xfitted
#-----------------------------------------Finding lanes-----------------------------------------#
def find_lanes_continous(binary_warped,left_lane_fit, right_lane_fit, ploty, left_lane_xfitted, right_lane_xfitted):

    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 50
    left_lane_inds = ((nonzerox > (left_lane_fit[0] * (nonzeroy ** 2) + left_lane_fit[1] * nonzeroy + left_lane_fit[2] - margin)) & (
    nonzerox < (left_lane_fit[0] * (nonzeroy ** 2) + left_lane_fit[1] * nonzeroy + left_lane.current_fit[2] + margin)))
    right_lane_inds = ((nonzerox > (right_lane_fit[0] * (nonzeroy ** 2) + right_lane_fit[1] * nonzeroy + right_lane_fit[2] - margin)) & (
    nonzerox < (right_lane_fit[0] * (nonzeroy ** 2) + right_lane_fit[1] * nonzeroy + right_lane_fit[2] + margin)))

    # Again, extract left and right line pixel positions
    left_lane_x = nonzerox[left_lane_inds]
    left_lane_y = nonzeroy[left_lane_inds]
    right_lane_x = nonzerox[right_lane_inds]
    right_lane_y = nonzeroy[right_lane_inds]
    # Fit a second order polynomial to each
    left_lane_fit = np.polyfit(left_lane_y, left_lane_x, 2)
    right_lane_fit = np.polyfit(right_lane_y, right_lane_x, 2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    left_lane_xfitted = left_lane_fit[0] * ploty ** 2 + left_lane_fit[1] * ploty + left_lane.current_fit[2]
    right_lane_xfitted = right_lane_fit[0] * ploty ** 2 + right_lane_fit[1] * ploty + right_lane_fit[2]

    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    window_img = np.zeros_like(out_img)
    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension
    y_eval = np.max(ploty)

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty * ym_per_pix, left_lane.recent_xfitted * xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty * ym_per_pix, right_lane.recent_xfitted * xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_lane.radius_of_curvature = ((1 + (
    2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit_cr[0])
    right_lane.radius_of_curvature = ((1 + (
    2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * right_fit_cr[0])

    left_lane.line_base_pos = left_lane.recent_xfitted[-1] * xm_per_pix
    right_lane.line_base_pos = right_lane.recent_xfitted[-1] * xm_per_pix

    return  left_lane_fit, right_lane_fit, ploty, left_lane_xfitted, right_lane_xfitted
#-----------------------------------------Curavature Calculation-----------------------------------------#
def curvature_calculation(left_lane_fit, right_lane_fit, ploty,left_lane_xfitted, right_lane_xfitted):
    
    # Define conversions in x and y from pixels space to meters (world space)
    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension
    
    y_eval = np.max(ploty)

    # Fitting new polynomials to x,y in world space (meters) for both lane lines
    left_fit_cr = np.polyfit(ploty * ym_per_pix, left_lane_xfitted * xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty * ym_per_pix, right_lane_xfitted * xm_per_pix, 2)
    
    # Calculate radius of curvature for each lane
    left_lane_radius_of_curvature = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit_cr[0])
    right_lane_radius_of_curvature = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * right_fit_cr[0])

    left_lane_line_base_pos = left_lane_xfitted[-1] * xm_per_pix
    right_lane_line_base_pos = right_lane_xfitted[-1] * xm_per_pix
    curvature_center = (left_lane_line_base_pos + right_lane_line_base_pos) / 2

    return curvature_center

#-----------------------------------------Image processing-----------------------------------------#
def process_image(img):
    global first
    first =0
    #test_image = cv2.cvtColor(cv2.imread(test_image_path),cv2.COLOR_BGR2RGB)
    undist_test_image = cv2.undistort(img, mtx, dist, None, mtx)
    
    binary_img = to_binary(undist_test_image)
    if first ==0:  
        # apply perspective transform
        perspective_img,unperspective_img, src,dst, M_inv= perspective_transform(binary_img)
        left_lane_fit, right_lane_fit, ploty, left_lane_xfitted, right_lane_xfitted=find_lanes(perspective_img)
        first+=1
    else:
        perspective_img,unperspective_img, src,dst, M_inv= perspective_transform(binary_img)
        left_lane_fit, right_lane_fit, ploty, left_lane_xfitted, right_lane_xfitted=find_lanes_continous(perspective_img,left_lane_fit, right_lane_fit, ploty, left_lane_xfitted, right_lane_xfitted)
   
    
    curvature_center= curvature_calculation(left_lane_fit, right_lane_fit, ploty, left_lane_xfitted, right_lane_xfitted)
   
    return draw_lane(perspective_img, undist_test_image, M_inv,left_lane_xfitted, right_lane_xfitted,curvature_center)

from moviepy.editor import VideoFileClip
from IPython.display import HTML

#-----------------------------------------Video processing-----------------------------------------#

output = 'project_video_lanes.mp4'
clip1 = VideoFileClip("project_video.mp4")  
clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
clip.write_videofile(output, audio=False)

test_images_path = glob.glob('test_images/*.jpg')
original_test_images = []
undist_test_images = []

for test_image_path in test_images_path:
    test_image = cv2.cvtColor(cv2.imread(test_image_path),cv2.COLOR_BGR2RGB)
    undist_test_image = cv2.undistort(test_image, mtx, dist, None, mtx)
   
    binary_img = to_binary(undist_test_image)
    # apply perspective transform
    perspective_img,unperspective_img, Minv ,src, dst= perspective_transform(binary_img)
    #plt.imshow(perspective_img) 
    
    #plt.show()
