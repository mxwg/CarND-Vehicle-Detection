import numpy as np
from skimage.feature import hog
import cv2

def get_windows(img):
    """Get a list of sliding windows to use."""
    w_foreground = slide_window(img, x_start_stop=[100, 1280], y_start_stop=[300, 720],
                       xy_window=(300,300), xy_overlap=(0.7, 0.7))
    w_very_far = slide_window(img, x_start_stop=[0, 1280], y_start_stop=[350, 550],
                         xy_window=(64,48), xy_overlap=(0.7, 0.7))
    w_far = slide_window(img, x_start_stop=[0, 1280], y_start_stop=[350, 550],
                         xy_window=(96,64), xy_overlap=(0.5, 0.5))
    w_medium_s = slide_window(img, x_start_stop=[0, 1280], y_start_stop=[400, 650],
                              xy_window=(128,96), xy_overlap=(0.7, 0.7))
    w_medium_l = slide_window(img, x_start_stop=[0, 1280], y_start_stop=[400, 660],
                              xy_window=(128,128), xy_overlap=(0.5, 0.5))
    w_medium_b = slide_window(img, x_start_stop=[18, 1280], y_start_stop=[380, 660],
                              xy_window=(164,128), xy_overlap=(0.7, 0.5))

    windows = w_medium_b + w_medium_s + w_medium_l + w_far
    windows = w_medium_b + w_medium_s + w_medium_l
    windows = w_far + w_medium_s + w_medium_l
    return windows

def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None],
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
    ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
    nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step)
    ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step)
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]

            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list

import os, pickle
import hashlib
cache_dir = "cache"

def search_windows(img, windows, clf, scaler, color_space='RGB',
                   spatial_size=(32, 32), hist_bins=32,
                   hist_range=(0, 256), orient=9,
                   pix_per_cell=8, cell_per_block=2,
                   hog_channel=0, hist_feat=True):

    h = hashlib.sha1(img).hexdigest()
    hn = os.path.join(cache_dir, h)
    if os.path.exists(hn):
        with open(hn, 'rb') as f:
            cached = pickle.load(f)
            return cached

    #1) Create an empty list to receive positive detection windows
    on_windows = []
    #2) Iterate over all windows in the list
    for window in windows:
        #3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
        #4) Extract features for that window using single_img_features()
        features = single_img_features(test_img, color_space=color_space,
                                       spatial_size=spatial_size, hist_bins=hist_bins,
                                       orient=orient, pix_per_cell=pix_per_cell,
                                       cell_per_block=cell_per_block,
                                       hog_channel=hog_channel, hist_feat=hist_feat)
        #5) Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        #6) Predict using your classifier
        prediction = clf.predict(test_features)
        #7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
            #8) Return windows for positive detections
    with open(hn, 'wb') as f:
        pickle.dump(on_windows, f)
        print("wrote to cache", hn)

    return on_windows



def single_img_features(img, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9,
                        pix_per_cell=8, cell_per_block=2, hog_channel=0, hist_feat=True):
    #1) Define an empty list to receive features
    img_features = []
    #2) Apply color conversion if other than 'RGB'
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else: feature_image = np.copy(img)
    if hist_feat == True:
        # Apply color_hist()
        from vehicle_detection.histogram import color_hist
        hist_features = color_hist(feature_image, nbins=hist_bins)
        img_features.append(hist_features)
    #3) Compute spatial features
    if hog_channel == 'ALL':
        hog_features = []
        for channel in range(feature_image.shape[2]):
            hog_features.extend(get_hog_features(feature_image[:,:,channel],
                                                 orient, pix_per_cell, cell_per_block,
                                                 vis=False))
        hog_features = np.ravel(hog_features)
    else:
        hog_features = get_hog_features(feature_image[:,:,hog_channel], orient,
                                        pix_per_cell, cell_per_block, vis=False)
    img_features.append(hog_features)
    #8) Append features to list
    #img_features.append(np.concatenate(hog_features))
    #img_features.append(hog_features)

    #9) Return concatenated array of features
    return np.concatenate(img_features)

# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block,
                        vis=False):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient,
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block),
                                  normalise=True,
                                  visualise=vis)
        return features, hog_image
    # Otherwise call with one output
    else:
        features = hog(img, orientations=orient,
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block),
                       normalise=True,
                       visualise=vis)
        return features


def get_heat_map(image, hits, threshold=2):
    heat = np.zeros_like(image[:,:,0]).astype(np.float)

    def add_heat(heatmap, bbox_list):
        # Iterate through list of bboxes
        for box in bbox_list:
            # Add += 1 for all pixels inside each bbox
            # Assuming each "box" takes the form ((x1, y1), (x2, y2))
            heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

        # Return updated heatmap
        return heatmap# Iterate through list of bboxes

    def apply_threshold(heatmap, threshold):
        # Zero out pixels below the threshold
        heatmap[heatmap <= threshold] = 0
        # Return thresholded map
        return heatmap


    # Add heat to each box in box list
    heat = add_heat(heat,hits)

    # Apply threshold to help remove false positives
    heat = apply_threshold(heat, threshold)

    # Visualize the heatmap when displaying
    heatmap = np.clip(heat, 0, 255)
    return heatmap

def suppress(heat, labels, threshold=15):
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        minx = np.min(nonzerox)
        miny = np.min(nonzeroy)
        maxx = np.max(nonzerox)
        maxy = np.max(nonzeroy)
        xsize = maxx - minx
        ysize = maxy - miny
        if xsize < threshold or ysize < threshold:
            bbox = ((minx, miny), (maxx, maxy))
            cv2.rectangle(heat, bbox[0], bbox[1], (0, 0, 0), -1) # filled
    return heat


def draw_labeled_bboxes(img, labels):
    result = img.copy()
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        minx = np.min(nonzerox)
        miny = np.min(nonzeroy)
        maxx = np.max(nonzerox)
        maxy = np.max(nonzeroy)
        #bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        print("size",car_number, maxx-minx, maxy-miny)
        bbox = ((minx, miny), (maxx, maxy))
        # Draw the box on the image
        cv2.rectangle(result, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return result
