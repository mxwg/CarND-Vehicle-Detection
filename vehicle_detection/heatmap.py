import cv2
import numpy as np

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

def get_locations(heat, labels):
    locations = []
    bboxes = []
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
        position = (minx + xsize//2, miny + ysize//2)
        bbox = ((minx, miny), (maxx, maxy))
        #print("\t\t\t\t\tCar {} is at {}...".format(car_number, position))
        locations.append(position)
        bboxes.append(bbox)
    return locations, bboxes

def suppress(heat, heat_orig, labels, img, params, clf, scaler, threshold=15):
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
        position = (minx + xsize//2, miny + ysize//2)
        #print("\t\t\t\t\tCar {} is at {}...".format(car_number, position))
        bbox = ((minx, miny), (maxx, maxy))
        if xsize < threshold or ysize < threshold:
            cv2.rectangle(heat, bbox[0], bbox[1], (0, 0, 0), -1) # filled
            cv2.rectangle(heat_orig, bbox[0], bbox[1], (0, 0, 0), -1) # filled


    return heat
