import os
import time
import glob
import cv2
import numpy as np
import matplotlib.image as mplimg
import math
import pickle
import shutil
from scipy.ndimage.measurements import label


from vehicle_detection.sliding_window import search_windows, get_windows
from vehicle_detection.heatmap import get_heat_map, get_locations, suppress
from vehicle_detection.visualization import draw_bounding_boxes, draw_labeled_bboxes, dist


with open('hog_svm.p', 'rb') as f:
    (params, clf, scaler) = pickle.load(f)

print("Using params:", params)

# Get images
input_folder = "test_images"
#input_folder = "track_images"
input_folder = "track_images2"
output_folder = "output_images"

try:
    shutil.rmtree(output_folder)
except FileNotFoundError:
    pass
os.mkdir(output_folder)

def save(prefix, image_name, image, cmap=None, output=True):
    if output:
        file_name = os.path.join(output_folder, prefix + "_" + os.path.basename(image_name))
        mplimg.imsave(file_name, image, cmap=cmap)

all_images = glob.glob(os.path.join(input_folder, "*.png"))

from collections import deque
maps = deque(maxlen=5)


import string, random
class Loc(object):
    def __init__(self, position, bbox):
        self.tracked = 0
        self.position = position
        self.bbox = bbox
        self.id = ''.join(random.choice(string.ascii_uppercase) for _ in range(4))
    def matches(self, location):
        d = dist(self.position, location)
        return d <= 30
    def update(self, location, box):
        x = int((self.position[0] + location[0])/2)
        y = int((self.position[1] + location[1])/2)
        self.position = (x, y)
        box_min = (int((self.bbox[0][0] + box[0][0])/2), int((self.bbox[0][1] + box[0][1])/2))
        box_max = (int((self.bbox[1][0] + box[1][0])/2), int((self.bbox[1][1] + box[1][1])/2))
        self.bbox = (box_min, box_max)
        self.tracked += 1
    def __str__(self):
        return "Car {} at {} tracked {} times.".format(self.id, self.position, self.tracked)

locations = []


def apply_pipeline(img, img_name, output=False):
    global locations
    # Get list of window locations
    windows = get_windows(img)

    # Classify all windows with the SVM
    hits = search_windows(img, windows, clf, scaler,
                          cell_per_block=params['cell_per_block'],
                          color_space=params['colorspace'],
                          hog_channel=params['hog_channel'],
                          orient=params['orient'],
                          pix_per_cell=params['pix_per_cell'],
                         hist_feat=params['hist_feat'])


    # Draw boxes around the hits
    hit_img = draw_bounding_boxes(img, hits)
    save("hits", img_name, hit_img, output=output)

    # Build a heat map of the most recent hits
    heat = get_heat_map(img, hits, threshold=1)
    #save("heatmap", img_name, heat, cmap='hot', output=output)

    # Average over the last 5 heat maps
    maps.append(heat)
    avg_heat = sum(maps)
    avg_heat[avg_heat <= 4] = 0 # threshold weak signals

    # Find labels corresponding to heat map clusters
    labels = label(avg_heat)

    # Suppress very small and spurious activations in the average and current heat map
    suppress(avg_heat, heat, labels, img, params, clf, scaler, threshold=15)
    maps.pop() # update with suppressed heatmap for next time
    maps.append(heat)
    #save("heatmap_sup", img_name, heat, cmap='hot', output=output)
    save("avgheatmap", img_name, avg_heat, cmap='hot', output=output)

    # Find the clusters again
    labels = label(avg_heat)

    # Get the locations of all clusters
    current_loc, current_bbox = get_locations(avg_heat, labels)

    # Update the locations of previously tracked clusters
    for loc in locations:
        prev_t = loc.tracked
        for current, box in zip(current_loc, current_bbox):
            if loc.matches(current):
                loc.update(current, box)
                #print(loc)
        if prev_t == loc.tracked:
            loc.tracked -= 2

    # Add new clusters to the tracking list
    for current, box in zip(current_loc, current_bbox):
        tracked = False
        for loc in locations:
            if loc.matches(current):
                tracked = True
        if not tracked:
            new_car = Loc(current, box)
            locations.append(new_car)
            #print("Adding new car: {}".format(new_car))

    # Prune tracked locations
    locations = [loc for loc in locations if loc.tracked >= 0]


    # Draw boxes around the tracked locations
    final = draw_labeled_bboxes(img, labels, params, clf, scaler, locations)
    save("detections", img_name, final, output=output)

    # Save an augmented detection map
    if img_name != "unknown":
        heat_name = os.path.join(output_folder, "avgheatmap" + "_" + os.path.basename(img_name))
        hm = mplimg.imread(heat_name)
        hm = cv2.cvtColor(cv2.imread(heat_name), cv2.COLOR_BGR2RGB)
        aug_heat = cv2.addWeighted(final, 1, hm, 0.6, 0)
        save("augmented", img_name, aug_heat, output=output)
    return final



if __name__ == '__main__':
    t1 = time.time()
    for img_name in sorted(all_images):
        print("image: {}".format(img_name))
        #img = mplimg.imread(img_name)
        img = cv2.cvtColor(cv2.imread(img_name), cv2.COLOR_BGR2RGB)

        #print("stats:", np.mean(img), np.std(img), np.max(img), np.min(img))
        apply_pipeline(img, img_name, output=True)
    print("Done, took {:.2f} s.".format(time.time() - t1))
