import os
import time
import glob
import matplotlib.image as mplimg
import numpy as np
import cv2
import math
import shutil
from scipy.ndimage.measurements import label


from vehicle_detection.visualization import draw_bounding_boxes
from vehicle_detection.sliding_window import search_windows, get_windows, get_heat_map, draw_labeled_bboxes, suppress
from vehicle_detection.sliding_window import get_locations

import pickle

with open('hog_svm.p', 'rb') as f:
    (params, clf, scaler) = pickle.load(f)

print("Using params:", params)

# Get images
input_folder = "test_images"
input_folder = "track_images"
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

def dist(loc1, loc2):
    return math.sqrt((loc1[0]-loc2[0])**2 + (loc1[1]-loc2[1])**2)

import string, random
class Loc(object):
    def __init__(self, position):
        self.tracked = 0
        self.position = position
        self.id = ''.join(random.choice(string.ascii_uppercase) for _ in range(4))
    def matches(self, location):
        d = dist(self.position, location)
        print("dist", d)
        return d <= 25
    def update(self, location):
        x = (self.position[0] + location[0])/2
        y = (self.position[1] + location[1])/2
        self.position = (x, y)
        self.tracked += 1
    def __str__(self):
        return "\t\t\tCar {} at {} tracked {} times.".format(self.id, self.position, self.tracked)

locations = []


def apply_pipeline(img, img_name, output=False):
    global locations
    windows = get_windows(img)
    print(img.shape, np.max(img))
    print("Using {} windows.".format(len(windows)))
    #boxes = draw_bounding_boxes(img, windows)
    #save("boxes", img_name, boxes, output=output)

    # cell_per_block: 3, colorspace: HSV, hog_channel: ALL, orient: 6, pix_per_cell: 12,
    t1 = time.time()
    #hits = search_windows(img, windows, clf, scaler,
    #                     cell_per_block=3, color_space='HSV', hog_channel='ALL', orient=6, pix_per_cell=12)
    hits = search_windows(img, windows, clf, scaler,
                          cell_per_block=params['cell_per_block'],
                          color_space=params['colorspace'],
                          hog_channel=params['hog_channel'],
                          orient=params['orient'],
                          pix_per_cell=params['pix_per_cell'],
                         hist_feat=params['hist_feat'])
    print("took {:.2f} s to find {} hits.".format(time.time()-t1, len(hits)))
    hit_img = draw_bounding_boxes(img, hits)
    #save("hits", img_name, hit_img, output=output)
    heat = get_heat_map(img, hits, threshold=1)
    maps.append(heat)
    #save("heatmap", img_name, heat, cmap='hot', output=output)

    avg_heat = sum(maps)#/len(maps)
    avg_heat[avg_heat <= 4] = 0
    print("max", np.max(avg_heat), avg_heat.mean(), avg_heat.std())

    labels = label(avg_heat)
    current_loc = get_locations(avg_heat, labels)
    for loc in locations:
        prev_t = loc.tracked
        for current in current_loc:
            if loc.matches(current):
                loc.update(current)
                print(loc)
        if prev_t == loc.tracked:
            print("lost track of", loc)
            loc.tracked -= 1
    for current in current_loc:
        tracked = False
        for loc in locations:
            if loc.matches(current):
                tracked = True
        if not tracked:
            print("\t\t\tAdding new car: {}".format(current))
            locations.append(Loc(current))

    # prune tracked locations
    locations = [loc for loc in locations if loc.tracked >= 0]


    suppress(avg_heat, heat, labels, img, params, clf, scaler, threshold=15)
    maps.pop() # update with suppressed heatmap for next time
    maps.append(heat)
    #save("heatmap_sup", img_name, heat, cmap='hot', output=output)
    #save("avgheatmap", img_name, avg_heat, cmap='hot', output=output)

    labels = label(avg_heat)

    final = draw_labeled_bboxes(img, labels, params, clf, scaler, locations)
    save("detections", img_name, final, output=output)

   #if img_name == "unknown":
   #    heat_name = os.path.join(output_folder, "avgheatmap" + "_" + os.path.basename(img_name))
   #    hm = mplimg.imread(heat_name)
   #    hm = cv2.cvtColor(cv2.imread(heat_name), cv2.COLOR_BGR2RGB)
   #    aug_heat = cv2.addWeighted(final, 1, hm, 0.6, 0)
   #    save("augmented", img_name, aug_heat, output=output)
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
