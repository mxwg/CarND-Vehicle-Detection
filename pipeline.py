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
from vehicle_detection.sliding_window import search_windows, get_windows, get_heat_map, draw_labeled_bboxes

import pickle

with open('hog_svm.p', 'rb') as f:
    (params, clf, scaler) = pickle.load(f)

# Get images
input_folder = "test_images"
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

all_images = glob.glob(os.path.join(input_folder, "*.jpg"))

def apply_pipeline(img, output=False):
    windows = get_windows(img)
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
                          pix_per_cell=params['pix_per_cell'])
    print("took {:.2f} s to find {} hits.".format(time.time()-t1, len(hits)))
    hit_img = draw_bounding_boxes(img, hits)
    save("hits", img_name, hit_img, output=output)
    heat = get_heat_map(img, hits, threshold=1)
    save("heatmap", img_name, heat, cmap='hot', output=output)

    labels = label(heat)
    final = draw_labeled_bboxes(img, labels)
    save("detections", img_name, final, output=output)

    heat_name = os.path.join(output_folder, "heatmap" + "_" + os.path.basename(img_name))
    hm = mplimg.imread(heat_name)
    aug_heat = cv2.addWeighted(final, 1, hm, 0.3, 0)
    save("augmented", img_name, aug_heat, output=output)



t1 = time.time()
for img_name in all_images:
    img = mplimg.imread(img_name)
    apply_pipeline(img, output=True)

    print("image: {}".format(img_name))
print("Done, took {:.2f} s.".format(time.time() - t1))
