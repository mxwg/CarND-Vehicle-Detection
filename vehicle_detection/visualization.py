import cv2
import numpy as np
import math
BLUE = (0, 0, 255)

from vehicle_detection.features import single_img_features

def draw_bounding_boxes(img, boxes, color=BLUE, thickness=6):
    """Draws bounding boxes from a list onto an image."""
    result = img.copy()
    for box in boxes:
        cv2.rectangle(result, box[0], box[1], color, thickness)
    return result

def dist(loc1, loc2):
    return math.sqrt((loc1[0]-loc2[0])**2 + (loc1[1]-loc2[1])**2)

def draw_labeled_bboxes(img, labels, params, clf, scaler, locations):
    result = img.copy()
    already_drawn = []
    for loc in locations:
        if loc.tracked > 5:
            for previous in already_drawn:
                if dist(previous, loc.position) < 50:
                    print("\t\t\t\t\t\t\t\tduplicate {} reset.".format(loc.id))
                    loc.tracked = -10
                    continue
            cv2.rectangle(result, loc.bbox[0], loc.bbox[1], (0,255,0), 6)
            #print("labeling car {}".format(loc.id))
            # verify with prediction
            test_img = cv2.resize(img[loc.bbox[0][1]:loc.bbox[1][1], loc.bbox[0][0]:loc.bbox[1][0]], (64, 64))
            features = single_img_features(test_img,
                              cell_per_block=params['cell_per_block'],
                              color_space=params['colorspace'],
                              hog_channel=params['hog_channel'],
                              orient=params['orient'],
                              pix_per_cell=params['pix_per_cell'],
                             hist_feat=params['hist_feat'])
            #5) Scale extracted features to be fed to classifier
            test_features = scaler.transform(np.array(features).reshape(1, -1))
            #6) Predict using your classifier
            prediction = clf.predict(test_features)
            if prediction == 1:
                loc.tracked += 2
            else:
                small = 64*64#/(1280*720)
                size = (loc.bbox[1][0]-loc.bbox[0][0])*(loc.bbox[1][1]-loc.bbox[0][1])
                print("size", size, size/small)
                factor = size/small
                before = loc.tracked
                if factor > 5:
                    loc.tracked = int(loc.tracked / 1.1)
                elif factor > 1:
                    loc.tracked = int(loc.tracked / 1.05)
                else:
                    loc.tracked -= 2
                print("{}-- {} -> {}".format(loc.id, before, loc.tracked))
            already_drawn.append(loc.position)
    return result
