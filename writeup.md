**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[hog]: ./images/hog_features.png
[sliding_windows]: ./images/sliding_windows.png
[detected1]: ./images/detected1.png
[detected2]: ./images/detected2.png
[failure]: ./images/failure.png
[hits]: ./images/avgheatmap_output_00752.png
[heatmap]: ./images/hits_output_00752.png
[augmented]: ./images/augmented_output_00752.png

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for SVN training can be found in the Jupyter Notebook `SVN-Training.ipynb`.

The function `extract_features` takes a list of images (one for vehicles, one for non-vehicles), loads the images and computes all features.
The images are read with `cv2.imread` and converted to RGB via `cv2.cvtColor`.
The features are calculated with the function `single_img_features` from `vehicle_detection/features.py` (line 5).

An example for HOG features can be seen below.

![Example HOG features][hog]


####2. + 3. Explain how you settled on your final choice of HOG parameters and describe how you trained a classifier

For finding the HOG parameters I used a small random grid search in the `SVN-Training.ipynb` notebook.

It trained a linear SVM on random combinations of colorspaces, orientations, pixels per cell, cells per block and HOG channels.

The training was done using the function `train_and_evaluate_svm`.

For each training, the final accuracy of the SVM was printed and I chose a parameter combination that achieved over .98 accuracy
(`p =  {'pix_per_cell': 8, 'colorspace': 'YCrCb', 'cell_per_block': 2, 'orient': 9, 'hog_channel': 'ALL', 'hist_feat': False}`).

The parameters, SVM and scaler were then pickled and saved to disk.


###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The sliding window approach is implemented in the file `vehicle_detection/sliding_window.py`.

The function `get_windows` creates a list of window positions that are hand-tuned to the area of the image where cars can be expected.
It is a combination of coarser boxes in the foreground and smaller boxes towards the horizon.

A visualization of the windows can be seen below.
![Sliding windows][sliding_windows]


####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Via the file `pipeline.py`, I incrementally built up my pipeline.
I implemented a `save` function that can save images of intermediate states to the harddisk.

There are some mechanisms to reject false detections.
First, a average heatmap over the last 5 detections is used for finding the final candidate locations of cars.
The heatmap is created using `get_heat_map` in `vehicle_detection/heatmap.py`.

In this heatmap, some basic thresholding of the activation is employed (`pipeline.py`) and another filter rejects regions of very small sizes
(function `suppress` in `vehicle_detection/heatmap.py`.

The final detection results can be seen below:

![Detection example][detected1]

![Detection example][detected2]

---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_augmented.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

In the function `apply_pipeline` in `pipeline.py`, the labels extracted via `scipy.ndimage.measurements.label()` from a heatmap 
are used to compute the center for each detected cluster (`get_locations` in `vehicle_detection/heatmap.py`).

For each cluster, the center position, a bounding box and a tracking score are tracked via an object of the type `Loc`
(`pipeline.py` line 47)
If the location is near enough to a previously tracked location, the tracked location is updated with the new data by
taking the average of the tracked and the new information.

If the location is not tracked yet, a new object is created and added to a global list (line 67 in `pipeline.py`).

The tracking score is increased when a cluster is detected and decreased when it is not.

If the tracking for a location is lost (determined via the tracking score), it is purged from the list.

Finally, the bounding boxes of all locations in the tracking list are drawn onto the image.

Here is an example image with the clusters drawn, the heatmap created from it and the final tracked locations (overlayed with the
heatmap:

![Multiple hits][hits]

![A heatmap][heatmap]

![Augmented result][augmented]

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

There were some problems with consistently loading and converting images that led to the classifier working well in training, but
not on the actual data.
This could be resolved, however, by carefully looking at the detections of the classifier on some example images.

The pipeline now works reasonably well, but still has some outliers (shown below).
The reason for this is the classifier, as it still classifies slanted lane markings as cars in many cases.
This could probably be remedied by adding more of these instances to the negative training images or even by using another
classifier altogether.

The tracking is also very basic at the moment, it could be improved by a more sophisticated tracking score handling (e.g.
re-detecting cars in the proposed bounding box, or other ways of scoring the quality of a match).

![The still detected outlier][failure]
