# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML

from pipeline import apply_pipeline

import argparse

def process_image(img):
    return apply_pipeline(img, "unknown")

#white_output = 'project_video_augmented.mp4'
#filename = "challenge_video.mp4"
#filename = "test_video.mp4"
filename = "project_video.mp4"

name, ext = filename.split('.')
result_filename = name + "_augmented." + ext
print("input: {}, output: {}".format(filename, result_filename))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('subclipping')
    parser.add_argument('from_t', type=int)
    parser.add_argument('to_t', type=int)
    args = parser.parse_args()
    print("Processing subclip from {} to {}.".format(args.from_t, args.to_t))

    clip1 = VideoFileClip(filename).subclip(args.from_t, args.to_t)
    white_clip = clip1.fl_image(process_image) #NOTE: this function expects color
    print("processed, writing...")

    white_clip.write_videofile(result_filename, audio=False)
