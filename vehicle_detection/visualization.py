import cv2
BLUE = (0, 0, 255)

def draw_bounding_boxes(img, boxes, color=BLUE, thickness=6):
    """Draws bounding boxes from a list onto an image."""
    result = img.copy()
    for box in boxes:
        cv2.rectangle(result, box[0], box[1], color, thickness)
    return result
