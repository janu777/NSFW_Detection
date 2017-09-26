import numpy as np
import cv2
from constants import*
def format_image(frame):
  if len(frame.shape) > 2 and frame.shape[2] == 3:
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  else:
    frame = cv2.imdecode(frame, cv2.CV_LOAD_IMAGE_GRAYSCALE)
  frame = cv2.resize(frame, (IMAGE_SIZE, IMAGE_SIZE), interpolation = cv2.INTER_CUBIC)
  return frame	