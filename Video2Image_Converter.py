import os
import cv2
import numpy as np
from constants import*
from Format_Image import format_image

path = '/home/linux/Documents/NSFW_dataset/pictures'
video_counter = 0
images = []
saved_frame_count = 0
for filename in os.listdir(path):
	cap = cv2.VideoCapture(path+'/'+filename)
	print(repr(filename))
	video_counter+=1
	frame_count = 0
	while(True):
		ret, frame = cap.read()
		if ret == True and frame_count%50==0:
			resized_frame = format_image(frame)
			cv2.imwrite('/home/linux/Documents/NSFW_dataset/SORTED_IMAGES/frame_new'+str(saved_frame_count)+'.jpg',resized_frame)
			saved_frame_count += 1
			print(saved_frame_count)
		if ret == False:
			break	
		frame_count+=1
cap.release()
