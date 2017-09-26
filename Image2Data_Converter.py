import numpy as np
from constants import*
import os
import cv2

def format_image(frame):
	if len(frame.shape) > 2 and frame.shape[2] == 3:
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	else:
		return(0,0)
	frame = cv2.resize(frame, (48, 48), interpolation = cv2.INTER_CUBIC)
	return(frame,1)

path = '/home/linux/Documents/NSFW_dataset/FINAL_SORT/NSFW'
image_counter = 0
images = []
for filename in os.listdir(path):
	image = cv2.imread(path+'/'+filename)
	image_counter+=1
	resized_image,is_true = format_image(image)
	if is_true==1:
		reshaped_image = np.reshape(resized_image,(-1))
		reshaped_image = np.append(reshaped_image,[1])
		images.append(reshaped_image)
		print(images[0].shape)	
print("Total: " + str(len(images)))
np.save('FINAL_SORT/Data48/data_NSFW.npy', images)
train_inputs = np.load('FINAL_SORT/Data48/data_NSFW.npy')
print(train_inputs.shape)