import os
import cv2
import numpy as np
from constants import*
import tensorflow as tf
from Format_Image import format_image
from NSFW_test import*

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
model = NSFW_model()
sess = tf.Session()  
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())
saver.restore(sess, MODEL_SAVE_PATH) 

video_counter = 0
for filename in os.listdir(VIDEO_SAVE_PATH):
  cap = cv2.VideoCapture(VIDEO_SAVE_PATH+'/'+filename)
  print('PROCESSING:  ',repr(filename))
  frame_count= 0
  saved_frame_count = 0
  NSFW_Score = 0
  NSFW_Count = 0
  while(True):
    ret, frame = cap.read()
    if ret == True and frame_count%50==0:
      yshape = np.zeros((1,),dtype=np.int64)
      resized_frame = format_image(frame)
      reshaped_frame = np.reshape(resized_frame,(1,IMAGE_SIZE,IMAGE_SIZE,1))
      predicted_label = sess.run(model.prediction,feed_dict={model.X: reshaped_frame,model.y: yshape,model.is_training: False})
      if predicted_label== 1: NSFW_Count +=1
      saved_frame_count+=1
      cv2.putText(frame, LABELS[int(predicted_label)], (50, 50),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
      cv2.imshow('frame',frame)
      cv2.waitKey(1)
    if ret == False:
      break 
    frame_count+=1    
   
  if saved_frame_count!=0:
    video_counter+=1
    NSFW_Score = NSFW_Count/saved_frame_count  
    print('*********VIDEO_SUMMARY*********')
    print('VIDEO_NUMBER: ',video_counter)
    print('NAME: ',repr(filename))
    print('NUMBER_OF_PROCESSED_FRAMES: ',frame_count)
    print('NSFW_SCORE: ',NSFW_Score)
    if(NSFW_Score > 0.50):
      print('THIS VIDEO HAS HIGH NSFW CONTENT')
    print('**************xxx**************')
  else:
    print('Cannot read frames. Please Check FilePath!')    
cap.release()
cv2.destroyAllWindows()    