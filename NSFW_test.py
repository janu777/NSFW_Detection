import tensorflow as tf

class NSFW_model(object):
    def __init__(self):
        self.X = tf.placeholder(tf.float32, [None, 48, 48, 1 ])
        self.y = tf.placeholder(tf.int64, [None])
        self.is_training = tf.placeholder(tf.bool)
        Wconv1=tf.get_variable(shape=[5,5,1,64],name = 'Wconv1')
        bconv1=tf.get_variable(shape=[64],name = 'bconv1')
        Wconv2=tf.get_variable(shape=[5,5,64,128],name = 'Wconv2')
        bconv2=tf.get_variable(shape=[128],name = 'bconv2')
        W1=tf.get_variable(shape=[12*12*128,3072],name = 'W1')
        b1=tf.get_variable(shape=[3072],name = 'b1')
        W2=tf.get_variable(shape=[3072,3],name = 'W2')
        b2=tf.get_variable(shape=[3],name = 'b2')
        hconv1=tf.nn.relu(tf.nn.conv2d(self.X,Wconv1,strides=[1,1,1,1],padding='SAME')+bconv1)
        hbn1=tf.contrib.layers.batch_norm(hconv1,center=True, scale=True,is_training=self.is_training,scope='hbn1')
        hpool1=tf.nn.max_pool(hbn1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
        hconv2=tf.nn.relu(tf.nn.conv2d(hpool1,Wconv2,strides=[1,1,1,1],padding='SAME')+bconv2)
        hbn2=tf.contrib.layers.batch_norm(hconv2,center=True, scale=True,is_training=self.is_training,scope='hbn2')
        hpool2=tf.nn.max_pool(hbn2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
        hpool2_flat=tf.reshape(hpool2,(-1,12*12*128))
        hfc1=tf.nn.relu(tf.matmul(hpool2_flat,W1)+b1)
        hbn3=tf.contrib.layers.batch_norm(hfc1,center=True, scale=True,is_training=self.is_training,scope='hbn3')
        y_out=tf.matmul(hbn3,W2)+b2
        variable_list = [Wconv1,bconv1,Wconv2,bconv2,W1,b1,W2,b2]
        self.prediction=tf.argmax(y_out,1) 
          
            