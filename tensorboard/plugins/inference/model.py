import tensorflow as tf
import numpy as np

class Network(object):
    
  def __init__(self):
    self.fch_nodes = 512
    self.drop_prob = 0.5
    self.learning_rate = 0.0001
    self.global_step = tf.Variable(0,trainable = False)
    #self.x = tf.placeholder(tf.float32,[None,784])
    self.init_x = tf.placeholder(tf.float32,[None,784])
    self.x = tf.reshape(self.init_x, [-1, 28, 28, 1])
    self.label = tf.placeholder(tf.float32,[None,10])
        
    #self.w = tf.Variable(tf.zeros([784,10]))
    #self.b = tf.Variable(tf.zeros([10]))
    #self.y = tf.nn.softmax(tf.matmul(self.x,self.w) + self.b)
    self.w = self.xavier_init(self.fch_nodes, 10)
    self.b = self.biases_init([10])
    self.hl = HiddenLayers(self.x, self.fch_nodes, self.drop_prob)
    self.y = tf.nn.softmax(tf.matmul(self.hl.predict(),self.w) + self.b)
    #self.y = tf.nn.softmax(tf.matmul(self.hidden_layers(self.x,self.fch_nodes,self.drop_prob),self.w) + self.b)
    #self.loss = -tf.reduce_mean(self.label * tf.log(self.y) + 1e-10)
    self.loss = tf.reduce_mean(-tf.reduce_sum(self.label * tf.log(self.y), reduction_indices = [1]))
    
    #self.train = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss,global_step = self.global_step)
    self.train = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step = self.global_step)

    predict = tf.equal(tf.argmax(self.label,1),tf.argmax(self.y,1))
    self.accuracy = tf.reduce_mean(tf.cast(predict,tf.float32))
  
    self.variable_names = [v.name for v in tf.trainable_variables()]

  def weight_init(self,shape):
    weights = tf.truncated_normal(shape, stddev=0.1,dtype=tf.float32)
    return tf.Variable(weights)

  def biases_init(self,shape):
    biases = tf.random_normal(shape,dtype=tf.float32)
    return tf.Variable(biases)

  def get_random_batchdata(self,n_samples, batchsize):
    start_index = np.random.randint(0, n_samples - batchsize)
    return (start_index, start_index + batchsize)

  def xavier_init(self,layer1, layer2, constant = 1): #random init w
    Min = -constant * np.sqrt(6.0 / (layer1 + layer2))
    Max = constant * np.sqrt(6.0 / (layer1 + layer2))
    return tf.Variable(tf.random_uniform((layer1, layer2), minval = Min, maxval = Max, dtype = tf.float32))

  def conv2d(self,x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')

  def max_pool_2x2(self,x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

class HiddenLayers(object):
  def __init__(self, x, fch_nodes, drop_prob):
    self.w_conv1 = self.weight_init([5, 5, 1, 16])                            
    self.b_conv1 = self.biases_init([16])
    self.h_conv1 = tf.nn.relu(self.conv2d(x, self.w_conv1) + self.b_conv1)  
    self.h_pool1 = self.max_pool_2x2(self.h_conv1)

    self.w_conv2 = self.weight_init([5, 5, 16, 32])                             
    self.b_conv2 = self.biases_init([32])
    self.h_conv2 = tf.nn.relu(self.conv2d(self.h_pool1, self.w_conv2) + self.b_conv2)    
    self.h_pool2 = self.max_pool_2x2(self.h_conv2)

    self.h_fpool2 = tf.reshape(self.h_pool2, [-1, 7*7*32])
    self.w_fc1 = self.xavier_init(7*7*32, fch_nodes)
    self.b_fc1 = self.biases_init([fch_nodes])
    self.h_fc1 = tf.nn.relu(tf.matmul(self.h_fpool2, self.w_fc1) + self.b_fc1)
    self.h_fc1_drop = tf.nn.dropout(self.h_fc1, keep_prob=drop_prob)

  def transpose(self,conv):
    print(conv.get_shape())
    conv_transpose = tf.transpose(conv, [0, 3, 1, 2])
    return conv_transpose

  def weight_init(self,shape):
    weights = tf.truncated_normal(shape, stddev=0.1,dtype=tf.float32)
    return tf.Variable(weights)

  def biases_init(self,shape):
    biases = tf.random_normal(shape,dtype=tf.float32)
    return tf.Variable(biases)

  def get_random_batchdata(self,n_samples, batchsize):
    start_index = np.random.randint(0, n_samples - batchsize)
    return (start_index, start_index + batchsize)

  def xavier_init(self,layer1, layer2, constant = 1):
    Min = -constant * np.sqrt(6.0 / (layer1 + layer2))
    Max = constant * np.sqrt(6.0 / (layer1 + layer2))
    return tf.Variable(tf.random_uniform((layer1, layer2), minval = Min, maxval = Max, dtype = tf.float32))

  def conv2d(self,x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')

  def max_pool_2x2(self,x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

  def predict(self):
    return self.h_fc1_drop

  def hidden_layers(self,x,fch_nodes):
    w_conv1 = self.weight_init([5, 5, 1, 16])                            
    b_conv1 = self.biases_init([16])
    h_conv1 = tf.nn.relu(self.conv2d(x, w_conv1) + b_conv1)    
    h_pool1 = self.max_pool_2x2(h_conv1)

    w_conv2 = self.weight_init([5, 5, 16, 32])                             
    b_conv2 = self.biases_init([32])
    h_conv2 = tf.nn.relu(self.conv2d(h_pool1, w_conv2) + b_conv2)    
    h_pool2 = self.max_pool_2x2(h_conv2)

    h_fpool2 = tf.reshape(h_pool2, [-1, 7*7*32])
    w_fc1 = self.xavier_init(7*7*32, fch_nodes)
    b_fc1 = self.biases_init([fch_nodes])
    h_fc1 = tf.nn.relu(tf.matmul(h_fpool2, w_fc1) + b_fc1)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob=self.drop_prob)
    return h_fc1_drop
