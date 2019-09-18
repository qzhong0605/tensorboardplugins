#from tensorboard.plugins.inference.model import Network
from tensorboard.plugins.inference.ReadTFRecord import read_and_decode
from tensorboard.plugins.inference.refresh_board import pred_refresh, fea_refresh
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import math
import cv2
import os

class Inference(object):
    
  def __init__(self,
               model_path = None,
               model_type = None):
    tf.reset_default_graph() 
    self.model_path = model_path
    self.model_type = model_type
    self.tensor_name = []
    self.tensor_in_graph = []
    self.tensor_channel_num=[]
    #self.net = Network()
    self.sess = tf.Session()
    self.sess.run(tf.global_variables_initializer())
    self.coord = tf.train.Coordinator()
    self.threads = tf.train.start_queue_runners(sess=self.sess, coord=self.coord)
    self.restore(self.model_path,self.model_type)
    self.loaded_graph = tf.get_default_graph()
    self.ifDone = False
    self.test_x=None
    self.test_label=None
    print('load susess')
    
  def restore(self,model_dir,model_type_name):
    saver = tf.train.import_meta_graph(model_dir+"/model-1000.meta")
    saver.restore(self.sess,model_dir+"/model-1000")

  def each_label_acc(self,label,pred):
    total_amount = [0]*10
    correct_amount = [0]*10
    for i in range(len(label)):
      total_amount[label[i]]+=1
      if(label[i]==pred[i]):
        correct_amount[label[i]]+=1        
    acc = np.true_divide(np.array(correct_amount),np.array(total_amount))
    return acc.tolist()

  def concact_features(self, conv_output):
    num_or_size_splits = int(math.sqrt(conv_output.shape[0]))
    img_out_list = []
    for j in range(num_or_size_splits):
      img_temp = conv_output[j*4]
      for i in range(num_or_size_splits-1):
        img_temp = np.concatenate((img_temp,conv_output[i+1+4*j]),axis=1)
      img_out_list.append(img_temp)
    img_out = img_out_list[0]
    for k in range(len(img_out_list)-1):
      img_out = np.concatenate((img_out,img_out_list[k+1]))
    return img_out

  def generate_tensor_single(self,conv):
    g = tf.Graph()     
    with tf.Session(graph=g) as sess:
      #conv_transpose = sess.run(tf.transpose(conv, [3, 1, 2, 0]))
      conv1_channel = sess.run(tf.transpose(conv[15], [2, 0, 1]))
    tensor_conv = tf.convert_to_tensor(conv1_channel)[:, :, :, np.newaxis]
    return tensor_conv

  def generate_tensor(self,conv):
    g = tf.Graph()     
    with tf.Session(graph=g) as sess:
      conv_transpose = sess.run(tf.transpose(conv, [3, 2, 1, 0]))
    self.tensor_channel_num.append(conv_transpose.shape[0])
    with tf.Session(graph=g) as sess:
      conv_concact = sess.run(tf.transpose(self.concact_features(conv_transpose), [2, 1, 0]))
    tensor_conv = tf.convert_to_tensor(conv_concact)[:, :, :, np.newaxis]
    return tensor_conv

  def predict(self,file_path,batchsize_s):
    batchsize = int(batchsize_s)
    filename_queue = tf.train.string_input_producer([file_path],num_epochs=None)
    img,label = read_and_decode(filename_queue,True,batchsize)
    #threads stop problem
    sv = tf.train.Supervisor()
    with sv.managed_session() as sess:
      test_x, test_label=sess.run([img,label])
    #acc = self.sess.run(self.net.accuracy,feed_dict = {self.net.init_x:test_x,self.net.label:test_label})
    y = self.sess.run(self.net.y,feed_dict = {self.net.init_x:test_x})
    values = self.sess.run(self.net.variable_names)
    print((np.array(values)).shape,(np.array(values[0])).shape,(np.array(values[1])).shape,(np.array(values[2])).shape,(np.array(values[3])).shape,(np.array(values[4])).shape,(np.array(values[5])).shape,(np.array(values[6])).shape,(np.array(values[7])).shape)
    h_conv1_dist=values[2]
    h_conv1_dist = np.transpose(h_conv1_dist, (2,3,0,1))
    h_conv1_dist = np.reshape(h_conv1_dist, (-1,25))
    print(h_conv1_dist[3],(np.array(h_conv1_dist)).shape)
    y_label = []
    y_pred = []
    for i in range(batchsize):
      y_label.append(np.argmax(test_label[i]))
      y_pred.append(np.argmax(y[i]))
    eachlabelacc = self.each_label_acc(y_label,y_pred)
    label = [0,1,2,3,4,5,6,7,8,9]
    data = []
    data.append(['label','accuracy'])
    for i in range(len(eachlabelacc)):
      data.append([label[i],eachlabelacc[i]])
    print(eachlabelacc,data)
    return {'acc': eachlabelacc, 'label': label, 'data':data}

  def accuracy(self,file_path,batchsize_s):
    batchsize = int(batchsize_s)
    filename_queue = tf.train.string_input_producer([file_path],num_epochs=None)
    img,label = read_and_decode(filename_queue,True,batchsize)#(batchsize,28,28)
    #threads stop problem
    sv = tf.train.Supervisor()
    with sv.managed_session() as sess:
      test_x, test_label=sess.run([img,label])
    acc = self.sess.run(self.net.accuracy,feed_dict = {self.net.init_x:test_x,self.net.label:test_label})
    return acc

  def channel(self,channel,layer_channel):
    layer = 0
    channel = int(channel)
    layer_channel = int(layer_channel)
    if(layer_channel==16):
      layer = 2
    #values = self.sess.run(self.net.variable_names)
    values = self.loaded_graph.get_tensor_by_name("layer1/Variable/Adam:0")
    #print((np.array(values)).shape,(np.array(values[0])).shape,(np.array(values[1])).shape,(np.array(values[2])).shape,(np.array(values[3])).shape,(np.array(values[4])).shape,(np.array(values[5])).shape,(np.array(values[6])).shape,(np.array(values[7])).shape)
    h_conv1_dist = self.sess.run(values)
    h_conv1_dist = np.transpose(h_conv1_dist, (2,3,0,1))
    h_conv1_dist = np.reshape(h_conv1_dist, (-1,25))
    weights = np.arange(len(h_conv1_dist[0]))
    data = []
    data.append(['weights','figure'])
    for i in range(len(weights)):
      data.append([weights[i].tolist(),h_conv1_dist[channel][i].tolist()])
    return {'data':data}

  def feature(self,file_path,batchsize_s):
    init_x = self.loaded_graph.get_tensor_by_name("input/Placeholder:0")
    label_y = self.loaded_graph.get_tensor_by_name("label/Placeholder:0")
    accuracy = self.loaded_graph.get_tensor_by_name("accuracy/Const:0")
    batchsize = int(batchsize_s)
    filename_queue = tf.train.string_input_producer([file_path],num_epochs=None)
    img,label = read_and_decode(filename_queue,True,batchsize)
    #threads stop problem
    sv = tf.train.Supervisor()
    with sv.managed_session() as sess:
      self.test_x, self.test_label=sess.run([img,label])
    self.get_all_tensor()
    acc, = self.sess.run(accuracy,feed_dict = {init_x:self.test_x,label_y:self.test_label})
    tf.reset_default_graph()
    g = tf.Graph()
    with g.as_default():
      feature_list = self.sess.run(self.tensor_in_graph, feed_dict={init_x:self.test_x})#list
    feature_tensor = []
    for i in range(len(self.tensor_in_graph)):
      feature_tensor.append(self.generate_tensor(feature_list[i]))
    print(self.tensor_channel_num)
    fea_refresh("/tmp/mnist/feature", feature_tensor, self.tensor_name)
    return acc,self.test_x,self.test_label

  def feature_edit(self,input_cache,label_cache,batchsize_s,batch,x,y,c,value):
    print("%%%%%%%%%%%%%%%%%%%")
    print(input_cache.shape)
    init_x = self.loaded_graph.get_tensor_by_name("input/Placeholder:0")
    label_y = self.loaded_graph.get_tensor_by_name("label/Placeholder:0")
    accuracy = self.loaded_graph.get_tensor_by_name("accuracy/Const:0")
    input_reshape = self.loaded_graph.get_tensor_by_name("input_reshape/Reshape:0")
    batchsize = int(batchsize_s)
    img_reshape = np.reshape(input_cache,(-1, input_reshape.shape[1], input_reshape.shape[2], input_reshape.shape[3]))
    print(batch[0],x[0],y[0],c[0],value[0])
    img_edit = self.edit(img_reshape,batch,x,y,c,value)#edit
    print(img_edit[batch[0],x[0],y[0],c[0]])
    img_reshape = np.reshape(img_edit,(-1,init_x.shape[1]))
    #threads stop problem
    self.get_all_tensor()
    acc = self.sess.run(accuracy,feed_dict = {input_reshape:img_edit, label_y:label_cache})
    tf.reset_default_graph()
    g = tf.Graph()
    with g.as_default():
      feature_list = self.sess.run(self.tensor_in_graph, feed_dict={input_reshape:img_edit})
    feature_tensor = []
    for i in range(len(self.tensor_in_graph)):
      feature_tensor.append(self.generate_tensor(feature_list[i]))
    fea_refresh("/tmp/mnist/feature", feature_tensor, self.tensor_name)
    return acc

  def edit(self,img,batch,x,y,c,value):
    new_img = img
    for i in range(len(batch)):
      new_img[batch[i],x[i],y[i],c[i]] = value[i]
    return new_img

  def get_all_tensor(self):
    print("############################################")
    v1 = self.sess.graph.get_operations()
    for i in range(len(v1)):
      if v1[i].name.startswith('layer'):
        print(v1[i].name)
    self.tensor_name.append("layer1/Conv2D:0")
    self.tensor_name.append("layer1/MaxPool:0")
    self.tensor_name.append("layer2/Conv2D:0")
    self.tensor_name.append("layer2/MaxPool:0")
    for i in range(len(self.tensor_name)):
      self.tensor_in_graph.append(self.loaded_graph.get_tensor_by_name(self.tensor_name[i]))
    print(self.tensor_name)
'''
    tf.reset_default_graph()
    g = tf.Graph()
    with g.as_default():
      conv1_16 = self.sess.run(self.net.hl.h_conv1, feed_dict={self.net.init_x:test_x})     # [1, 28, 28 ,16] 
    with tf.Session(graph=g) as sess:
      conv1_transpose = sess.run(tf.transpose(conv1_16, [3, 2, 1, 0]))
    with tf.Session(graph=g) as sess:
      conv1_concact = sess.run(tf.transpose(self.concact_features(conv1_transpose), [2, 1, 0])) 
    tensor_conv1 = tf.convert_to_tensor(conv1_concact)[:, :, :, np.newaxis]
    print(tensor_conv1.get_shape())
'''






