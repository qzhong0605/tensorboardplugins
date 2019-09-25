from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

'''
model_path = '/root/Desktop/ckpt'
data_path = '/root/Desktop/data/mnist/mnist.tfrecords'
'''
import tensorflow as tf
import requests
import shutil
import json
from tensorboard.plugins.inference.model_prediction import Inference

class Infer:
  def __init__(
      self,
      model_path = None,
      model_type = None):
    self.model_path = model_path
    self.model_type = model_type
    self.tensor_name = None
    self.tensor_channel_num = None
    self.data_path = None
    self.input_cache=None
    self.label_cache=None

  def start(
          self,
          datapath,
          batchsize,
          inputshape,
          outputshape):
    self.data_path = datapath
    model = self.classification()
    result, self.input_cache, self.label_cache = model.feature(
                                                     self.data_path, batchsize, inputshape, outputshape)
    self.tensor_name = model.tensor_name
    self.tensor_channel_num = model.tensor_channel_num
    return result,self.tensor_name,self.tensor_channel_num

  def edit(self,datapath,batchsize,edit_log):
    edit_log = json.loads(edit_log)
    trans_log = edit_log[0].copy()
    for item in trans_log:
      trans_log[item] = []
    for i in range(0,len(edit_log)):
      for item in trans_log:
        trans_log[item].append(edit_log[i][item])
    self.data_path = datapath
    model = self.classification()
    result = model.feature_edit(
        self.input_cache, 
        self.label_cache, 
        batchsize,trans_log['batch'], 
        trans_log['x'], 
        trans_log['y'], 
        trans_log['c'],
        trans_log['new'])
    self.tensor_name = model.tensor_name
    self.tensor_channel_num = model.tensor_channel_num
    return result

  def config(self,channel,layer_channel):
    model = self.classification()
    result = model.channel(channel,layer_channel)
    return result

  def classification(self):
    model = Inference(self.model_path,self.model_type)
    return model

class Common:
  def __init__(self,path):
    self.path=path
  def delete(self):
    shutil.rmtree(self.path)
    return {'ifDone': True}

class splitfig:
  def __init__(self,url = None):
    self.url = url
    self.split(self.url)
    print("done!")

  def split(self,url):
    r = requests.get(url)
    print("111")
    with open('/root/Desktop/dadsa.png', 'wb') as w:
      w.write(r.content)
