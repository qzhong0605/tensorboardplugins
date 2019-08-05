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
from tensorboard.plugins.inference.model_prediction import Inference

class Infer:
  def __init__(
      self,
      model_path = None,
      model_type = None):
    self.model_path = model_path
    self.model_type = model_type
    self.data_path = None

  def start(self,datapath,batchsize):
    self.data_path = datapath
    model = self.classification()
    result = model.feature(self.data_path,batchsize)
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
