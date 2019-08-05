from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

import tensorflow as tf
import numpy as np
FLAGS = None

def pred_writer(log_dir):
  file = open('/root/tensorboard/tensorboard/plugins/inference/cache/cache_each_label_acc.png', 'rb')
  data = file.read()
  file.close()
  image = tf.image.decode_png(data, channels=4)
  image = tf.expand_dims(image, 0)

  sess = tf.Session()
  writer = tf.summary.FileWriter(log_dir)
  summary_op = tf.summary.image("each_label_acc", image)

  summary = sess.run(summary_op)
  writer.add_summary(summary)

  writer.close()
  sess.close()

def fea_writer(log_dir,image1,image2,image3,image4):
  init_x = tf.placeholder(tf.float32,[None,28,28,1])
  x = tf.reshape(init_x, [-1, 784])
  sess = tf.Session()
  writer = tf.summary.FileWriter(log_dir)
  summary_op = tf.summary.image("conv1_16", image1, max_outputs=3)
  summary = sess.run(summary_op)
  writer.add_summary(summary)

  summary_op = tf.summary.image("pool1_16", image2, max_outputs=3)
  summary = sess.run(summary_op)
  writer.add_summary(summary)

  summary_op = tf.summary.image("conv2_32", image3, max_outputs=3)
  summary = sess.run(summary_op)
  writer.add_summary(summary)

  summary_op = tf.summary.image("pool2_32", image4, max_outputs=3)
  summary = sess.run(summary_op)
  writer.add_summary(summary)
  writer.close()
  sess.close()


def pred_refresh(log_dir):
  if tf.gfile.Exists(log_dir):
    tf.gfile.DeleteRecursively(log_dir)
  tf.gfile.MakeDirs(log_dir)
  with tf.Graph().as_default():
    pred_writer(log_dir)
    print("done!")

def fea_refresh(log_dir,image1,image2,image3,image4):
  if tf.gfile.Exists(log_dir):
    tf.gfile.DeleteRecursively(log_dir)
  tf.gfile.MakeDirs(log_dir)
  fea_writer(log_dir,image1,image2,image3,image4)
  print("done!")

'''
  summary_op = tf.summary.image("pool1_16", image2, max_outputs=3)
  summary = sess.run(summary_op)
  writer.add_summary(summary)

  summary_op = tf.summary.image("conv2_32", image3, max_outputs=3)
  summary = sess.run(summary_op)
  writer.add_summary(summary)

  summary_op = tf.summary.image("pool2_32", image4, max_outputs=3)
  summary = sess.run(summary_op)
  writer.add_summary(summary)
'''
'''
def main(_):
  if tf.gfile.Exists(FLAGS.log_dir):
    tf.gfile.DeleteRecursively(FLAGS.log_dir)
  tf.gfile.MakeDirs(FLAGS.log_dir)
  with tf.Graph().as_default():
    refresh_board()
    print("done!")

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--log_dir',
      type=str,
      default=os.path.join(os.getenv('TEST_TMPDIR', '/tmp'),
                           'tensorflow/mnist/logs/mnist_with_summaries'),
      help='Summaries log directory')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
'''
