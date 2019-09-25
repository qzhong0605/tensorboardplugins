import tensorflow as tf

#filename = './data/mnist/test.tfrecords'

def read_and_decode(filename_queue, is_batch, batch_size, input_shape, output_shape):
 
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                   features={
                             'label': tf.FixedLenFeature([], tf.string),
                             'sample' : tf.FixedLenFeature([], tf.string),})  
    img= tf.decode_raw(features['sample'],tf.float32)
    img= tf.reshape(img, eval(input_shape))
    label = tf.decode_raw(features['label'],tf.float64)
    label = tf.reshape(label, eval(output_shape))
    print("################")
    print(eval(input_shape),eval(output_shape))
    if is_batch:
        min_after_dequeue = 10
        capacity = min_after_dequeue+3*batch_size
        img, label = tf.train.shuffle_batch([img, label],
                        batch_size=batch_size, 
                        num_threads=3, 
                        capacity=capacity,
                        min_after_dequeue=min_after_dequeue)
    return img, label


