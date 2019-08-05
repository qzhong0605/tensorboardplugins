# TensorBoard [![Travis build status](https://www.travis-ci.com/tensorflow/tensorboard.svg?branch=master)](https://travis-ci.com/tensorflow/tensorboard/) [![Compat check PyPI](https://python-compatibility-tools.appspot.com/one_badge_image?package=tensorboard)](https://python-compatibility-tools.appspot.com/one_badge_target?package=tensorboard)

TensorBoard is a suite of web applications for inspecting and understanding your
TensorFlow runs and graphs.

This README gives an overview of key concepts in TensorBoard, as well as how to
interpret the visualizations TensorBoard provides. For an in-depth example of
using TensorBoard, see the tutorial: [TensorBoard: Visualizing
Learning][].
For in-depth information on the Graph Visualizer, see this tutorial: 
[TensorBoard: Graph Visualization][].

[TensorBoard: Visualizing Learning]: https://www.tensorflow.org/get_started/summaries_and_tensorboard
[TensorBoard: Graph Visualization]: https://www.tensorflow.org/get_started/graph_viz

You may also want to watch
[this video tutorial][] that walks
through setting up and using TensorBoard. There's an associated 
[tutorial with an end-to-end example of training TensorFlow and using TensorBoard][].

[this video tutorial]: https://www.youtube.com/watch?v=eBbEDRsCmv4

[tutorial with an end-to-end example of training TensorFlow and using TensorBoard]: https://github.com/martinwicke/tf-dev-summit-tensorboard-tutorial

# Usage

Before running TensorBoard, make sure you have generated summary data in a log
directory by creating a summary writer:

# Requirement 

In order to run TensorBoard, you must install the following package:

```
torch >= 1.0.1
future 
tensorflow >=1.12
```

# Plugins 

- GraphEdit: A web-based graph editor for different deep learning framework, including caffe/caffe2, pytorch, onnx and tensorflow 
- Convert: A tool for transformation among different deep learning frameworks  
- DebugDB: A grapg debug tool for tracing the training process of different deep learning frameworks  
- Inference: A tool for checking a pretrained deep learning model, including features, data and model comparison
