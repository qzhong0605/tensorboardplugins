# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""The TensorBoard Convert plugin."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import six
import os
import onnx
import torch
import caffe2.python.onnx.frontend as c2_onnx
from caffe2.python.onnx.backend import Caffe2Backend as c2
from caffe2.proto import caffe2_pb2
from werkzeug import wrappers
from onnx import ModelProto
from tensorboard.backend import http_util
from tensorboard.backend import process_graph
from tensorboard.backend.event_processing import plugin_event_accumulator as event_accumulator  # pylint: disable=line-too-long
from tensorboard.compat.proto import config_pb2
from tensorboard.compat.proto import graph_pb2
from tensorboard.plugins import base_plugin
from tensorboard.plugins.convert import graph_util
from tensorboard.plugins.convert import keras_util
from tensorboard.util import tb_logging

from tensorboard.plugins.graph_edit import c2graph_util
from tensorboard.plugins.graph_edit import caffe_util
from tensorboard.plugins.graph_edit import onnx_util
from tensorboard.plugins.graph_edit import torch_util
from tensorboard.plugins.graph_edit import onnx_write_util
from torchvision.models.densenet import densenet121, densenet169, densenet201, densenet161
from torchvision.models.vgg import vgg16, vgg19
from torchvision.models.resnet import resnet101, resnet152, resnet18, resnet34, resnet50
from torchvision.models.googlenet import googlenet
logger = tb_logging.get_logger()

_PLUGIN_PREFIX_ROUTE = 'convert'

# The Summary API is implemented in TensorFlow because it uses TensorFlow internal APIs.
# As a result, this SummaryMetadata is a bit unconventional and uses non-public
# hardcoded name as the plugin name. Please refer to link below for the summary ops.
# https://github.com/tensorflow/tensorflow/blob/11f4ecb54708865ec757ca64e4805957b05d7570/tensorflow/python/ops/summary_ops_v2.py#L757
_PLUGIN_NAME_RUN_METADATA = 'graph_run_metadata'
# https://github.com/tensorflow/tensorflow/blob/11f4ecb54708865ec757ca64e4805957b05d7570/tensorflow/python/ops/summary_ops_v2.py#L788
_PLUGIN_NAME_RUN_METADATA_WITH_GRAPH = 'graph_run_metadata_graph'
# https://github.com/tensorflow/tensorflow/blob/565952cc2f17fdfd995e25171cf07be0f6f06180/tensorflow/python/ops/summary_ops_v2.py#L825
_PLUGIN_NAME_KERAS_MODEL = 'graph_keras_model'


class ConvertPlugin(base_plugin.TBPlugin):
  """Convert Plugin for TensorBoard.

  It's used to convert a source network definition into an another network definition

  For tensorboard, it visualize different framework network through tensorboard graph
  definition. However, in order to transform among different framework, the network
  must be converted into ONNX IR and then is transformed into the target network
  definition.

  Here, there two different network IR, including Tensorboard IR and ONNX IR.

  For model transformation among different frameworks, they are pretrained models
  and stored as protobuf-format file.
  """

  plugin_name = _PLUGIN_PREFIX_ROUTE

  def __init__(self, context):
    """Instantiates ConvertPlugin via TensorBoard core.

    Args:
      context: A base_plugin.TBContext instance.
    """
    self._multiplexer = context.multiplexer
    self._src_tb_graph = None
    self._dst_tb_graph = None

    # src type and dst type are used to record the input model type
    self._src_type = None
    self._dst_type = None
    # caffe2
    self.predict_net = None
    self.init_net = None
    # other models
    self.model_file = None
    # torch
    self.input_tensor_size = None

    self.s_node_count = None
    self.d_node_count = None

  def get_plugin_apps(self):
    return {
        '/info': self.info_route,
        '/run_metadata': self.run_metadata_route,
        '/load': self.load_model,
        '/convert': self.convert_model,
        '/statistics': self.get_statistics,
    }

  @wrappers.Request.application
  def load_model(self, request):
    """ It used to parse model file and convert it to tensorboard IR
    """
    self._src_type = request.args.get("source_type")
    if self._src_type == "torch":
      self.input_tensor_size = request.args.get("input_tensor_size")
      self.model_file = request.args.get('source_path')
      if not os.path.exists(self.model_file):
        # send a response to frontend and report file not existing
        pass
      self._src_tb_graph = torch_util.freeze_graph(googlenet(pretrained=True, transform_input=False),
                                                   torch.randn(1, 3, 224, 224))
    elif self._src_type == "caffe2":
      self.predict_net = request.args.get("predict_net")
      self.init_net = request.args.get("init_net")
      if not (os.path.exists(self.predict_net) and os.path.exists(self.init_net)):
        # send a response to frontend and report that model file doesnot exist
        pass
      self._src_tb_graph = c2graph_util.C2Graph(self.predict_net, self.init_net, "pb")
    elif self._src_type == "caffe":
      self.model_file = request.args.get('source_path')
      self._src_tb_graph = caffe_util.CaffeGraph(self.model_file, "pb")
    elif self._src_type == "onnx":
      self.model_file = request.args.get('source_path')
      self._src_tb_graph = onnx_util.OnnxGraph(self.model_file, "onnx")
    elif self._src_type == "tf":
      self.model_file = request.args.get('source_path')
      pass
    else:
      # send a response to frontend and report model type error
      pass

    # for torch graph, it's generated by tracing method
    graph = self._src_tb_graph
    if self._src_type != "torch":
        self._src_tb_graph.ConvertNet()
        graph = self._src_tb_graph.GetTBGraph()

    # count the number of nodes in the input model
    self.s_node_count = 0
    for node in graph.node:
      self.s_node_count += 1

    return http_util.Respond(request, str(graph), 'text/x-protobuf')

  @wrappers.Request.application
  def convert_model(self, request):
    self._dst_type = request.args.get('destination_type')
    if self._dst_type == 'caffe2':
      dst_predict_net = request.args.get('predict_net')
      dst_init_net = request.args.get('init_net')
      logger.warn(dst_init_net)
      logger.warn(dst_predict_net)
    else:
      destination_path = request.args.get('destination_path')
      logger.warn(destination_path)

    if self._dst_type == 'onnx':
      if self._src_type == 'caffe2':
        data_type = onnx.TensorProto.FLOAT
        # data_shape = (1, 3, 299, 299) if model is inceptionv3/4
        data_shape = (1, 3, 224, 224)
        value_info = {
            'data': (data_type, data_shape)
        }

        self.predict_net = caffe2_pb2.NetDef()
        with open('predict_net.pb', 'rb') as f:
          self.predict_net.ParseFromString(f.read())

        self.init_net = caffe2_pb2.NetDef()
        with open('init_net.pb', 'rb') as f:
          self.init_net.ParseFromString(f.read())

        # if self._src_tb_graph._predict_net.name == '':
        #     self._src_tb_graph._predict_net.name = 'modelName'

        onnx_model = c2_onnx.caffe2_net_to_onnx_model(self.predict_net,
                                                        self.init_net,
                                                        value_info)
        with open(destination_path, 'wb') as f:
          f.write(onnx_model.SerializeToString())

        self._dst_tb_graph = onnx_util.OnnxGraph(destination_path, "onnx")

      elif self._src_type == 'torch':
        destination_path = '/home/memo/PycharmProjects/ONNX/1.onnx'
        # TODO: choose input_net
        input_net = 'resnet18'

        logger.warn(destination_path)
        if input_net not in ['inceptionv3', 'inceptionv4', 'inceptionresnetv2']:
          x = torch.randn(1, 3, 224, 224)
        else:
          x = torch.randn(1, 3, 299, 299)
        model = resnet18(pretrained=True)
        torch.onnx.export(model, x, destination_path, verbose=True)

    elif self._dst_type == 'caffe2':
      if self._src_type == 'onnx':
        onnx_model_proto = ModelProto()
        with open(self.model_file, "rb") as onnx_model_path:
          onnx_model_proto.ParseFromString(onnx_model_path.read())

        init_net_model, predict_net_model = c2.onnx_graph_to_caffe2_net(onnx_model_proto)
        with open(dst_predict_net, 'wb') as f_pre:
          f_pre.write(predict_net_model.SerializeToString())
        with open(dst_init_net, 'wb') as f_init:
          f_init.write(init_net_model.SerializeToString())
        self._dst_tb_graph = c2graph_util.C2Graph(dst_predict_net, dst_init_net, "pb")

    logger.warn('Converting completed.')
    self._dst_tb_graph.ConvertNet()
    graph = self._dst_tb_graph.GetTBGraph()

    # count the number of nodes in the output model
    self.d_node_count = 0
    for node in graph.node:
      self.d_node_count += 1

    return http_util.Respond(request, str(graph), 'text/x-protobuf')
  def is_active(self):
    """The graphs plugin is active iff any run has a graph."""
    return True

  def frontend_metadata(self):
    return super(ConvertPlugin, self).frontend_metadata()._replace(
        element_name='tf-convert-dashboard',
        # TODO(@chihuahua): Reconcile this setting with Health Pills.
        disable_reload=True,
    )

  def info_impl(self):
    """Returns a dict of all runs and tags and their data availabilities."""
    result = {}
    def add_row_item(run, tag=None):
      run_item = result.setdefault(run, {
          'run': run,
          'tags': {},
          # A run-wide GraphDef of ops.
          'run_graph': False})

      tag_item = None
      if tag:
        tag_item = run_item.get('tags').setdefault(tag, {
            'tag': tag,
            'conceptual_graph': False,
            # A tagged GraphDef of ops.
            'op_graph': False,
            'profile': False})
      return (run_item, tag_item)

    mapping = self._multiplexer.PluginRunToTagToContent(
        _PLUGIN_NAME_RUN_METADATA_WITH_GRAPH)
    for run_name, tag_to_content in six.iteritems(mapping):
      for (tag, content) in six.iteritems(tag_to_content):
        # The Summary op is defined in TensorFlow and does not use a stringified proto
        # as a content of plugin data. It contains single string that denotes a version.
        # https://github.com/tensorflow/tensorflow/blob/11f4ecb54708865ec757ca64e4805957b05d7570/tensorflow/python/ops/summary_ops_v2.py#L789-L790
        if content != b'1':
          logger.warn('Ignoring unrecognizable version of RunMetadata.')
          continue
        (_, tag_item) = add_row_item(run_name, tag)
        tag_item['op_graph'] = True

    # Tensors associated with plugin name _PLUGIN_NAME_RUN_METADATA contain
    # both op graph and profile information.
    mapping = self._multiplexer.PluginRunToTagToContent(
        _PLUGIN_NAME_RUN_METADATA)
    for run_name, tag_to_content in six.iteritems(mapping):
      for (tag, content) in six.iteritems(tag_to_content):
        if content != b'1':
          logger.warn('Ignoring unrecognizable version of RunMetadata.')
          continue
        (_, tag_item) = add_row_item(run_name, tag)
        tag_item['profile'] = True
        tag_item['op_graph'] = True

    # Tensors associated with plugin name _PLUGIN_NAME_KERAS_MODEL contain
    # serialized Keras model in JSON format.
    mapping = self._multiplexer.PluginRunToTagToContent(
        _PLUGIN_NAME_KERAS_MODEL)
    for run_name, tag_to_content in six.iteritems(mapping):
      for (tag, content) in six.iteritems(tag_to_content):
        if content != b'1':
          logger.warn('Ignoring unrecognizable version of RunMetadata.')
          continue
        (_, tag_item) = add_row_item(run_name, tag)
        tag_item['conceptual_graph'] = True

    for (run_name, run_data) in six.iteritems(self._multiplexer.Runs()):
      if run_data.get(event_accumulator.GRAPH):
        (run_item, _) = add_row_item(run_name, None)
        run_item['run_graph'] = True

    for (run_name, run_data) in six.iteritems(self._multiplexer.Runs()):
      if event_accumulator.RUN_METADATA in run_data:
        for tag in run_data[event_accumulator.RUN_METADATA]:
          (_, tag_item) = add_row_item(run_name, tag)
          tag_item['profile'] = True

    return result

  def graph_impl(self, run, tag, is_conceptual, limit_attr_size=None, large_attrs_key=None):
    """Result of the form `(body, mime_type)`, or `None` if no graph exists."""
    if is_conceptual:
      tensor_events = self._multiplexer.Tensors(run, tag)
      # Take the first event if there are multiple events written from different
      # steps.
      keras_model_config = json.loads(tensor_events[0].tensor_proto.string_val[0])
      graph = keras_util.keras_model_to_graph_def(keras_model_config)
    elif tag:
      tensor_events = self._multiplexer.Tensors(run, tag)
      # Take the first event if there are multiple events written from different
      # steps.
      run_metadata = config_pb2.RunMetadata.FromString(
          tensor_events[0].tensor_proto.string_val[0])
      graph = graph_pb2.GraphDef()

      for func_graph in run_metadata.function_graphs:
        graph_util.combine_graph_defs(graph, func_graph.pre_optimization_graph)
    else:
      graph = self._multiplexer.Graph(run)

    # This next line might raise a ValueError if the limit parameters
    # are invalid (size is negative, size present but key absent, etc.).
    process_graph.prepare_graph_for_ui(graph, limit_attr_size, large_attrs_key)
    return (str(graph), 'text/x-protobuf')  # pbtxt

  def run_metadata_impl(self, run, tag):
    """Result of the form `(body, mime_type)`, or `None` if no data exists."""
    try:
      run_metadata = self._multiplexer.RunMetadata(run, tag)
    except ValueError:
      # TODO(stephanwlee): Should include whether FE is fetching for v1 or v2 RunMetadata
      # so we can remove this try/except.
      tensor_events = self._multiplexer.Tensors(run, tag)
      if tensor_events is None:
        return None
      # Take the first event if there are multiple events written from different
      # steps.
      run_metadata = config_pb2.RunMetadata.FromString(
          tensor_events[0].tensor_proto.string_val[0])
    if run_metadata is None:
      return None
    return (str(run_metadata), 'text/x-protobuf')  # pbtxt

  @wrappers.Request.application
  def info_route(self, request):
    info = self.info_impl()
    return http_util.Respond(request, info, 'application/json')

  @wrappers.Request.application
  def run_metadata_route(self, request):
    """Given a tag and a run, return the session.run() metadata."""
    tag = request.args.get('tag')
    run = request.args.get('run')
    if tag is None:
      return http_util.Respond(
          request, 'query parameter "tag" is required', 'text/plain', 400)
    if run is None:
      return http_util.Respond(
          request, 'query parameter "run" is required', 'text/plain', 400)
    result = self.run_metadata_impl(run, tag)
    if result is not None:
      (body, mime_type) = result  # pylint: disable=unpacking-non-sequence
      return http_util.Respond(request, body, mime_type)
    else:
      return http_util.Respond(request, '404 Not Found', 'text/plain',
                               code=404)
  @wrappers.Request.application
  def get_statistics(self, request):

    info = 'info {\n  key: "SOURCE"\n  value: ' + str(self.s_node_count) +\
           '\n}\ninfo {\n  key: "DESTINATION"\n  value: ' + str(self.d_node_count) + '\n}'
    return http_util.Respond(request, info, 'text/x-protobuf')
