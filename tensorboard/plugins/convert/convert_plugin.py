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
import caffe2.python.onnx.frontend as c2_onnx
from caffe2.python.onnx.backend import Caffe2Backend as c2

from werkzeug import wrappers

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
from tensorboard.plugins.graph_edit import onnx_write_util

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

  def get_plugin_apps(self):
    return {
        '/info': self.info_route,
        '/run_metadata': self.run_metadata_route,
        '/load': self.load_model,
        '/convert': self.convert_model
    }

  @wrappers.Request.application
  def load_model(self, request):
    """ It used to parse model file and convert it to tensorboard IR
    """
    model_type = request.args.get("source_type")
    if model_type == "torch":
      input_tensor_size = request.args.get("input_tensor_size")
      model_file = request.args.get('model_file')
      if not os.path.exists(model_file):
        # send a response to frontend and report file not existing
        pass
      pass
    elif model_type == "caffe2":
      predict_net = request.args.get("predict_net")
      init_net = request.args.get("init_net")
      if not (os.path.exists(predict_net) and os.path.exists(init_net)):
        # send a response to frontend and report that model file doesnot exist
        pass
      self._src_tb_graph = c2graph_util.C2Graph(predict_net, init_net, "pb")
    elif model_type == "caffe":
      model_file = request.args.get('source_path')
      self._src_tb_graph = caffe_util.CaffeGraph(model_file, "pb")
    elif model_type == "onnx":
      model_file = request.args.get('source_path')
      self._src_tb_graph = onnx_util.OnnxGraph(model_file, "onnx")
    elif model_type == "tf":
      model_file = request.args.get('source_path')
      pass
    else:
      # send a response to frontend and report model type error
      pass
    self._src_tb_graph.ConvertNet()
    graph = self._src_tb_graph.GetTBGraph()
    return http_util.Respond(request,str(graph), 'text/x-protobuf')

  @wrappers.Request.application
  def convert_model(self, request):
    destination_type = request.args.get('destination_type')

    if destination_type=='caffe2':
      predict_net = request.args.get('predict_net')
      init_net = request.args.get('init_net')
      logger.warn(init_net)
      logger.warn(predict_net)
    else:
      destination_path = request.args.get('destination_path')
      logger.warn(destination_path)

    if destination_type == 'onnx':
      data_type = onnx.TensorProto.FLOAT
      # data_shape = (1, 3, 299, 299) if model is inceptionv3/4
      data_shape = (1, 3, 224, 224)
      value_info = {
        'data': (data_type, data_shape)
      }

      if self._src_tb_graph._predict_net.name == '':
        self._src_tb_graph._predict_net.name = 'modelName'

      onnx_model = c2_onnx.caffe2_net_to_onnx_model(predict_net=self._src_tb_graph._predict_net,
                                                    init_net=self._src_tb_graph._init_net,
                                                    value_info=value_info)
      with open(destination_path, 'wb') as f:
        f.write(onnx_model.SerializeToString())

      self._dst_tb_graph = onnx_util.OnnxGraph(destination_path, "onnx")

    elif destination_type == 'caffe2':

      init_net_model, predict_net_model = c2.onnx_graph_to_caffe2_net(self._src_tb_graph._onnx_model)
      with open(predict_net, 'wb') as f_pre:
        f_pre.write(predict_net_model.SerializeToString())
      with open(init_net, 'wb') as f_init:
        f_init.write(init_net_model.SerializeToString())
      self._dst_tb_graph = c2graph_util.C2Graph(predict_net, init_net, "pb")

    logger.warn('Converting completed.')
    self._dst_tb_graph.ConvertNet()
    graph = self._dst_tb_graph.GetTBGraph()
    return http_util.Respond(request,str(graph) ,'text/x-protobuf')

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
