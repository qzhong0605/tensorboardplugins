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
"""The TensorBoard Graphs plugin."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import six
import os
from werkzeug import wrappers

from tensorboard.backend import http_util
from tensorboard.backend import process_graph
from tensorboard.backend.event_processing import plugin_event_accumulator as event_accumulator  # pylint: disable=line-too-long
from tensorboard.compat.proto import config_pb2
from tensorboard.compat.proto import graph_pb2
from tensorboard.compat.proto import node_def_pb2
from tensorboard.compat.proto import attr_value_pb2

from tensorboard.plugins import base_plugin
from tensorboard.plugins.graph import graph_util
from tensorboard.plugins.graph import keras_util
from tensorboard.plugins.graph import c2graph_util
from tensorboard.plugins.graph import caffe_util
from tensorboard.plugins.graph import onnx_util
from tensorboard.plugins.graph import onnx_write_util

from tensorboard.util import tb_logging

logger = tb_logging.get_logger()
# logger.setLevel(tb_logging.logging.INFO)
_PLUGIN_PREFIX_ROUTE = 'graphedit'

# The Summary API is implemented in TensorFlow because it uses TensorFlow internal APIs.
# As a result, this SummaryMetadata is a bit unconventional and uses non-public
# hardcoded name as the plugin name. Please refer to link below for the summary ops.
# https://github.com/tensorflow/tensorflow/blob/11f4ecb54708865ec757ca64e4805957b05d7570/tensorflow/python/ops/summary_ops_v2.py#L757
_PLUGIN_NAME_RUN_METADATA = 'graph_run_metadata'
# https://github.com/tensorflow/tensorflow/blob/11f4ecb54708865ec757ca64e4805957b05d7570/tensorflow/python/ops/summary_ops_v2.py#L788
_PLUGIN_NAME_RUN_METADATA_WITH_GRAPH = 'graph_run_metadata_graph'
# https://github.com/tensorflow/tensorflow/blob/565952cc2f17fdfd995e25171cf07be0f6f06180/tensorflow/python/ops/summary_ops_v2.py#L825
_PLUGIN_NAME_KERAS_MODEL = 'graph_keras_model'


class GraphEditPlugin(base_plugin.TBPlugin):
  """GraphEdit plugins for TensorBoard.

  For visualization, the Graph IR(Intermediate Representation) is on graph_pb2.
  For all other model graph, it must be converted into tensorboard IR.

  The workflow is to load an graph representation(including empty graph if not
  including graph definition) and edit on the dashboard of tensorboard

  Currently, the supported model type including caffe2/caffe, pytorch, onnx, tensorflow.
  """
  plugin_name = _PLUGIN_PREFIX_ROUTE

  def __init__(self, context):
    """Instantiates GraphsPlugin via TensorBoard core.

    Args:
      context: A base_plugin.TBContext instance.
    """
    logger.warn('__init__')
    self._multiplexer = context.multiplexer
    self._tb_graph = None

  def get_plugin_apps(self):
    return {
        '/init':self.init_graph,
        '/save': self.save_model,
        '/edit': self.edit_model,
        '/graph': self.graph_route,
        '/info': self.info_route,
        '/run_metadata': self.run_metadata_route,
        '/load': self.parse_from_model,
        '/op_intro':self.op_intro,
        '/op_default':self.op_default,
    }

  @wrappers.Request.application
  def parse_from_model(self, request):
      """ It used to parse model file and convert it to tensorboard IR
      """
      model_type = request.args.get("source_type")
      file_type = request.args.get("file_type")
      model_file = request.args.get('model_file')
      if model_type == "torch":
          pass
      elif model_type == "caffe2":
          predict_net = request.args.get("predict_net")
          init_net = request.args.get("init_net")
          if not (os.path.exists(predict_net) and os.path.exists(init_net)):
              # send a response to frontend and report that model file doesnot exist
              pass
          self._tb_graph = c2graph_util.C2Graph(predict_net, file_type, init_net)
      elif model_type == "caffe":
          self._tb_graph = caffe_util.CaffeGraph(model_file, file_type)
      elif model_type == "onnx":
          self._tb_graph = onnx_util.OnnxGraph(model_file, file_type)
      elif model_type == "tf":
          pass
      else:
          # send a response to frontend and report model type error
          pass

      if not os.path.exists(model_file):
          # send a response to frontend and report file not existing
          pass

      self._tb_graph.ConvertNet()
      graph = self._tb_graph._tb_graph
      self.graph = graph
      return http_util.Respond(request,str(graph) ,'text/x-protobuf')

  @wrappers.Request.application
  def init_graph(self, request):
    self.graph = graph_pb2.GraphDef()
    return http_util.Respond(request, 'init', 'text/plain')

  def is_active(self):
      """ GraphEdit plugin is always active """
      return True

  def frontend_metadata(self):
    return super(GraphEditPlugin, self).frontend_metadata()._replace(
        element_name='tf-graph-dashboard',
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
      self.graph = graph
    
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
  def graph_route(self, request):
    """Given a single run, return the graph definition in protobuf format."""
    run = request.args.get('run')
    tag = request.args.get('tag', '')
    conceptual_arg = request.args.get('conceptual', False)
    is_conceptual = True if conceptual_arg == 'true' else False

    if run is None:
      return http_util.Respond(
          request, 'query parameter "run" is required', 'text/plain', 400)

    limit_attr_size = request.args.get('limit_attr_size', None)
    if limit_attr_size is not None:
      try:
        limit_attr_size = int(limit_attr_size)
      except ValueError:
        return http_util.Respond(
            request, 'query parameter `limit_attr_size` must be an integer',
            'text/plain', 400)

    large_attrs_key = request.args.get('large_attrs_key', None)

    try:
      result = self.graph_impl(run, tag, is_conceptual, limit_attr_size, large_attrs_key)
    except ValueError as e:
      return http_util.Respond(request, e.message, 'text/plain', code=400)
    else:
      if result is not None:
        (body, mime_type) = result  # pylint: disable=unpacking-non-sequence
        return http_util.Respond(request, body, mime_type)
      else:
        return http_util.Respond(request, '404 Not Found', 'text/plain',
                                 code=404)

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
  def save_model(self, request):
    model_type = request.args.get('type')
    graph = self.graph
    logger.warn(len(graph.node))
    if model_type=='onnx':
      onnx_write_util.SaveInOnnxModel(graph)
    return http_util.Respond(request, 'save progress success', 'text/plain')

  @wrappers.Request.application
  def edit_model(self, request):
    eidt_type = request.args.get('type')
    # logger.warn(eidt_type)
    data = json.loads(request.args.get('data'))
    message = ''
    
    graph = self.graph
      
    if eidt_type == 'add_node':
      # logger.warn(data['attr'])
      newNode = node_def_pb2.NodeDef()
      newNode.name = data['name']
      newNode.op = data['op']
      # 根据返回的属性类型，设置相应值
      # for attr_item in data['attr']:
      #   logger.warn(attr_item) 
      #   newAttr = attr_value_pb2.AttrValue()
      #   newNode.attr.get_or_create()
      # logger.warn(newNode)
      graph.node.extend([newNode])
      message = 'ok'

    if eidt_type == 'edit':
      # logger.warn(data['attr'])
      for i in range(0, len(graph.node)):
        if graph.node[i].name == data['name']:
          tmp_node_attrs = graph.node[i].attr
          for attr_item in data['attr']:
            attr_to_edit = tmp_node_attrs.get_or_create(attr_item['key'])
            # logger.warn(attr_to_edit)
            # 根据返回的属性类型，设置相应值
          break
      message = 'ok'

    if eidt_type == 'delete_node':
      delete_pos = -1
      for i in range(0, len(graph.node)):
        tmp_node = graph.node[i]
        if data['op'] in tmp_node.input:
          tmp_node.input.remove(data['op'])
        if tmp_node.name == data['op']:
          delete_pos = i
      del graph.node[delete_pos] 
      message = 'ok'   

    if eidt_type == 'add_edge':
      for i in range(0, len(graph.node)):
        if graph.node[i].name == data['w']:
          graph.node[i].input.append(data['v'])
      message = 'ok'

    if eidt_type == 'delete_edge':
      for i in range(0, len(graph.node)):
        if graph.node[i].name == data['w']:
          graph.node[i].input.remove(data['v'])
      message = 'ok'

    return http_util.Respond(request, message, 'text/plain')

  @wrappers.Request.application
  def op_intro(self, request):
    op = request.args.get('op')
    logger.warn(op)
    return http_util.Respond(request, op, 'text/plain')

  @wrappers.Request.application
  def op_default(self, request):
    output = request.args.get('output')
    logger.warn(output)
    # default 属性默认值信息
    default = [{'name':'kernel','type':'a','default':'0'},{'name':'stride','type':'b','default':'0'}]
    data = json.dumps(default)
    return http_util.Respond(request, data, 'application/json')