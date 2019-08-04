# Convert node-based onnx network to node-based TensorBoard network
#
# The difference between onnx and TensorBoard is that the node on `onnx`
# does have output, while the TensorBoard doesn't have output
#
################################################################################
""" In order to visualize the onnx graph model on TensorBoard, we need convert the
onnx model to the TensorBoard model

The network definition for onnx is on the file `tensorboard/compat/proto/onnx/onnx.proto`
And the network for TensorBoard is on the file `tensorboard/compat/proto/graph.proto`

The node on the TensorBoard network is a SSA format
"""
from tensorboard.compat.proto import graph_pb2
from tensorboard.compat.proto import node_def_pb2
from tensorboard.compat.proto import attr_value_pb2
from tensorboard.compat.proto import tensor_shape_pb2
from tensorboard.compat.proto import tensor_pb2
from tensorboard.compat.proto import types_pb2
from tensorboard.compat.proto.onnx import onnx_pb2

from tensorboard.plugins.graph import tbgraph_base
from tensorboard.util import tb_logging

logger = tb_logging.get_logger()

def _onnx_dtype_to_tb(onnx_dtype):
    """ Convert onnx dtype into tensorboad dtype

    Arg:
        onnx_dtype: an number indicating the onnx data type

    Return:
        tensorboard data type
    """
    if onnx_dtype == onnx_pb2.TensorProto.FLOAT:
        return types_pb2.DT_FLOAT
    elif onnx_dtype == onnx_pb2.TensorProto.UINT8:
        return types_pb2.DT_UINT8
    elif onnx_dtype == onnx_pb2.TensorProto.INT8:
        return types_pb2.DT_INT8
    elif onnx_dtype == onnx_pb2.TensorProto.UINT16:
        return types_pb2.DT_UINT16
    elif onnx_dtype == onnx_pb2.TensorProto.INT16:
        return types_pb2.DT_INT16
    elif onnx_dtype == onnx_pb2.TensorProto.INT32:
        return types_pb2.DT_INT32
    elif onnx_dtype == onnx_pb2.TensorProto.INT64:
        return types_pb2.DT_INT64
    elif onnx_dtype == onnx_pb2.TensorProto.STRING:
        return types_pb2.DT_STRING
    elif onnx_dtype == onnx_pb2.TensorProto.BOOL:
        return types_pb2.DT_BOOL
    elif onnx_dtype == onnx_pb2.TensorProto.FLOAT16:
        # TBD
        return types_pb2.DT_BFLOAT16
    elif onnx_dtype == onnx_pb2.TensorProto.DOUBLE:
        return types_pb2.DT_DOUBLE
    elif onnx_dtype == onnx_pb2.TensorProto.UINT32:
        return types_pb2.DT_UINT32
    elif onnx_dtype == onnx_pb2.TensorProto.UINT64:
        return types_pb2.DT_UINT64
    elif onnx_dtype == onnx_pb2.TensorProto.COMPLEX64:
        return types_pb2.DT_COMPLEX64
    elif onnx_dtype == onnx_pb2.TensorProto.COMPLEX128:
        return types_pb2.DT_COMPLEX128
    elif onnx_dtype == onnx_pb2.TensorProto.BFLOAT16:
        return types_pb2.DT_BFLOAT16
    else:
        return types_pb2.DT_INVALID


def _onnx_dims_to_tb_shape(onnx_tensor, tb_tensor):
    """ Build tensor shape from onnx dims

    Args:
        onnx_tensor: an onnx tensor message proto
        tb_tensor: a tb tensor message proto
    """
    tb_tensor_shape = tensor_shape_pb2.TensorShapeProto()
    for dim in onnx_tensor.dims:
        tb_dim = tensor_shape_pb2.TensorShapeProto.Dim()
        tb_dim.size = dim
        tb_tensor_shape.dim.extend([tb_dim])
    tb_tensor.tensor_shape.CopyFrom(tb_tensor_shape)


def _onnx_tensor_to_tb(onnx_tensor, tb_tensor):
    """ Convert an onnx Tensor into Tensorboard Tensor

    Here we assume that the data for onnx tensor is kept on the `raw_data` with
    `bytes` format.
    And the tensor content for tb_tensor is kept on the `tensor_content` field

    Args:
        onnx_tensor: an onnx tensor message proto
        tb_tensor: a tensorboard tensor message proto
    """
    tb_tensor.dtype = _onnx_dtype_to_tb(onnx_tensor.data_type)
    _onnx_dims_to_tb_shape(onnx_tensor, tb_tensor)
    tb_tensor.tensor_content = onnx_tensor.raw_data


def _make_onnxattr_to_tbattr(onnx_attr, tb_attr):
    """ visit all the onnx node attributes and convert them to the attributes of
    tensorboard nodes

    Args:
        onnx_attr: an AttributeProto message proto for onnx
        tb_attr: an AttrValue message proto for tensorboard
    """
    if onnx_attr.type == onnx_pb2.AttributeProto.FLOAT:
        tb_attr.f = onnx_attr.f
    elif onnx_attr.type == onnx_pb2.AttributeProto.INT:
        tb_attr.i = onnx_attr.i
    elif onnx_attr.type == onnx_pb2.AttributeProto.STRING:
        tb_attr.s = onnx_attr.s
    elif onnx_attr.type == onnx_pb2.AttributeProto.TENSOR:
        tb_tensor = tensor_pb2.TensorProto()
        _onnx_tensor_to_tb(onnx_attr.t, tb_tensor)
        tb_attr.tensor.CopyFrom(tb_tensor)
    elif onnx_attr.type == onnx_pb2.AttributeProto.GRAPH:
        raise NotImplementedError("onnx graph attribute doesn't support yet")
    elif onnx_attr.type == onnx_pb2.AttributeProto.FLOATS:
        floats_list = attr_value_pb2.AttrValue.ListValue()
        floats_list.f.extend(onnx_attr.floats)
        tb_attr.list.CopyFrom(floats_list)
    elif onnx_attr.type == onnx_pb2.AttributeProto.INTS:
        ints_list = attr_value_pb2.AttrValue.ListValue()
        ints_list.i.extend(onnx_attr.ints)
        tb_attr.list.CopyFrom(ints_list)
    elif onnx_attr.type == onnx_pb2.AttributeProto.STRINGS:
        strings_list = attr_value_pb2.AttrValue.ListValue()
        strings_list.s.extend(onnx_attr.strings)
        tb_attr.list.CopyFrom(strings_list)
    elif onnx_attr.type == onnx_pb2.AttributeProto.TENSORS:
        raise NotImplementedError("onnx tensors attribute doesn't support yet")
    elif onnx_attr.type == onnx_pb2.AttributeProto.GRAPHS:
        raise NotImplementedError("onnx graphs attribute doesn't support yet")
    else:
        raise NotImplementedError("onnx undefined type attribute doesn't support yet")


class OnnxGraph(tbgraph_base.TBGraph):
    def __init__(self, onnx_model, onnx_type="pb"):
        """ onnx_type is the file type for onnx model, currently including pb """
        super(OnnxGraph,self).__init__()
        self._onnx_model = onnx_pb2.ModelProto()
        if onnx_type == "pb":
            with open(onnx_model, "rb") as onnx_stream:
                self._onnx_model.ParseFromString(onnx_stream.read())
        else:
            raise NotImplementedError("Onnx file type {} doesn't support yet".format(onnx_type))
        # a map from the node to version, while the node name is unique
        self._nodes_version = {}
        # a map from the name to nodes
        self.tb_nodes = {}

    def convert_to_nodes(self, onnx_node):
        new_node = node_def_pb2.NodeDef()
        new_node.op = onnx_node.op_type

        for onnx_input in onnx_node.input:
            if onnx_input not in self._nodes_version:
                in_node = node_def_pb2.NodeDef()
                self._nodes_version[onnx_input] = 0
                in_node.name = '{}_{}'.format(onnx_input, 0)
                self.tb_nodes[in_node.name] = in_node
                self._tb_graph.node.extend([in_node])
            new_node.input.append('{}_{}'.format(onnx_input, self._nodes_version[onnx_input]))

        if onnx_node.output[0] in self._nodes_version:
            self._nodes_version[onnx_node.output[0]] += 1
        else:
            self._nodes_version[onnx_node.output[0]] = 0

        # handle onnx node attribute
        for onnx_attr in onnx_node.attribute:
            attr_value = attr_value_pb2.AttrValue()
            _make_onnxattr_to_tbattr(onnx_attr, attr_value)
            new_node.attr[onnx_attr.name].CopyFrom(attr_value)

        new_node.name = '{}_{}'.format(onnx_node.output[0], self._nodes_version[onnx_node.output[0]])
        self.tb_nodes[new_node.name] = new_node
        self._tb_graph.node.extend([new_node])

        for onnx_output in onnx_node.output[1:]:
            if onnx_output not in self._nodes_version:
                self._nodes_version[onnx_output] = 0
            else:
                self._nodes_version[onnx_output] += 1
            out_node = node_def_pb2.NodeDef()
            out_node.name = '{}_{}'.format(onnx_output, self._nodes_version[onnx_output])
            self.tb_nodes[out_node.name] = out_node
            self._tb_graph.node.extend([out_node])


    def ConvertNet(self):
        onnx_graph = self._onnx_model.graph
        for onnx_node in onnx_graph.node:
            self.convert_to_nodes(onnx_node)
