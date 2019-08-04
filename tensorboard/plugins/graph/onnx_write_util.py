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

def _tb_dtype_to_onnx(tb_dtype):
    """ Convert tensorboad dtype into onnx dtype

    Arg:
        tb_dtype: an number indicating the tensorboad data type

    Return:
        onnx data type
    """
    if tb_dtype == types_pb2.DT_FLOAT:
        return onnx_pb2.TensorProto.FLOAT
    elif tb_dtype == types_pb2.DT_UINT8:
        return onnx_pb2.TensorProto.UINT8
    elif tb_dtype == types_pb2.DT_INT8:
        return onnx_pb2.TensorProto.INT8
    elif tb_dtype == types_pb2.DT_UINT16:
        return onnx_pb2.TensorProto.UINT16
    elif tb_dtype == types_pb2.DT_INT16:
        return onnx_pb2.TensorProto.INT16
    elif tb_dtype == types_pb2.DT_INT32:
        return onnx_pb2.TensorProto.INT32
    elif tb_dtype == types_pb2.DT_INT64:
        return onnx_pb2.TensorProto.INT64
    elif tb_dtype == types_pb2.DT_STRING:
        return onnx_pb2.TensorProto.STRING
    elif tb_dtype == types_pb2.DT_BOOL:
        return onnx_pb2.TensorProto.BOOL
    elif tb_dtype == types_pb2.DT_BFLOAT16:
        # TBD
        return onnx_pb2.TensorProto.FLOAT16
    elif tb_dtype == types_pb2.DT_DOUBLE:
        return onnx_pb2.TensorProto.DOUBLE
    elif tb_dtype == types_pb2.DT_UINT32:
        return onnx_pb2.TensorProto.UINT32
    elif tb_dtype == types_pb2.DT_UINT64:
        return onnx_pb2.TensorProto.UINT64
    elif tb_dtype == types_pb2.DT_COMPLEX64:
        return onnx_pb2.TensorProto.COMPLEX64
    elif tb_dtype == types_pb2.DT_COMPLEX128:
        return onnx_pb2.TensorProto.COMPLEX128
    elif tb_dtype == types_pb2.DT_BFLOAT16:
        return onnx_pb2.TensorProto.BFLOAT16
    else:
        return onnx_pb2.DT_INVALID

def _tb_shape_to_onnx_dims(tb_shape, onnx_tensor):
    """ Build onnx dims from tensor shape 
    Args:
        onnx_tensor: an onnx tensor message proto
        tb_tensor: a tb tensor message proto
    """
    for dim in tb_shape.dim:
        onnx_tensor.dims.extend([dim.size])

def _tb_tensor_to_onnx(tb_tensor, onnx_tensor):
    """ Convert an Tensorboard Tensor into onnx Tensor

    Here we assume that the data for Tensorboard tensor is kept on the `raw_data` with
    `bytes` format.
    And the tensor content for onnx_tensor is kept on the `raw_data` field

    Args:
        onnx_tensor: an onnx tensor message proto
        tb_tensor: a tensorboard tensor message proto
    """
    onnx_tensor.data_type = _tb_dtype_to_onnx(tb_tensor.dtype)
    _tb_shape_to_onnx_dims(tb_tensor.tensor_shape, onnx_tensor)
    onnx_tensor.raw_data = tb_tensor.tensor_content

def _make_tbattr_to_onnxattr(tb_attr, tb_attr_name, onnx_attr):
    """ visit all the tensorboard node attributes and convert them to the attributes of
    onnx nodes

    Args:
        onnx_attr: an AttributeProto message proto for onnx
        tb_attr: an AttrValue message proto for tensorboard
    """
    onnx_attr.name = tb_attr_name
    if tb_attr.HasField('s'):
        onnx_attr.s = tb_attr.s
        onnx_attr.type = onnx_pb2.AttributeProto.STRING
    elif tb_attr.HasField('i'):
        onnx_attr.i = tb_attr.i
        onnx_attr.type = onnx_pb2.AttributeProto.INT
    elif tb_attr.HasField('f'):
        onnx_attr.f = tb_attr.f
        onnx_attr.type = onnx_pb2.AttributeProto.FLOAT
    elif tb_attr.HasField('b'):
        if tb_attr.b:
            onnx_attr.i = 1
        else:
            onnx_attr.i = 0
    elif tb_attr.HasField('type'):
        onnx_attr.i = tb_attr.type
        onnx_attr.type = onnx_pb2.AttributeProto.INT
        # raise NotImplementedError("onnx undefined type attribute doesn't support yet")
    elif tb_attr.HasField('shape'):
        onnx_tensor = onnx_pb2.TensorProto()
        _tb_shape_to_onnx_dims(tb_attr.shape, onnx_tensor)
        onnx_attr.t.CopyFrom(onnx_tensor)
        onnx_attr.type = onnx_pb2.AttributeProto.TENSOR
    elif tb_attr.HasField('tensor'):
        onnx_tensor = onnx_pb2.TensorProto()
        _tb_tensor_to_onnx(tb_attr.tensor, onnx_tensor)
        onnx_attr.t.CopyFrom(onnx_tensor)
        onnx_attr.type = onnx_pb2.AttributeProto.TENSOR
    elif tb_attr.HasField('list'):
        if len(tb_attr.list.s):
            for s_item in tb_attr.list.s:
                onnx_attr.strings.extend([s_item])
            onnx_attr.type = onnx_pb2.AttributeProto.STRINGS
        elif len(tb_attr.list.i):
            for i_item in tb_attr.list.i:
                onnx_attr.ints.extend([i_item])
            onnx_attr.type = onnx_pb2.AttributeProto.INTS
        elif len(tb_attr.list.f):
            for f_item in tb_attr.list.f:
                onnx_attr.floats.extend([f_item])
            onnx_attr.type = onnx_pb2.AttributeProto.FLOATS
        elif len(tb_attr.list.b):
            raise NotImplementedError("onnx undefined type attribute doesn't support yet")
        elif len(tb_attr.list.type):
            raise NotImplementedError("onnx undefined type attribute doesn't support yet")
        elif len(tb_attr.list.shape):
            for shape_item in tb_attr.list.shape:
                onnx_tensor = onnx_pb2.TensorProto()
                _tb_shape_to_onnx_dims(shape_item, onnx_tensor)
                onnx_attr.tensors.extend([onnx_tensor])
            onnx_attr.type = onnx_pb2.AttributeProto.TENSORS
        elif len(tb_attr.list.tensor):
            for t_item in tb_attr.list.shape:
                onnx_tensor = onnx_pb2.TensorProto()
                _tb_tensor_to_onnx(t_item, onnx_tensor)
                onnx_attr.tensors.extend([onnx_tensor])
            onnx_attr.type = onnx_pb2.AttributeProto.TENSORS
        elif len(tb_attr.list.func):
            raise NotImplementedError("onnx undefined type attribute doesn't support yet")
    elif tb_attr.HasField('func'):
        raise NotImplementedError("onnx undefined type attribute doesn't support yet")
    elif tb_attr.HasField('placeholder'):
        raise NotImplementedError("onnx undefined type attribute doesn't support yet")
    else:
        raise NotImplementedError("onnx undefined type attribute doesn't support yet")
    
def convert_to_nodes(tb_node, onnx_nodes, outputs):
    new_node = onnx_pb2.NodeProto()
    new_node.op_type = tb_node.op

    for tb_input in tb_node.input:
        tb_input = tb_input.replace('^','')
        tb_input = tb_input.replace(':1','')
        new_node.input.extend([tb_input])
        if tb_input in outputs:
            outputs[tb_input].extend([tb_node.name])
        else:
            outputs[tb_input] = [tb_node.name]
    for tb_attr in tb_node.attr:
        attr_value = onnx_pb2.AttributeProto()
        _make_tbattr_to_onnxattr(tb_node.attr[tb_attr], tb_attr, attr_value)
        new_node.attribute.extend([attr_value])
        
    new_node.name = tb_node.name
    onnx_nodes[new_node.name] = new_node

def ConvertNet(tb_graph):
    # a map from the name to nodes
    onnx_model = onnx_pb2.ModelProto()
    onnx_nodes = {}
    outputs = {}
    for tb_node in tb_graph.node:
        convert_to_nodes(tb_node, onnx_nodes, outputs)
    for node_name in outputs:
        onnx_nodes[node_name].output.extend(outputs[node_name])
    for node in onnx_nodes:
        onnx_model.graph.node.extend([onnx_nodes[node]])
    return onnx_model

def WriteToOnnx(onnx_model):
    onnx_write_model = open("/tmp/onnx_graph.onnx", "wb")
    onnx_to_write = onnx_model
    onnx_write_model.write(onnx_to_write.SerializeToString())
    onnx_write_model.close()
    # onnx_read = onnx_pb2.ModelProto()
    # onnx_read.ParseFromString(open("/tmp/onnx_graph.onnx", "rb").read())
    # logger.warn(onnx_read.graph.node[0])

def SaveInOnnxModel(tb_graph):
    WriteToOnnx(ConvertNet(tb_graph))