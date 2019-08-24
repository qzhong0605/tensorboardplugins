# Convert the caffe2 model into tensorboard GraphDef
#
# The details of caffe2 model is on the compat/proto/caffe2/caffe2.proto
# And the details of GraphDef model is on the compat/proto/graph.proto
#
################################################################################

from tensorboard.compat.proto import graph_pb2
from tensorboard.compat.proto import attr_value_pb2
from tensorboard.compat.proto import node_def_pb2
from tensorboard.compat.proto import tensor_shape_pb2
from tensorboard.compat.proto import tensor_pb2
from tensorboard.compat.proto import types_pb2
from tensorboard.compat.proto.caffe2 import caffe2_pb2
from tensorboard.util import tb_logging

from tensorboard.plugins.graph_edit import tbgraph_base

from google.protobuf import text_format

logger = tb_logging.get_logger()

class C2Graph(tbgraph_base.TBGraph):
    """ In order to visualize the caffe2 model graph, it converts the caffe2
    format model graph into the tensoboard-format model graph.

    The information about caffe2 model is on the proto
    `compat/proto/caffe2/caffe2.proto`.  And the tensorboard model is on the
    proto `compat/proto/graph.proto`

    In order to avoid the same tensor name and they are built from the different
    operators, we adopt the SSA form, which is used to differentiate different tensor
    """
    def __init__(self, predict_net, init_net, predict_net_type="pb"):
        super(C2Graph, self).__init__()
        self._predict_net = caffe2_pb2.NetDef()
        if predict_net_type == "pb":
            with open(predict_net, "rb") as predict_stream:
                self._predict_net.ParseFromString(predict_stream.read())
            logger.info("parse caffe2 predict net {} with protobuf format".format(predict_net))
        elif predict_net_type == "txt":
            with open(predict_net, "r") as predict_stream:
                text_format.Parse(predict_stream.read(), self._predict_net)
            logger.info("parse caffe2 predict net {} with text format".format(predict_net))
        else:
            raise NotImplementedError("The predict net type: {} doesn't support".format(predict_net_type))

        self._init_net = caffe2_pb2.NetDef()
        with open(init_net, "rb") as init_stream:
           self._init_net.ParseFromString(init_stream.read())
           logger.info("load caffe2 init net {} with protobuf format".format(init_net))

        # a map from node key to node, where the node key is globaly unique
        self.nodes = {}
        # a map from caffe2 operator to output, which is a SSA-format
        self.c2_op_out = {}
        # record the blob version for inplace-change
        self.blob_version = {}
        # a map from node name to shape info
        self.shapes = {}
        # a map from node name to dtype
        self.types = {}

    def _build_nodes_shapetype(self):
        """ Build an inner node shape information given the weights information for network """
        # add shape information
        if self._init_net is None:
            return
        for init_op in self._init_net.op:
            for init_arg in init_op.arg:
                if init_arg.name == "shape":
                    self.shapes[init_op.output[0]] = init_arg.ints
                elif init_arg.name == "values":
                    if len(init_arg.floats):
                        self.types[init_op.output[0]] = types_pb2.DT_FLOAT
                    elif len(init_arg.ints):
                        self.types[init_op.output[0]] = types_pb2.DT_INT64
                    elif len(init_arg.strings):
                        self.types[init_op.output[0]] = types_pb2.DT_STRING
                    else:
                        raise NotImplementedError("Not Supported Field: {}".format(init_arg))

    def _add_node_shapetype(self, node, shape_name):
        """ build an internal node shape map if given the weights information """
        if shape_name in self.shapes:
            tensor_shape = tensor_shape_pb2.TensorShapeProto()
            for shape_i in self.shapes[shape_name]:
                shape_dim = tensor_shape_pb2.TensorShapeProto.Dim()
                shape_dim.size = shape_i
                tensor_shape.dim.extend([shape_dim])
            attr_value = attr_value_pb2.AttrValue()
            attr_value.shape.CopyFrom(tensor_shape)
            node.attr['shape'].CopyFrom(attr_value)

        # add optional dtype
        if shape_name in self.types:
            attr_value = attr_value_pb2.AttrValue()
            attr_value.type = self.types[shape_name]
            node.attr['dtype'].CopyFrom(attr_value)

    def _MakeSSAName(self, name):
        """ It's used to make a unique name through a ssa-based format for `name`
        """
        if name not in self.blob_version:
            self.blob_version[name] = 0
        else:
            self.blob_version[name] += 1
        ret_name = "{}_{}".format(name, self.blob_version[name])
        return ret_name

    def convert_to_nodes(self, c2_op):
        """ Convert a caffe2 OperatorDef into TB nodes

        The nodes for TensorBoard have only inputs and don't have outputs. Therefore
        a caffe2 operator maybe converted into muliple nodes

        Arg:
            c2_op: a caffe2 OperatorDef
        """
        new_node = node_def_pb2.NodeDef()
        new_node.op = c2_op.type

        for c2_input in c2_op.input:
            if c2_input not in self.blob_version:
                # These inputs are weights or input data for current
                # tensorboard node. Therefore, the `op` is set to
                # `Initialization`
                in_node = node_def_pb2.NodeDef()
                self._add_node_shapetype(in_node, c2_input)
                self.blob_version[c2_input] = 0
                in_node.name = '{}_{}'.format(c2_input, self.blob_version[c2_input])
                in_node.op = "Initialization"
                self.nodes["{}_{}".format(c2_input, 0)] = in_node
                self._tb_graph.node.extend([in_node])
            new_node.input.append('{}_{}'.format(c2_input, self.blob_version[c2_input]))

        if len(c2_op.output) == 0:
            # There are no outputs for current C2 operator. Therefore, the node
            # name is set to C2 operation type
            new_node.name = self._MakeSSAName(c2_op.type)
        else:
            new_node.name = self._MakeSSAName(c2_op.output[0])
            # If more than one output, we build `Sibling` tensorboard node for
            # other outpouts
            for c2_output in c2_op.output[1:]:
                sibling_node = node_def_pb2.NodeDef()
                sibling_node.op = 'Sibling'
                sibling_node.name = self._MakeSSAName(c2_output)
                sibling_node.input.extend([new_node.name])
                self._add_node_shapetype(sibling_node, c2_output)
                self.nodes[sibling_node.name] = sibling_node
                self._tb_graph.node.extend([sibling_node])

        # add argument
        for c2_arg in c2_op.arg:
            attr = attr_value_pb2.AttrValue()
            if c2_arg.HasField('i'):
                attr.i = c2_arg.i
            elif c2_arg.HasField('f'):
                attr.f = c2_arg.f
            elif c2_arg.HasField('s'):
                attr.s = c2_arg.s
            elif len(c2_arg.floats):
                list_value = attr_value_pb2.AttrValue.ListValue()
                list_value.f.extend(c2_args.floats)
                attr.list = list_value
            elif len(c2_arg.ints):
                list_value = attr_value_pb2.AttrValue.ListValue()
                list_value.i.extend(c2_arg.ints)
                attr.list.CopyFrom(list_value)
            elif len(c2_arg.strings):
                list_value = attr_value_pb2.AttrValue.ListValue()
                list_value.s.extend(c2_arg.strings)
                attr.list.CopyFrom(list_value)
            new_node.attr[c2_arg.name].CopyFrom(attr)

        self._add_node_shapetype(new_node, c2_op.output[0])
        self.nodes[new_node.name] = new_node
        self._tb_graph.node.extend([new_node])

    def ConvertNet(self):
        """ Convert the full network of caffe2 into TB network """
        self._build_nodes_shapetype()
        for c2_op in self._predict_net.op:
            self.convert_to_nodes(c2_op)
