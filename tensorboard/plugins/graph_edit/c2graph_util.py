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

from tensorboard.plugins.graph_edit import tbgraph_base

from google.protobuf import text_format

class C2Graph(tbgraph_base.TBGraph):
    """ In order to visualize the caffe2 model graph, it converts the caffe2
    format model graph into the tensoboard-format model graph.

    The information about caffe2 model is on the proto
    `compat/proto/caffe2/caffe2.proto`.  And the tensorboard model is on the
    proto `compat/proto/graph.proto`

    In order to avoid the same tensor name and they are built from the different
    operators, we adopt the SSA form, which is used to differentiate different tensor
    """
    def __init__(self, predict_net, predict_net_type="pb", init_net=None):
        self.predict_net = caffe2_pb2.NetDef()
        if predict_net_type == "pb":
            with open(predict_net, "rb") as predict_stream:
                self.predict_net.ParseFromString(predict_stream.read())
        elif predict_net_type == "txt":
            with open(predict_net, "r") as predict_stream:
                text_format.Parse(predict_stream.read(), self.predict_net)
        else:
            raise NotImplementedError("The predict net type: {} doesn't support".format(predict_net_type))

        self.init_net = None
        if init_net is not None:
            self.init_net = caffe2_pb2.NetDef()
            with open(init_net, "rb") as init_stream:
                self.init_net.ParseFromString(init_stream.read())
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
        if self.init_net is None:
            return
        for init_op in self.init_net.op:
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
                in_node = node_def_pb2.NodeDef()
                self._add_node_shapetype(in_node, c2_input)
                self.blob_version[c2_input] = 0
                in_node.name = '{}_{}'.format(c2_input, self.blob_version[c2_input])
                self.nodes["{}_{}".format(c2_input, 0)] = in_node
                self._tb_graph.node.extend([in_node])
            new_node.input.append('{}_{}'.format(c2_input, self.blob_version[c2_input]))

        if c2_op.output[0] in self.blob_version:
            # have a previous blob and increase current version
            self.blob_version[c2_op.output[0]] += 1
        else:
            # this is a first version
            self.blob_version[c2_op.output[0]] = 0
        new_node.name = '{}_{}'.format(c2_op.output[0], self.blob_version[c2_op.output[0]])

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

        for c2_output in c2_op.output[1:]:
            if c2_output in self.blob_version:
                # have a previous blob and update the version
                self.blob_version[c2_output] += 1
            else:
                # this is the first blob
                self.blob_version[c2_output] = 0
            out_node = node_def_pb2.NodeDef()
            out_node.input.append(new_node.name)
            out_node.name = '{}_{}'.format(c2_output, self.blob_version[c2_output])
            self._add_node_shapetype(out_node, c2_output)
            self.nodes["{}_{}".format(c2_output, self.blob_version[c2_output])] = out_node
            self._tb_graph.node.extend([out_node])


    def ConvertNet(self):
        """ Convert the full network of caffe2 into TB network """
        self._build_nodes_shapetype()
        for c2_op in self.predict_net.op:
            self.convert_to_nodes(c2_op)
