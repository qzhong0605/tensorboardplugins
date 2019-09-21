# convert module-based torch model to node-based tensorboard model
#
# The model for pytorch is dynamic, whose architecture is not known when
# are built. The graph model for pytorch is constructed when the model is
# forward.
#
################################################################################
""" The whole thing used to visualize the pytorch module-based graph is to hook and
wrap the pytorch.

For pytorch, it build deep learning graph through module, tensor and functional.
The real thing about the module is done on the functional with `forward` for
the module.

In order to mantain the hierachical namespace, we save the origial namespace. It
means that if the type of graph node for pytroch is basic module, such as convolution,
relu, we do it on the module-based wrapper. For the tensor-based graph node of pytorch,
we do it on the tensor-based wrapper.

There are some graph nodes being handled on the functional module. They are conflicted with
the module-based node. In order to handle this problem, we first check whether tensorboard node
has been created on the pytorch module wrapper or not. If is done, we don't do any more
thing on the functional wrapper

In order to avoid inplace operation, we build an SSA-based node graph. And we build three important
graph node maps, including __name_to_version__, __tensor_to_name__ and __module_to_name__.
"""

from tensorboard.compat.proto import graph_pb2
from tensorboard.compat.proto import attr_value_pb2
from tensorboard.compat.proto import types_pb2
from tensorboard.compat.proto import node_def_pb2

import torch
from torch.nn import modules
from torch.nn.modules import Module as torch_module
from torch.nn import functional as F


# It's used to indicate which module type need to be handled by module class
# Other type are handled by replacing the function operator with the hooked
# operators
__class_modules__ = [
    'Identity', 'Linear', 'Conv1d', 'Conv2d', 'Conv3d', 'ConvTranspose1d', 'ConvTranspose2d',
    'ConvTranspose3d', 'Threshold', 'ReLU', 'Hardtanh', 'ReLU6', 'Sigmoid', 'Tanh',
    'Softmax', 'Softmax2d', 'LogSoftmax', 'ELU', 'SELU', 'CELU', 'GLU', 'Hardshrink',
    'LeakyReLU', 'LogSigmoid', 'Softplus', 'Softshrink', 'MultiheadAttention', 'PReLU',
    'Softsign', 'Softmin', 'Tanhshrink', 'RReLU', 'L1Loss', 'NLLLoss', 'KLDivLoss', 'MSELoss',
    'BCELoss', 'BCEWithLogitsLoss', 'NLLLoss2d', 'PoissonNLLLoss', 'CosineEmbeddingLoss', 'CTCLoss',
    'HingeEmbeddingLoss', 'MarginRankingLoss', 'MultiLabelMarginLoss', 'MultiLabelSoftMarginLoss',
    'MultiMarginLoss', 'SmoothL1Loss', 'SoftMarginLoss', 'CrossEntropyLoss',
    'AvgPool1d', 'AvgPool2d', 'AvgPool3d', 'MaxPool1d', 'MaxPool2d', 'MaxPool3d',
    'MaxUnpool1d', 'MaxUnpool2d', 'MaxUnpool3d', 'FractionalMaxPool2d', 'FractionalMaxPool3d',
    'LPPool1d', 'LPPool2d', 'LocalResponseNorm', 'BatchNorm1d', 'BatchNorm2d',
    'BatchNorm3d', 'InstanceNorm1d', 'InstanceNorm2d', 'InstanceNorm3d', 'LayerNorm',
    'GroupNorm', 'SyncBatchNorm', 'Dropout', 'Dropout2d', 'Dropout3d', 'AlphaDropout',
    'FeatureAlphaDropout', 'ReflectionPad1d', 'ReflectionPad2d', 'ReplicationPad2d', 'ReplicationPad1d',
    'ReplicationPad3d', 'CrossMapLRN2d', 'Embedding', 'EmbeddingBag', 'RNNBase',
    'RNN', 'LSTM', 'GRU', 'RNNCellBase', 'RNNCell', 'LSTMCell', 'GRUCell', 'PixelShuffle',
    'Upsample', 'UpsamplingNearest2d', 'UpsamplingBilinear2d', 'PairwiseDistance',
    'AdaptiveMaxPool1d', 'AdaptiveMaxPool2d', 'AdaptiveMaxPool3d', 'AdaptiveAvgPool1d',
    'AdaptiveAvgPool3d', 'TripletMarginLoss', 'ZeroPad2d', 'ConstantPad1d',
    'ConstantPad2d', 'ConstantPad3d', 'Bilinear', 'CosineSimilarity',
    'Unfold', 'Fold', 'AdaptiveLogSoftmaxWithLoss'
]


# tensorboard graph def for pytorch module-based graph
class TorchTBGraph(object):
    torch_tb_graph = graph_pb2.GraphDef()

# map from the tensor reference to tensor name
class TensorToName(object):
    __tensor_to_name__ = {}

    @classmethod
    def get_tensor_name(cls, tensor):
        """ According to the specified tensor, retrieve the target name for the tensor

        Arg:
            tensor: an instance of Tensor
        Return:
            a hierarchical name for the tensor
        """
        assert isinstance(tensor, torch.Tensor)
        return cls.__tensor_to_name__[id(tensor)]

    @classmethod
    def add_tensor_name(cls, tensor, name):
        """ Add a new pair containing tensor and name to the tensor_to_name map table

        Args:
            tensor: an instance of Tensor
            name: an string representing it
        """
        assert isinstance(tensor, torch.Tensor)
        cls.__tensor_to_name__[id(tensor)] = name


# map from name to version
__name_to_version__ = {}

# map from the module reference to module name
class ModuleToName(object):
    __module_to_name__ = {}

    @classmethod
    def get_module_name(cls, module):
        """ According to the specified module, retrieve the target name for the module

        Arg:
            module: an instance of torch module wrapper
        Return:
            a hierarchical name for the module wrapper
        """
        assert isinstance(module, torch.nn.Module)
        return cls.__module_to_name__[id(module)]

    @classmethod
    def add_module_name(cls, module, name):
        """ Add a new pair containing module and name to the module_to_name map table

        Args:
            module: an instance of torch module
            name: a string name for the module
        """
        assert isinstance(module, torch.nn.Module)
        cls.__module_to_name__[id(module)] = name


def _make_int_attr_value(node, name, int_attr):
    attr_value = attr_value_pb2.AttrValue()
    attr_value.i = int_attr
    node.attr[name].CopyFrom(attr_value)


def _make_bool_attr_value(node, name, bool_attr):
    attr_value = attr_value_pb2.AttrValue()
    attr_value.b = bool_attr
    node.attr[name].CopyFrom(attr_value)


def _make_float_attr_value(node, name, float_attr):
    attr_value = attr_value_pb2.AttrValue()
    attr_value.f = float_attr
    node.attr[name].CopyFrom(attr_value)


def _make_bytes_attr_value(node, name, bytes_attr):
    attr_value = attr_value_pb2.AttrValue()
    attr_value.s = bytes(bytes_attr, "utf-8")
    node.attr[name].CopyFrom(attr_value)


def _make_int_list_attr_value(node, name, int_list_attr):
    attr_value = attr_value_pb2.AttrValue()
    int_list_value = attr_value_pb2.AttrValue.ListValue()
    int_list_value.i.extend(int_list_attr)
    attr_value.list.CopyFrom(int_list_value)
    node.attr[name].CopyFrom(attr_value)


def _update_name_version(module):
    """ update the name version map and return the node name for the module """
    if ModuleToName.get_module_name(module) in __name_to_version__:
        __name_to_version__[ModuleToName.get_module_name(module)] += 1
    else:
        __name_to_version__[ModuleToName.get_module_name(module)] = 0
    node_name = "{}#{}".format(
        ModuleToName.get_module_name(module),
        __name_to_version__[ModuleToName.get_module_name(module)]
    )
    return node_name


def _make_convnd_node(
    in_channels, out_channels, kernel_size, stride, padding,
    dilation, groups, bias, padding_mode,
    input_tensor, node_name,
    conv_type = "Conv2d"
):
    conv_node = node_def_pb2.NodeDef()
    conv_node.input.extend([TensorToName.get_tensor_name(input_tensor)])
    conv_node.name = node_name
    conv_node.op = conv_type
    _make_int_attr_value(conv_node, "in_channels", in_channels)
    _make_int_attr_value(conv_node, "out_channels", out_channels)
    _make_int_list_attr_value(conv_node, "kernel_size", kernel_size)
    _make_int_list_attr_value(conv_node, "stride", stride)
    _make_int_list_attr_value(conv_node, "padding", padding)
    _make_int_list_attr_value(conv_node, "dilation", dilation)
    _make_int_attr_value(conv_node, "groups", groups)
    _make_bool_attr_value(conv_node, "bias", bias)
    _make_bytes_attr_value(conv_node, "padding_mode", padding_mode)
    return conv_node


class Conv1dCustom(modules.Conv1d):
    def __init__(self, conv1d_module, tb_graph):
        """ Conv1dCustom is a wrapper for Conv1d Module

        Args:
            conv1d_module: It's an instance of torch Conv1 module
            tb_graph: it's an tb GraphDef message proto
        """
        super(Conv1dCustom, self).__init__(
            conv1d_module.in_channels, conv1d_module.out_channels,
            conv1d_module.kernel_size, stride = conv1d_module.stride,
            padding = conv1d_module.padding, dilation = conv1d_module.dilation,
            groups = conv1d_module.groups,
            bias = (True if hasattr(conv1d_module, 'bias') else False),
            padding_mode = conv1d_module.padding_mode
        )
        self.tb_graph = tb_graph

    def forward(self, input):
        """ Input is an tensor wrapper for overloading the tensor operation and hook
        the process for tensor. It's an instance of TensorHook
        """
        result_tensor = super(Conv1dCustom, sef).forward(input._torch_tensor)

        node_name = _update_name_version(self)
        conv1d_node = _make_convnd_node(
            self.in_channels, self.out_channels, self.kernel_size,
            self.stride, self.padding, self.dilation,
            self.groups,
            True if hasattr(self, 'bias') else False,
            self.padding_mode,
            input._torch_tensor, node_name,
            conv_type="Conv1d"
        )
        self.tb_graph.node.extend([conv1d_node])
        # add a tensor map entry
        TensorToName.add_tensor_name(result_tensor, node_name)
        return TensorHook(result_tensor)


class Conv2dCustom(modules.Conv2d):
    def __init__(self, conv2d_module, tb_graph):
        """ Conv2dCustom is a wrapper for Conv2d

        Args:
            conv2d_module: it's an instance of Conv2d module
            tb_graph: it's an tb GraphDef message proto
        """
        super(Conv2dCustom, self).__init__(
            in_channels = conv2d_module.in_channels,
            out_channels = conv2d_module.out_channels,
            kernel_size = conv2d_module.kernel_size, stride = conv2d_module.stride,
            padding = conv2d_module.padding, dilation = conv2d_module.dilation,
            groups = conv2d_module.groups,
            bias = (True if hasattr(conv2d_module, 'bias') else False),
            padding_mode=conv2d_module.padding_mode
        )
        self.tb_graph = tb_graph

    def forward(self, input):
        """ Input is an tensor wrapper
        """
        result_tensor = super(Conv2dCustom, self).forward(input._torch_tensor)

        node_name = _update_name_version(self)
        conv2d_node = _make_convnd_node(
            self.in_channels, self.out_channels, self.kernel_size,
            self.stride, self.padding, self.dilation,
            self.groups,
            True if hasattr(self, 'bias') else False,
            self.padding_mode,
            input._torch_tensor, node_name,
            conv_type="Conv2d"
        )
        TensorToName.add_tensor_name(result_tensor, node_name)

        # add the generated node to tb graph
        self.tb_graph.node.extend([conv2d_node])

        return TensorHook(result_tensor)


class Conv3dCustom(modules.Conv3d):
    def __init__(self, conv3d_module, tb_graph):
        super(Conv3dCustom, self).__init__(
            conv3d_module.in_channels, conv3d_module.out_channels,
            conv3d_module.kernel_size, stride = conv3d_module.stride,
            padding = conv3d_module.padding, dilation = conv3d_module.dilation,
            groups = conv3d_module.groups,
            bias = (True if hasattr(conv3d_module, 'bias') else False),
            padding_mode = conv3d_module.padding_mode
        )
        self.tb_graph = tb_graph

    def forward(self, input):
        """ Input is an tensor wrapper
        """
        result_tensor = super(Conv3dCustom, self).forward(input._torch_tensor)

        node_name = _update_name_version(self)
        conv3d_node = _make_convnd_node(
            self.in_channels, self.out_channels, self.kernel_size,
            self.stride, self.padding, self.dilation,
            self.groups,
            True if hasattr(self, 'bias') else False,
            self.padding_mode,
            input._torch_tensor, node_name,
            conv_type="Conv3d"
        )
        TensorToName.add_tensor_name(result_tensor, node_name)

        # add the generated node to tb graph
        self.tb_graph.node.extend([conv3d_node])

        return TensorHook(result_tensor)


def _make_xrelu_node(input_tensor, module_name,
                     inplace=True, op_type="ReLU"):
    relu_node = node_def_pb2.NodeDef()
    relu_node.op = op_type
    relu_node.input.extend([TensorToName.get_tensor_name(input_tensor)])

    if inplace:
        # inplace relu
        _name = TensorToName.get_tensor_name(input_tensor).split('#')[0]
        __name_to_version__[_name] += 1
        node_name = "{}#{}".format(
            _name, __name_to_version__[_name]
        )
        relu_node.name = node_name
    else:
        # non-inplace relu
        if module_name in __name_to_version__:
            __name_to_version__[module_name] += 1
        else:
            __name_to_version__[module_name] = 0

        node_name = "{}#{}".format(
            module_name, __name_to_version__[module_name]
        )
        relu_node.name = node_name

    return relu_node


class ReLUCustom(modules.ReLU):
    def __init__(self, relu_module, tb_graph):
        super(ReLUCustom, self).__init__(relu_module.inplace)
        self.tb_graph = tb_graph

    def forward(self, input):
        """ Input is an tensor wrapper
        """
        result_tensor = super(ReLUCustom, self).forward(input._torch_tensor)

        relu_node = _make_xrelu_node(
            input._torch_tensor, ModuleToName.get_module_name(self),
            self.inplace, op_type="ReLU"
        )
        TensorToName.add_tensor_name(result_tensor, relu_node.name)

        self.tb_graph.node.extend([relu_node])
        return TensorHook(result_tensor)


class ReLU6Custom(modules.ReLU6):
    def __init__(self, relu6_module, tb_graph):
        super(ReLU6Custom, self).__init__(relu6_module.inplace)
        self.tb_graph = tb_graph

    def forward(self, input):
        """ Input is a tensor wrapper
        """
        result_tensor = super(ReLU6Custom, self).forward(input._torch_tensor)

        relu6_node = _make_xrelu_node(
            input._torch_tensor, ModuleToName.get_module_name(self),
            self.inplace, op_type="ReLU6"
        )
        TensorToName.add_tensor_name(result_tensor, relu6_node.name)

        self.tb_graph.node.extend([relu6_node])
        return TensorHook(result_tensor)


class LeakyReLUCustom(modules.LeakyReLU):
    def __init__(self, leakyrelu_module, tb_graph):
        super(LeakyReLUCustom, self).__init__(
            leakyrelu_module.negative_slope,
            leakyrelu_module.inplace
        )
        self.tb_graph = tb_graph

    def forward(self, input):
        """ Input is a tensor wrapper
        """
        result_tensor = super(LeakyReLUCustom, self).forward(input._torch_tensor)

        leakyrelu_node = _make_xrelu_node(
            input._torch_tensor, ModuleToName.get_module_name(self),
            self.inplace, op_type="LeakyReLU"
        )
        TensorToName.add_tensor_name(result_tensor, leakyrelu_node.name)

        self.tb_graph.node.extend([leakyrelu_node])
        return TensorHook(result_tensor)


def _make_avgpoolnd_node(kernel_size, stride, padding, ceil_mode, count_include_pad,
                         input_tensor, node_name, op_type="AvgPool2d"):
    avgpoolnd_node = node_def_pb2.NodeDef()
    avgpoolnd_node.op = op_type
    avgpoolnd_node.input.extend([TensorToName.get_tensor_name(input_tensor)])
    avgpoolnd_node.name = node_name

    if type(kernel_size) == tuple:
        _make_int_list_attr_value(avgpoolnd_node, "kernel_size", kernel_size)
    else:
        _make_int_attr_value(avgpoolnd_node, "kernel_size", kernel_size)

    if stride is not None:
        if type(stride) == tuple:
            _make_int_list_attr_value(avgpoolnd_node, "stride", stride)
        else:
            _make_int_attr_value(avgpoolnd_node, "stride", stride)

    if type(padding) == tuple:
        _make_int_list_attr_value(avgpoolnd_node, "padding", padding)
    else:
        _make_int_attr_value(avgpoolnd_node, "padding", padding)

    _make_bool_attr_value(avgpoolnd_node, "ceil_mode", ceil_mode)
    _make_bool_attr_value(avgpoolnd_node, "count_include_pad", count_include_pad)

    return avgpoolnd_node


class AvgPool1dCustom(modules.AvgPool1d):
    def __init__(self, avgpool1d_module, tb_graph):
        super(AvgPool1dCustom, self).__init__(
            avgpool1d_module.kernel_size, avgpool1d_module.stride,
            avgpool1d_module.padding, avgpool1d_module.ceil_mode,
            avgpool1d_module.count_include_pad
        )
        self.tb_graph = tb_graph

    def forward(self, input):
        """ Input is a tensor input
        """
        result_tensor = super(AvgPool1dCustom, self).forward(input._torch_tensor)

        node_name = _update_name_version(self)
        avgpool1d_node = _make_avgpoolnd_node(
            self.kernel_size, self.stride, self.padding, self.ceil_mode,
            self.count_include_pad, input._torch_tensor,
            node_name, op_type="AvgPool1d"
        )
        TensorToName.add_tensor_name(result_tensor, node_name)

        self.tb_graph.node.extend([avgpool1d_node])
        return TensorHook(result_tensor)


class AvgPool2dCustom(modules.AvgPool2d):
    def __init__(self, avgpool2d_module, tb_graph):
        super(AvgPool2dCustom, self).__init__(
            avgpool2d_module.kernel_size, avgpool2d_module.stride,
            avgpool2d_module.padding, avgpool2d_module.ceil_mode,
            avgpool2d_module.count_include_pad
        )
        self.tb_graph = tb_graph

    def forward(self, input):
        """ Input is a tensor input
        """
        result_tensor = super(AvgPool2dCustom, self).forward(input._torch_tensor)
        # result_tensor = F.avg_pool2d(input, self.kernel_size, self.stride,
        #     self.padding, self.ceil_mode, self.count_include_pad, self.divisor_override
        # )

        node_name = _update_name_version(self)
        avgpool2d_node = _make_avgpoolnd_node(
            self.kernel_size, self.stride, self.padding, self.ceil_mode,
            self.count_include_pad, input._torch_tensor,
            node_name, op_type="AvgPool2d"
        )
        TensorToName.add_tensor_name(result_tensor, node_name)

        self.tb_graph.node.extend([avgpool2d_node])
        return TensorHook(result_tensor)


class AvgPool3dCustom(modules.AvgPool3d):
    def __init__(self, avgpool3d_module, tb_graph):
        super(AvgPool3dCustom, self).__init__(
            avgpool3d_module.kernel_size, avgpool3d_module.stride,
            avgpool3d_module.padding, avgpool3d_module.ceil_mode,
            avgpool3d_module.count_include_pad
        )
        self.tb_graph = tb_graph

    def forward(self, input):
        """ Input is a tensor wrapper
        """
        result_tensor = super(AvgPool3dCustom, self).forward(input._torch_tensor)

        node_name = _update_name_version(self)
        avgpool3d_node = _make_avgpoolnd_node(
            self.kernel_size, self.stride, self.padding, self.ceil_mode,
            self.count_include_pad, input._torch_tensor,
            node_name, op_type="AvgPool3d"
        )
        TensorToName.add_tensor_name(result_tensor, node_name)

        self.tb_graph.node.extend([avgpool3d_node])
        return TensorHook(result_tensor)


class AdaptiveAvgPool1dCustom(modules.AdaptiveAvgPool1d):
    def __init__(self, adaptiveavgpool1d_module, tb_graph):
        super(AdaptiveAvgPool1dCustom, self).__init__(
            adaptiveavgpool1d_module.output_size,
            adaptiveavgpool1d_module.return_indices
        )
        self.tb_graph = tb_graph

    def forward(self, input):
        """ Input is a tensor wrapper
        """
        result_tensor = super(AdaptiveAvgPool1dCustom, self).forward(input)

        node_name = _update_name_version(self)
        adap_avgpool1d_node = node_def_pb2.NodeDef()
        adap_avgpool1d_node.input.extend([TensorToName.get_tensor_name(input._torch_tensor)])
        adap_avgpool1d_node.op = "AdaptiveAvgPool1d"
        adap_avgpool1d_node.name = node_name
        _make_int_attr_value(adap_avgpool1d_node, "output_size", output_size)
        _make_bool_attr_value(adap_avgpool1d_node, "return_indices", return_indices)

        TensorToName.add_tensor_name(result_tensor, node_name)

        self.tb_graph.node.extend([adap_avgpool1d_node])
        return TensorHook(result_tensor)


class AdaptiveAvgPool3dCustom(modules.AdaptiveAvgPool3d):
    def __init__(self, adaptiveavgpool3d_module, tb_graph):
        super(AdaptiveAvgPool3dCustom, self).__init__(
            adaptiveavgpool3d_module.output_size,
            adaptiveavgpool3d_module.return_indices
        )
        self.tb_graph = tb_graph

    def forward(self, input):
        """ Input is a tensor wrapper
        """
        result_tensor = super(AdaptiveAvgPool3dCustom, self).forward(input._torch_tensor)

        node_name = _update_name_version(self)
        adap_avgpool3d_node = node_def_pb2.NodeDef()
        adap_avgpool3d_node.input.extend([TensorToName.get_tensor_name(input._torch_tensor)])
        adap_avgpool3d_node.op = "AdaptiveAvgPool3d"
        adap_avgpool3d_node.name = node_name
        _make_int_list_attr_value(adap_avgpool3d_node, "output_size", output_size)
        _make_bool_attr_value(adap_avgpool3d_node, "return_indices", return_indices)

        TensorToName.add_tensor_name(result_tensor, node_name)

        self.tb_graph.node.extend([adap_avgpool3d_node])
        return TensorHook(result_tensor)


def _make_maxpoolnd_node(kernel_size, stride, padding, dilation, return_indices, ceil_mode,
                         input_tensor, node_name, op_type="MaxPool1d"):
    maxpoolnd_node = node_def_pb2.NodeDef()
    maxpoolnd_node.name = node_name
    maxpoolnd_node.input.extend([TensorToName.get_tensor_name(input_tensor)])
    maxpoolnd_node.op = op_type

    if type(kernel_size) == tuple:
        _make_int_list_attr_value(maxpoolnd_node, "kernel_size", kernel_size)
    else:
        _make_int_attr_value(maxpoolnd_node, "kernel_size", kernel_size)

    if type(stride) == tuple:
        _make_int_list_attr_value(maxpoolnd_node, "stride", stride)
    else:
        _make_int_attr_value(maxpoolnd_node, "stride", stride)

    if type(padding) == tuple:
        _make_int_list_attr_value(maxpoolnd_node, "padding", padding)
    else:
        _make_int_attr_value(maxpoolnd_node, "padding", padding)

    if type(dilation) == tuple:
        _make_int_list_attr_value(maxpoolnd_node, "dilation", dilation)
    else:
        _make_int_attr_value(maxpoolnd_node, "dilation", dilation)

    _make_bool_attr_value(maxpoolnd_node, "return_indices", return_indices)
    _make_bool_attr_value(maxpoolnd_node, "ceil_mode", ceil_mode)
    return maxpoolnd_node


class MaxPool1dCustom(modules.MaxPool1d):
    def __init__(self, maxpool1d_module, tb_graph):
        super(MaxPool1dCustom, self).__init__(
            maxpool1d_module.kernel_size, maxpool1d_module.stride,
            maxpool1d_module.padding, maxpool1d_module.dilation,
            maxpool1d_module.return_indices, maxpool1d_module.ceil_mode
        )
        self.tb_graph = tb_graph

    def forward(self, input):
        """ Input is a tensor wrapper
        """
        result_tensor = super(MaxPool1dCustom, self).forward(input._torch_tensor)

        node_name = _update_name_version(self)
        maxpool1d_node = _make_maxpoolnd_node(
            self.kernel_size, self.stride, self.padding, self.dilation,
            self.return_indices, self.ceil_mode,
            input._torch_tensor, node_name, op_type="MaxPool1d"
        )
        TensorToName.add_tensor_name(result_tensor, node_name)

        self.tb_graph.node.extend([maxpool1d_node])
        return TensorHook(result_tensor)


class MaxPool2dCustom(modules.MaxPool2d):
    def __init__(self, maxpool2d_module, tb_graph):
        super(MaxPool2dCustom, self).__init__(
            maxpool2d_module.kernel_size, maxpool2d_module.stride,
            maxpool2d_module.padding, maxpool2d_module.dilation,
            maxpool2d_module.return_indices, maxpool2d_module.ceil_mode
        )
        self.tb_graph = tb_graph

    def forward(self, input):
        """ Input is a tensor wrapper
        """
        result_tensor = super(MaxPool2dCustom, self).forward(input._torch_tensor)

        node_name = _update_name_version(self)
        maxpool2d_node = _make_maxpoolnd_node(
            self.kernel_size, self.stride, self.padding, self.dilation,
            self.return_indices, self.ceil_mode,
            input._torch_tensor, node_name, op_type="MaxPool2d"
        )
        TensorToName.add_tensor_name(result_tensor, node_name)

        self.tb_graph.node.extend([maxpool2d_node])
        return TensorHook(result_tensor)


def _make_batchnormnd_node(num_features, eps, momentum, affine, track_running_stats,
                           input_tensor, node_name, op_type="BatchNorm1d"):
    bn_node = node_def_pb2.NodeDef()
    bn_node.op = op_type
    bn_node.input.extend([TensorToName.get_tensor_name(input_tensor)])
    bn_node.name = node_name
    _make_int_attr_value(bn_node, "num_features", num_features)
    _make_float_attr_value(bn_node, "eps", eps)
    _make_float_attr_value(bn_node, "momentum", momentum)
    _make_bool_attr_value(bn_node, "affine", affine)
    _make_bool_attr_value(bn_node, "track_running_stats", track_running_stats)
    return bn_node


class BatchNorm1dCustom(modules.BatchNorm1d):
    def __init__(self, bn1d_module, tb_graph):
        super(BatchNorm1dCustom, self).__init__(
            bn1d_module.num_features, bn1d_module.eps,
            bn1d_module.momentum, bn1d_module.affine,
            bn1d_module.track_running_stats
        )
        self.tb_graph = tb_graph

    def forward(self, input):
        """ Input is a tensor wrapper
        """
        result_tensor = super(BatchNorm1dCustom, self).forward(input._torch_tensor)

        node_name = _update_name_version(self)
        bn1d_node = _make_batchnormnd_node(
            self.num_features, self.eps, self.momentum,
            self.affine, self.track_running_stats,
            input._torch_tensor, node_name, op_type="BatchNorm1d"
        )
        TensorToName.add_tensor_name(result_tensor, node_name)

        self.tb_graph.node.extend([bn1d_node])
        return TensorHook(result_tensor)


class BatchNorm2dCustom(modules.BatchNorm2d):
    def __init__(self, bn2d_module, tb_graph):
        super(BatchNorm2dCustom, self).__init__(
            bn2d_module.num_features, bn2d_module.eps,
            bn2d_module.momentum, bn2d_module.affine,
            bn2d_module.track_running_stats
        )
        self.tb_graph = tb_graph

    def forward(self, input):
        """ Input is a tensor wrapper
        """
        result_tensor = super(BatchNorm2dCustom, self).forward(input._torch_tensor)

        node_name = _update_name_version(self)
        bn2d_node = _make_batchnormnd_node(
            self.num_features, self.eps, self.momentum,
            self.affine, self.track_running_stats,
            input._torch_tensor, node_name, op_type="BatchNorm2d"
        )
        TensorToName.add_tensor_name(result_tensor, node_name)

        self.tb_graph.node.extend([bn2d_node])
        return TensorHook(result_tensor)


class BatchNorm3dCustom(modules.BatchNorm3d):
    def __init__(self, bn3d_module, tb_graph):
        super(BatchNorm3dCustom, self).__init__(
            bn3d_module.num_features, bn3d_module.eps,
            bn3d_module.momentum, bn3d_module.affine,
            bn3d_module,track_running_stats
        )
        self.tb_graph = tb_graph

    def forward(self, input):
        """ Input is a tensor wrapper
        """
        result_tensor = super(BatchNorm3dCustom, self).forward(input._torch_tensor)

        node_name = _update_name_version(self)
        bn3d_node = _make_batchnormnd_node(
            self.num_features, self.eps, self.momentum,
            self.affine, self.track_running_stats,
            input._torch_tensor, node_name, op_type="BatchNorm3d"
        )
        TensorToName.add_tensor_name(result_tensor, node_name)

        self.tb_graph.node.extend([bn3d_node])
        return TensorHook(result_tensor)


def _make_softmax_node(dim, input_tensor, node_name):
    softmax_node = node_def_pb2.NodeDef()
    softmax_node.op = "Softmax"
    softmax_node.input.extend([TensorToName.get_tensor_name(input_tensor)])
    softmax_node.name = node_name
    _make_int_attr_value(softmax_node, "dim", dim)
    return softmax_node


class SoftmaxCustom(modules.Softmax):
    def __init__(self, softmax_module, tb_graph):
        super(SoftmaxCustom, self).__init__(
            softmax_module.dim
        )
        self.tb_graph = tb_graph

    def forward(self, input):
        """ Input is a tensor wrapper
        """
        result_tensor = super(SoftmaxCustom, self).forward(input._torch_tensor)

        node_name = _update_name_version(self)
        softmax_node = _make_softmax_node(self.dim, input._torch_tensor, node_name)

        TensorToName.add_tensor_name(result_tensor, node_name)

        self.tb_graph.node.extend([softmax_node])
        return TensorHook(result_tensor)


def _make_linear_node(in_features, out_features, bias, input_tensor, node_name):
    linear_node = node_def_pb2.NodeDef()
    linear_node.op = "Linear"
    linear_node.name = node_name
    linear_node.input.extend([TensorToName.get_tensor_name(input_tensor)])
    _make_int_attr_value(linear_node, "in_features", in_features)
    _make_int_attr_value(linear_node, "out_features", out_features)
    _make_bool_attr_value(linear_node, "bias", bias)
    return linear_node


class LinearCustom(modules.Linear):
    def __init__(self, linear_module, tb_graph):
        super(LinearCustom, self).__init__(
            linear_module.in_features, linear_module.out_features,
            bias = True if hasattr(linear_module, 'bias') else False
        )
        self.tb_graph = tb_graph

    def forward(self, input):
        """ Input is a tensor wrapper
        """
        result_tensor = super(LinearCustom, self).forward(input._torch_tensor)

        node_name = _update_name_version(self)

        linear_node = _make_linear_node(self.in_features, self.out_features,
                                        True if hasattr(self, 'bias') else False,
                                        input._torch_tensor, node_name)
        TensorToName.add_tensor_name(result_tensor, node_name)

        self.tb_graph.node.extend([linear_node])
        return TensorHook(result_tensor)


def _make_zeropad2d_node(padding, input_tensor, node_name):
    assert isinstance(input_tensor, torch.Tensor)
    zeropad2d_node = node_def_pb2.NodeDef()
    zeropad2d_node.op = "ZeroPad2d"
    zeropad2d_node.input.extend([TensorToName.get_tensor_name(input_tensor)])
    zeropad2d_node.name = node_name

    if type(padding) == int:
        _make_int_attr_value(zeropad2d_node, "padding", padding)
    else:
        _make_int_list_attr_value(zeropad2d_node, "padding", padding)
    return zeropad2d_node


class ZeroPad2dCustom(modules.ZeroPad2d):
    def __init__(self, zeropad2d_module, tb_graph):
        super(ZeroPad2dCustom, self).__init__(
            zeropad2d_module.padding
        )
        self.tb_graph = tb_graph

    def forward(self, input):
        """ Input is a tensor wrapper
        """
        result_tensor = super(ZeroPad2dCustom, self).forward(input._torch_tensor)

        node_name = _update_name_version(self)
        zeropad2d_node = _make_zeropad2d_node(
            self.padding, input._torch_tensor, node_name
        )
        TensorToName.add_tensor_name(result_tensor, node_name)

        self.tb_graph.node.extend([zeropad2d_node])

        return TensorHook(result_tensor)


def _make_xdropout_node(input_tensor, module_name, drop_rate,
                        inplace=True, op_type="Dropout"):
    xdropout_node = node_def_pb2.NodeDef()
    xdropout_node.op = op_type
    xdropout_node.input.extend([TensorToName.get_tensor_name(input_tensor)])
    _make_float_attr_value(xdropout_node, "drop_rate", drop_rate)

    node_name = None
    if inplace:
        # inplace dropout
        _name = TensorToName.get_tensor_name(input_tensor).split('#')[0]
        __name_to_version__[_name] += 1
        node_name = "{}#{}".format(
            _name, __name_to_version__[_name]
        )
    else:
        # non-inplace dropout
        if module_name in __name_to_version__:
            __name_to_version__[module_name] += 1
        else:
            __name_to_version__[module_name] = 0
        node_name = "{}#{}".format(
            module_name, __name_to_version__[module_name]
        )

    xdropout_node.name = node_name
    return xdropout_node


class DropoutCustom(modules.Dropout):
    def __init__(self, dropout_module, tb_graph):
        super(DropoutCustom, self).__init__(
            dropout_module.p, dropout_module.inplace
        )
        self.tb_graph = tb_graph

    def forward(self, input):
        """ Input is an tensor wrapper
        """
        result_tensor = super(DropoutCustom, self).forward(input._torch_tensor)
        dropout_node = _make_xdropout_node(
            input._torch_tensor, ModuleToName.get_module_name(self),
            self.p, self.inplace, op_type="Dropout"
        )
        TensorToName.add_tensor_name(result_tensor, dropout_node.name)
        self.tb_graph.node.extend([dropout_node])
        return TensorHook(result_tensor)


class Dropout2dCustom(modules.Dropout2d):
    def __init__(self, dropout2d_module, tb_graph):
        super(Dropout2dCustom, self).__init__(
            dropout2d_module.p, dropout2d_module.inplace
        )
        self.tb_graph = tb_graph

    def forward(self, input):
        """ Input is a tensor wrapper
        """
        result_tensor = super(Dropout2dCustom, self).forward(input._torch_tensor)

        dropout2d_node = _make_xdropout_node(
            input._torch_tensor, ModuleToName.get_module_name(self),
            self.p, self.inplace, op_type="Dropout2d"
        )
        TensorToName.add_tensor_name(result_tensor, dropout2d_node.name)

        self.tb_graph.node.extend([dropout2d_node])
        return TensorHook(result_tensor)


class AlphaDropoutCustom(modules.AlphaDropout):
    def __init__(self, alphadropout_module, tb_graph):
        super(AlphaDropoutCustom, self).__init__(
            alphadropout_module.p, alphadropout_module.inplace
        )
        self.tb_graph = tb_graph

    def forward(self, input):
        """ Input is a tensor wrapper
        """
        result_tensor = super(AlphaDropoutCustom, self).forward(input._torch_tensor)

        alphadropout_node = _make_xdropout_node(
            input._torch_tensor, ModuleToName.get_module_name(self),
            self.inplace, op_type="AlphaDropout"
        )
        TensorToName.add_tensor_name(result_tensor, alphadropout_node.name)

        self.tb_graph.node.extend([alphadropout_node])
        return TensorHook(result_tensor)


class TensorHook(object):
    """ TensorHook is a wrapper for torch Tensor

    Because the network for pytorch is dynamic, its details are got to be known when
    the network is forwarded
    And for pytorch, two different important components include `Tensor` and `Module`.
    However, the module can be hooked before pre-forward processing and `Tensor` can't
    be done like this.
    Therefore, we build a wrapper for tensor
    """
    # a unique number which indicates the tensor version
    __tensor_seq__ = 0

    # tb graph def
    __tb_graph__ = TorchTBGraph.torch_tb_graph

    def __init__(self, torch_tensor):
        self._torch_tensor = torch_tensor

    def __add__(self, value, out=None):
        """ Here value and out is a TensorHook, which is a tensor wrapper """
        out_tensor = self._torch_tensor + value._torch_tensor
        TensorToName.add_tensor_name(out_tensor,
                                     "{}#{}".format("tensor", self.__tensor_seq__))

        # upate version
        tensor_add_node = node_def_pb2.NodeDef()
        tensor_add_node.op = "Add"
        tensor_add_node.input.extend([TensorToName.get_tensor_name(self._torch_tensor)])
        tensor_add_node.input.extend([TensorToName.get_tensor_name(value._torch_tensor)])
        tensor_add_node.name = TensorToName.get_tensor_name(out_tensor)
        self.__tb_graph__.node.extend([tensor_add_node])
        self.__tensor_seq__ += 1

        if out is not None:
            out._torch_tensor = out_tensor
        else:
            return TensorHook(out_tensor)

    def size(self, dim):
        return self._torch_tensor.size(dim)

    def dim(self):
        return self._torch_tensor.dim()

    def contiguous(self):
        """ Returns a contiguous tensor containing the same data as :attr:`self` tensor """
        result_tensor = self._torch_tensor.contiguous()

        # create a contiguous node
        contiguous_node = node_def_pb2.NodeDef()
        contiguous_node.input.extend([TensorToName.get_tensor_name(self._torch_tensor)])
        contiguous_node.op = "Contiguous"
        contiguous_node.name = "{}#{}".format("tensor", self.__tensor_seq__)
        TensorToName.add_tensor_name(result_tensor, contiguous_node.name)

        # update version
        self.__tb_graph__.node.extend([contiguous_node])
        self.__tensor_seq__ += 1

        return TensorHook(result_tensor)


    def __iadd__(self, value):
        """ value is a TensorHook type, which is a tensor wrapper """
        tensor_iadd_node = node_def_pb2.NodeDef()
        tensor_iadd_node.op = "Add"
        tensor_iadd_node.input.extend([TensorToName.get_tensor_name(self._torch_tensor)])
        tensor_iadd_node.input.extend([TensorToName.get_tensor_name(value._torch_tensor)])

        self._torch_tensor += value._torch_tensor
        # update version
        self.__tensor_seq__ += 1
        _name = TensorToName.get_tensor_name(self._torch_tensor).split('#')[0]
        __name_to_version__[_name] += 1
        TensorToName.add_tensor_name(self._torch_tensor,
                                     "{}#{}".format(_name, __name_to_version__[_name]))
        tensor_iadd_node.name = TensorToName.get_tensor_name(self._torch_tensor)

        # add this node to tb graph
        self.__tb_graph__.node.extend([tensor_iadd_node])
        return self

    def __sub__(self, value, out=None):
        """ Here value and out is a TensorHook, which is a tensor wrapper """
        out_tensor = self._torch_tensor - value._torch_tensor
        TensorToName.add_tensor_name(
            out_tensor,
            "{}#{}".format("tensor", self.__tensor_seq__)
        )

        # create new tensor
        tensor_sub_node = node_def_pb2.NodeDef()
        tensor_sub_node.op = "Sub"
        tensor_sub_node.input.extend([TensorToName.get_tensor_name(self._torch_tensor)])
        tensor_sub_node.input.extend([TensorToName.get_tensor_name(value._torch_tensor)])
        tensor_sub_node.name = TensorToName.get_tensor_name(out_tensor)
        self.__tb_graph__.node.extend([tensor_sub_node])
        self.__tensor_seq__ += 1

        if out is not None:
            out._torch_tensor = out_tensor
        else:
            return TensorHook(out_tensor)

    def __isub__(self, value):
        """ value is a TensorHook type, which is a tensor wrapper """
        tensor_isub_node = node_def_pb2.NodeDef()
        tensor_isub_node.op = "Sub"
        tensor_isub_node.input.extend([TensorToName.get_tensor_name(self._torch_tensor)])
        tensor_isub_node.input.extend([TensorToName.get_tensor_name(value._torch_tensor)])

        self._torch_tensor -= value._torch_tensor
        # update version
        self.__tensor_seq__ += 1
        _name = TensorToName.get_tensor_name(self._torch_tensor).split('#')[0]
        __name_to_version__[_name] += 1
        TensorToName.add_tensor_name(
            self._torch_tensor, "{}#{}".format(_name, __name_to_version__[_name])
        )
        tensor_isub_node.name = TensorToName.get_tensor_name(self._torch_tensor)

        # add isub node to tb graph
        self.__tb_graph__.node.extend([tensor_isub_node])

    def view(self, *args):
        """ convert Tensor view to reshape operation """
        view_tensor = self._torch_tensor.view(*args)
        view_node = node_def_pb2.NodeDef()
        view_node.input.extend([TensorToName.get_tensor_name(self._torch_tensor)])

        # shape attribute
        shape_attr = attr_value_pb2.AttrValue()
        shape_list_value = attr_value_pb2.AttrValue.ListValue()
        shape_list_value.i.extend(args)
        shape_attr.list.CopyFrom(shape_list_value)
        view_node.attr['reshape'].CopyFrom(shape_attr)

        view_node.name = "{}#{}".format("tensor", self.__tensor_seq__)
        view_node.op = "Reshape"
        TensorToName.add_tensor_name(
            view_tensor, view_node.name
        )
        self.__tb_graph__.node.extend([view_node])
        __name_to_version__["tensor"] = self.__tensor_seq__

        # update version
        self.__tensor_seq__ += 1

        return TensorHook(view_tensor)

    def mean(self, dim, keepdim=False):
        """ input argument `dim` may be a list of dimensions or a single int """
        mean_tensor = self._torch_tensor.mean(dim, keepdim=keepdim)

        mean_node = node_def_pb2.NodeDef()
        mean_node.input.extend([TensorToName.get_tensor_name(self._torch_tensor)])

        if type(dim) == int:
            _make_int_attr_value(mean_node, "dim", dim)
        else:
            _make_int_list_attr_value(mean_node, "dim", dim)
        _make_bool_attr_value(mean_node, "keepdim", keepdim)

        mean_node.op = "Mean"
        mean_node.name = "{}#{}".format("tensor", self.__tensor_seq__)
        TensorToName.add_tensor_name(mean_tensor, mean_node.name)

        __name_to_version__["tensor"] = self.__tensor_seq__
        self.__tb_graph__.node.extend([mean_node])

        # upate version
        self.__tensor_seq__ += 1

        return TensorHook(mean_tensor)

    def __getitem__(self, *key):
        """ This is an overload operation [] for processing tensor """
        getitem_tensor = self._torch_tensor.__getitem__(*key)

        getitem_node = node_def_pb2.NodeDef()
        getitem_node.input.extend([TensorToName.get_tensor_name(self._torch_tensor)])
        _make_bytes_attr_value(getitem_node, "slice", str(key))
        getitem_node.op = "GetItem"
        getitem_node.name = "{}#{}".format(
            "numpy", self.__tensor_seq__
        )

        TensorToName.add_tensor_name(getitem_tensor, getitem_node.name)
        self.__tb_graph__.node.extend([getitem_node])

        # update version
        self.__tensor_seq__ += 1
        return TensorHook(getitem_tensor)


    def __mul__(self, value, out=None):
        """ Here value and out is a TensorHook, which is a tensor wrapper """
        mul_tensor = self._torch_tensor * value._torch_tensor

        mul_node = node_def_pb2.NodeDef()
        mul_node.input.extend([TensorToName.get_tensor_name(self._torch_tensor)])
        mul_node.input.extend([TensorToName.get_tensor_name(value._torch_tensor)])
        mul_node.op = "Mul"
        mul_node.name = "{}#{}".format(
            "numpy", self.__tensor_seq__
        )
        TensorToName.add_tensor_name(mul_tensor, mul_node.name)

        self.__tb_graph__.node.extend([mul_node])

        # update version
        self.__tensor_seq__ += 1
        if out is not None:
            raise NotImplementedError("You can't provide out parameter to hold result")
        else:
            return TensorHook(mul_tensor)

    def __imul__(self, value):
        """ value is a TensorHook type, which is a tensor wrapper """
        mul_node = node_def_pb2.NodeDef()
        mul_node.input.extend([TensorToName.get_tensor_name(self._torch_tensor)])
        mul_node.input.extend([TensorToName.get_tensor_name(value._torch_tensor)])
        mul_node.op = "Mul"

        self._torch_tensor *= value._torch_tensor
        _name = TensorToName.get_tensor_name(self._torch_tensor).split('#')[0]
        __name_to_version__[_name] += 1
        mul_node.name = "{}#{}".format(_name, __name_to_version__[_name])

    def __div__(self, value, out=None):
        """ Here value and out is a TensorHook, which is a tensor wrapper """
        if out is not None:
            out._torch_tensor = self._torch_tensor / value._torch_tensor
        else:
            return TensorHook(self._torch_tensor / value._torch_tensor)

    def __idiv__(self, value):
        """ value is a TensorHook type, which is a tensor wrapper """
        self._torch_tensor /= value._torch_tensor

# map from the original module into custom module class
__module_class__ = {
    "Conv1d":            Conv1dCustom,
    "Conv2d":            Conv2dCustom,
    "Conv3d":            Conv3dCustom,

    "ReLU":              ReLUCustom,
    "ReLU6":             ReLU6Custom,
    "LeakyReLU":         LeakyReLUCustom,

    "BatchNorm1d":       BatchNorm1dCustom,
    "BatchNorm2d":       BatchNorm2dCustom,
    "BatchNorm3d":       BatchNorm3dCustom,

    "Dropout":           DropoutCustom,
    "Dropout2d":         Dropout2dCustom,
    "AlphaDropout":      AlphaDropoutCustom,

    "ZeroPad2d":         ZeroPad2dCustom,

    "MaxPool1d":         MaxPool1dCustom,
    "MaxPool2d":         MaxPool2dCustom,

    "AvgPool1d":         AvgPool1dCustom,
    "AvgPool2d":         AvgPool2dCustom,
    "AvgPool3d":         AvgPool3dCustom,

    "AdaptiveAvgPool1d": AdaptiveAvgPool1dCustom,
    "AdaptiveAvgPool3d": AdaptiveAvgPool3dCustom,

    "Linear":            LinearCustom,
    "Softmax":           SoftmaxCustom,
}


def _make_input_data_node(input_tensor, shape_size):
    """ Build input data node for placeholder

    Arg:
        input_tensor: an instance of TensorHook
        shape_size: a tuple containing the tensor shape
    """
    data_node = node_def_pb2.NodeDef()
    data_node.op = "data"
    data_node.name = "data"

    shape_attr = attr_value_pb2.AttrValue()
    shape_list_value = attr_value_pb2.AttrValue.ListValue()
    shape_list_value.i.extend([_shape for _shape in shape_size])
    shape_attr.list.CopyFrom(shape_list_value)
    data_node.attr['shape'].CopyFrom(shape_attr)

    return data_node


def freeze_graph(nn_module, input):
    """ Freeze the module architecture for the pytorch module

    Args:
        nn_module: pytorch module
        input: a pytorch Tensor

    Return:
        a node-based tb graph def
    """

    def __build_modules_internal(module, tb_graph, prefix):
        for name, _module in module.named_children():
            if _module.__class__.__name__ in __class_modules__:
                custom_class = __module_class__[_module.__class__.__name__]
                new_custom_module = custom_class(_module, tb_graph)
                ModuleToName.add_module_name(new_custom_module, "{}/{}".format(prefix, name))
                setattr(module, name, new_custom_module)
            else:
                __build_modules_internal(_module, tb_graph,
                                         "{}/{}".format(prefix, name))

    def _build_modules(module, tb_graph):
        for name, _module in module.named_children():
            if _module.__class__.__name__ in __class_modules__:
                custom_class = __module_class__[_module.__class__.__name__]
                new_custom_module = custom_class(_module, tb_graph)
                ModuleToName.add_module_name(new_custom_module, name)
                setattr(module, name, new_custom_module)
            else:
                __build_modules_internal(_module, tb_graph, name)

    class TBModule(object):
        def __init__(self, nn_module, tb_graph):
            self.nn_module = nn_module
            self.tb_graph = tb_graph

        def build_custom_module(self):
            _build_modules(self.nn_module, self.tb_graph)

        def __call__(self, input_tensor):
            self.nn_module(input_tensor)


    # In order to build tensorboard graph with all python, we need to handle
    # the numpy interfaces for torch. We replace all the torch numpy symbol
    # with the wrap function
    import torch
    class NumpyWrap(object):
        __seq__ = 0
        __tb_graph__ = TorchTBGraph.torch_tb_graph

    class TorchAbs(object):
        def __init__(self, torch_abs):
            self._torch_abs = torch_abs

        def __call__(self, input, out=None):
            input = input._torch_tensor
            abs_node = node_def_pb2.NodeDef()
            abs_node.input.extend([TensorToName.get_tensor_name(input)])
            abs_node.op = "Abs"
            if out is not None:
                raise NotImplementedError("You can't provide out parameter to hold result")
            else:
                result_tensor = self._torch_abs(input)
                abs_node.name = "{}#{}".format(
                    "numpy", NumpyWrap.__seq__
                )
                TensorToName.add_tensor_name(result_tensor, abs_node.name)
                __name_to_version__["numpy"] = NumpyWrap.__seq__
                NumpyWrap.__tb_graph__.node.extend([abs_node])
                NumpyWrap.__seq__ += 1
                return TensorHook(result_tensor)

    torch.abs = TorchAbs(torch.abs)

    class TorchUnsqueeze(object):
        def __init__(self, torch_unsqueeze):
            self._torch_unsqueeze = torch_unsqueeze

        def __call__(self, input, dim):
            if instance(input, torch.Tensor):
                return self._torch_unsqueeze(input)

            unsqueeze_node = node_def_pb2.NodeDef()
            unsqueeze_node.input.extend([TensorToName.get_tensor_name(input)])
            unsqueeze_node.op = "Unsqueeze"
            _make_int_attr_value(unsqueeze_node, "dim", dim)

            result_tensor = self._torch_unsqueeze(input, dim)
            unsqueeze_node.name = "{}#{}".format(
                "numpy", NumpyWrap.__seq__
            )
            TensorToName.add_tensor_name(result_tensor, unsqueeze_node.name)
            __name_to_version__["numpy"] = NumpyWrap.__seq__
            NumpyWrap.__tb_graph__.node.extend([unsqueeze_node])
            NumpyWrap.__seq__ += 1
            return TensorHook(result_tensor)

    torch.unsqueeze = TorchUnsqueeze

    class TorchFlatten(object):
        def __init__(self, torch_flatten):
            self._torch_flatten = torch_flatten

        def __call__(self, input, start_dim=0, end_dim=-1):
            if isinstance(input, torch.Tensor):
                # it has been handled by upper module
                return self._torch_flatten(input)

            input = input._torch_tensor
            flatten_node = node_def_pb2.NodeDef()
            flatten_node.input.extend([TensorToName.get_tensor_name(input)])
            flatten_node.op = "Flatten"
            _make_int_attr_value(flatten_node, "start_dim", start_dim)
            _make_int_attr_value(flatten_node, "end_dim", end_dim)

            result_tensor = self._torch_flatten(input, start_dim, end_dim)
            flatten_node.name = "{}#{}".format(
                "numpy", NumpyWrap.__seq__
            )
            TensorToName.add_tensor_name(result_tensor, flatten_node.name)
            __name_to_version__["numpy"] = NumpyWrap.__seq__
            NumpyWrap.__tb_graph__.node.extend([flatten_node])
            NumpyWrap.__seq__ += 1
            return TensorHook(result_tensor)

    torch.flatten = TorchFlatten(torch.flatten)

    class TorchCat(object):
        def __init__(self, torch_cat):
            self._torch_cat = torch_cat

        def __call__(self, tensors, dim=0):
            input_tensors = [_tensor._torch_tensor for _tensor in tensors]

            cat_node = node_def_pb2.NodeDef()
            cat_node.input.extend([TensorToName.get_tensor_name(input) for input in input_tensors])
            cat_node.op = "Concat"
            _make_int_attr_value(cat_node, "dim", dim)

            result_tensor = self._torch_cat(input_tensors, dim)
            cat_node.name = "{}#{}".format(
                "numpy", NumpyWrap.__seq__
            )

            TensorToName.add_tensor_name(result_tensor, cat_node.name)
            __name_to_version__["numpy"] = NumpyWrap.__seq__
            NumpyWrap.__tb_graph__.node.extend([cat_node])
            NumpyWrap.__seq__ += 1
            return TensorHook(result_tensor)

    torch.cat = TorchCat(torch.cat)

    class TorchRelu(object):
        def __init__(self, torch_relu):
            # no-inplace relu
            self._torch_relu = torch_relu

        def __call__(self, input):
            input = input._torch_tensor

            relu_node = node_def_pb2.NodeDef()
            relu_node.input.extend([TensorToName.get_tensor_name(input)])
            relu_node.op = "Relu"

            result_tensor = self._torch_relu(input)
            relu_node.name = "{}#{}".format(
                "numpy", NumpyWrap.__seq__
            )
            TensorToName.add_tensor_name(result_tensor, relu_node.name)
            __name_to_version__["numpy"] = NumpyWrap.__seq__
            NumpyWrap.__tb_graph__.node.extend([relu_node])
            NumpyWrap.__seq__ += 1
            return TensorHook(result_tensor)

    torch.relu = TorchRelu(torch.relu)

    class TorchRelu_(object):
        def __init__(self, torch_relu_):
            # inplace relu
            self._torch_relu_ = torch_relu_

        def __call__(self, input):
            if isinstance(input, torch.Tensor):
                # it has been handled by upper module
                return self._torch_relu_(input)

            input = input._torch_tensor

            relu_node = node_def_pb2.NodeDef()
            relu_node.input.extend([TensorToName.get_tensor_name(input)])
            relu_node.op = "Relu_"

            _name = TensorToName.get_tensor_name(input).split('#')[0]
            __name_to_version__[_name] += 1

            result_tensor = self._torch_relu_(input)
            relu_node.name = "{}#{}".format(_name, __name_to_version__[_name])
            TensorToName.add_tensor_name(result_tensor, relu_node.name)

            NumpyWrap.__tb_graph__.node.extend([relu_node])
            NumpyWrap.__seq__ += 1

            return TensorHook(result_tensor)

    torch.relu_ = TorchRelu_(torch.relu_)

    class TorchPadding(object):
        def __init__(self, torch_padding):
            self.torch_padding = torch_padding

        def __call__(self, input, pad, mode='constant', value=0):
            if type(input) == torch.Tensor:
                return self.torch_padding(input, pad, mode=mode, value=value)

            input = input._torch_tensor
            padding_node = node_def_pb2.NodeDef()
            padding_node.input.extend([TensorToName.get_tensor_name(input)])
            padding_node.op = "Pad"
            padding_node.name = "{}#{}".format(
                "numpy", NumpyWrap.__seq__
            )
            result_tensor = self.torch_padding(input, pad, mode=mode, value=value)
            TensorToName.add_tensor_name(result_tensor, padding_node.name)

            NumpyWrap.__tb_graph__.node.extend([padding_node])
            NumpyWrap.__seq__ += 1

            return TensorHook(result_tensor)

    F.pad = TorchPadding(F.pad)

    class TorchAdaptiveAvgPool2d(object):
        def __init__(self, torch_adaptive_avg_pool2d):
            self.torch_adaptive_avg_pool2d = torch_adaptive_avg_pool2d

        def __call__(self, input, output_size):
            input = input._torch_tensor

            adap_avgpool2d_node = node_def_pb2.NodeDef()
            adap_avgpool2d_node.input.extend([TensorToName.get_tensor_name(input)])
            adap_avgpool2d_node.op = "AdaptiveAvgPool2d"
            adap_avgpool2d_node.name = "{}#{}".format(
                "numpy", NumpyWrap.__seq__
            )
            result_tensor = self.torch_adaptive_avg_pool2d(input, output_size)
            TensorToName.add_tensor_name(result_tensor, adap_avgpool2d_node.name)
            NumpyWrap.__tb_graph__.node.extend([adap_avgpool2d_node])
            NumpyWrap.__seq__ += 1

            return TensorHook(result_tensor)

    F.adaptive_avg_pool2d = TorchAdaptiveAvgPool2d(F.adaptive_avg_pool2d)

    class TorchMaxPool2d(object):
        def __init__(self, torch_maxpool2d):
            self.torch_maxpool2d = torch_maxpool2d

        def __call__(self, input, kernel_size, stride,
                     padding=0, dilation=1, ceil_mode=False, return_indices=False):
            if type(input) == torch.Tensor:
                # it has been handled by upper module-based wrapper
                return self.torch_maxpool2d(input, kernel_size, stride, padding,
                                            dilation, ceil_mode, return_indices)
            else:
                # it's called from the functional
                input = input._torch_tensor

                result_tensor = self.torch_maxpool2d(
                    input, kernel_size, stride, padding, dilation,
                    ceil_mode, return_indices
                )

                node_name = "{}#{}".format("numpy", NumpyWrap.__seq__)
                maxpool2d_node = _make_maxpoolnd_node(
                    kernel_size, stride, padding, dilation,
                    return_indices, ceil_mode,
                    input, node_name, op_type="MaxPool2d"
                )

                TensorToName.add_tensor_name(result_tensor, node_name)
                NumpyWrap.__tb_graph__.node.extend([maxpool2d_node])
                NumpyWrap.__seq__ += 1
                __name_to_version__["numpy"] = NumpyWrap.__seq__

                return TensorHook(result_tensor)

    F.max_pool2d = TorchMaxPool2d(F.max_pool2d)

    class TorchAvgPool2d(object):
        def __init__(self, torch_avgpool2d):
            self.torch_avgpool2d = torch_avgpool2d

        def __call__(self, input, kernel_size, stride=None,
                     padding=0, ceil_mode=False, count_include_pad=True,
                     divisor_override=None):
            if type(input) == torch.Tensor:
                # it has been handled by upper module-based wrapper
                return self.torch_avgpool2d(
                    input, kernel_size, stride, padding, ceil_mode,
                    count_include_pad, divisor_override
                )
            else:
                # it's called from the functional first
                input = input._torch_tensor

                result_tensor = self.torch_avgpool2d(
                    input, kernel_size, stride, padding, ceil_mode,
                    count_include_pad, divisor_override
                )
                node_name = "{}#{}".format("numpy", NumpyWrap.__seq__)
                avgpool2d_node = _make_avgpoolnd_node(
                    kernel_size, stride, padding, ceil_mode, count_include_pad,
                    input, node_name, op_type="AvgPool2d"
                )

                TensorToName.add_tensor_name(result_tensor, node_name)
                NumpyWrap.__tb_graph__.node.extend([avgpool2d_node])
                NumpyWrap.__seq__ += 1
                __name_to_version__["numpy"] = NumpyWrap.__seq__

                return TensorHook(result_tensor)

    F.avg_pool2d = TorchAvgPool2d(F.avg_pool2d)

    class TorchDropout(object):
        def __init__(self, torch_dropout):
            self.torch_dropout = torch_dropout

        def __call__(self, input, p=0.5, training=True, inplace=False):
            if type(input) == torch.Tensor:
                return self.torch_dropout(input, p, training, inplace)

            input = input._torch_tensor

            dropout_node = node_def_pb2.NodeDef()
            dropout_node.input.extend([TensorToName.get_tensor_name(input)])
            result_tensor = self.torch_dropout(input, p=p, training=training, inplace=inplace)

            dropout_node.op = "Dropout"
            if inplace:
                # inplace dropout
                _name = TensorToName.get_tensor_name(input).split('#')[0]
                __name_to_version__[_name] += 1
                dropout_node.name = "{}#{}".format(
                    _name, __name_to_version__[_name]
                )
            else:
                # non-inplace dropout
                dropout_node.name = "{}#{}".format(
                    "numpy", NumpyWrap.__seq__
                )
                NumpyWrap.__seq__ += 1

            TensorToName.add_tensor_name(result_tensor, dropout_node.name)
            NumpyWrap.__tb_graph__.node.extend([dropout_node])
            return TensorHook(result_tensor)

    F.dropout = TorchDropout(F.dropout)

    # add an input data tensor
    input_tensor = TensorHook(input)
    TensorToName.add_tensor_name(input_tensor._torch_tensor, "data")
    data_node = _make_input_data_node(input_tensor, input_tensor._torch_tensor.size())
    TorchTBGraph.torch_tb_graph.node.extend([data_node])

    tb_module = TBModule(nn_module, TorchTBGraph.torch_tb_graph)
    tb_module.build_custom_module()
    tb_module(input_tensor)

    return TorchTBGraph.torch_tb_graph
