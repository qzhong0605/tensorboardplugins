# Convert an layer-based caffe network into node-based TensorBoard network
#
# There are many differences between caffe network and TensorBoard network.
# 1. You can perform inplace operation on caffe, while you must perform
# SSA-format operation on Tensorboard network
# 2. For Tensorboard network, there isn't any output on the node. They are
# connectd by inputs
#
################################################################################

from tensorboard.compat.proto import types_pb2
from tensorboard.compat.proto import graph_pb2
from tensorboard.compat.proto import node_def_pb2
from tensorboard.compat.proto import attr_value_pb2
from tensorboard.compat.proto import tensor_shape_pb2
from tensorboard.compat.proto.caffe import caffe_pb2

from tensorboard.plugins.graph_edit import tbgraph_base

from google.protobuf import text_format


def _make_filler_attr(attr, filler):
    """ Convert an caffe filler message into TensorBoard Attribute Message.
    It's built into a map from name to AttrValue

    Args:
        attr: a AttrValue message for TensorBoard proto
        filler: a FillerParameter message on caffe proto
    """
    name_attr_list = attr_value_pb2.NameAttrList()
    name_attr_list.name = "filler"

    if filler.HasField('type'):
        attr_value = attr_value_pb2.AttrValue()
        attr_value.s = bytes(filler.type, "utf-8")
        name_attr_list.attr['type'].CopyFrom(attr_value)
    if filler.HasField('value'):
        attr_value = attr_value_pb2.AttrValue()
        attr_value.f = filler.value
        name_attr_list.attr['value'].CopyFrom(attr_value)
    if filler.HasField('min'):
        attr_value = attr_value_pb2.AttrValue()
        attr_value.f = filler.min
        name_attr_list.attr['min'].CopyFrom(attr_value)
    if filler.HasField('max'):
        attr_value = attr_value_pb2.AttrValue()
        attr_value.f = filler.max
        name_attr_list.attr['max'].CopyFrom(attr_value)
    if filler.HasField('mean'):
        attr_value = attr_value_pb2.AttrValue()
        attr_value.f = filler.mean
        name_attr_list.attr['mean'].CopyFrom(attr_value)
    if filler.HasField('std'):
        attr_value = attr_value_pb2.AttrValue()
        attr_value.f = filler.std
        name_attr_list.attr['std'].CopyFrom(attr_value)
    if filler.HasField('sparse'):
        attr_value = attr_value_pb2.AttrValue()
        attr_value.i = filler.sparse
        name_attr_list.attr['sparse'].CopyFrom(attr_value)
    if filler.HasField('variance_norm'):
        attr_value = attr_value_pb2.AttrValue()
        attr_value.i = filler.variance_norm
        name_attr_list.attr['variance_norm'].CopyFrom(attr_value)
    attr.func.CopyFrom(name_attr_list)


def _make_backend_attr(attr_value, backend):
    """ Convert a caffe backend into a string

    A map from number to backend:
      0: LEVELDB
      1: LMDB

    Args:
        attr_value: TensorBoard AttrValue message proto
        backend: a number
    """
    if backend == 0:
        attr_value.s = bytes("LEVELDB", "utf-8")
    elif backend == 1:
        attr_value.s = bytes("LMDB", "utf-8")
    else:
        raise NotImplementedError("backend {} out of range [0, 1]".format(backend))


def _make_shape_attr(attr_value, shape):
    """ Convert a caffe BlobShape message into a TensorBoard TensorShapeProto message

    Args:
        attr_value: TensorBoard AttrValue message proto
        shape: caffe BlobShape message proto
    """
    tensor_shape = tensor_shape_pb2.TensorShapeProto()
    for dim in shape.dim:
        tensor_dim = tensor_shape_pb2.TensorShapeProto.Dim()
        tensor_dim.size = dim
        tensor_shape.dim.extend([tensor_dim])

def _make_engine_attr(attr_value, engine):
    """ Convert a caffe engine number into a string

    The map from number into string is the following:
        0: 'DEFAULT'
        1: 'CAFFE'
        2: 'CUDNN'

    Args:
        attr_value: TensorBoard AttrValue message proto
        engine: a number
    """
    if engine == 0:
        attr_value.s = bytes("DEFAULT", "utf-8")
    elif engine == 1:
        attr_value.s = bytes("CAFFE", "utf-8")
    elif engine == 2:
        attr_value.s = bytes("CUDNN", "utf-8")
    else:
        raise NotImplementedError("engine {} is out of the range: [0,1,2]".format(engine))


def _make_pooling_attr(attr_value, pool):
    """ Convert a caffe pool number into a string

    The map from number into string is the following:
        0: MAX
        1: AVE
        2: STOCHASTIC
    """
    if pool == 0:
        attr_value.s = bytes("MAX", "utf-8")
    elif pool == 1:
        attr_value.s = bytes("AVE", "utf-8")
    elif pool == 2:
        attr_value.s = bytes("STOCHASTIC", "utf-8")
    else:
        raise NotImplementedError("pool {} is out of the range: [0, 1, 2]".format(pool))


def _make_round_attr(attr_value, round_mode):
    """
    The map from number into string is the following:
        0: CEIL
        1: FLOOR
    """
    if round_mode == 0:
        attr_value.s = bytes("CEIL", "utf-8")
    elif round_mode == 1:
        attr_value.s = bytes("FLOOR", "utf-8")
    else:
        raise NotImplementedError("round mode {} is out of range [0, 1]".format(round_mode))


class ParamToAttr(object):
    """ There are many kinds of parameters for caffe layer, so we need to generate
    different parameter handlers for different caffe layers

    The `_param_to_attr_` is a map from caffe layer type to parameter handler
    """
    _param_to_attr_ = {}

    @classmethod
    def register(cls, param_type):
        """ A decorator for registering parameter handler """
        def Wrapper(func):
            cls._param_to_attr_[param_type] = func
            return func

        return Wrapper

    @classmethod
    def HandleParam(cls, param_type, node, caffe_layer):
        handler_fn = cls._param_to_attr_[param_type]
        handler_fn(node, caffe_layer)


@ParamToAttr.register("accuracy_param")
def _accuracy_param_to_attr(node, caffe_layer):
    accuracy_param = caffe_layer.accuracy_param
    if accuracy_param.HasField('top_k'):
        attr_value = attr_value_pb2.AttrValue()
        attr_value.i = accuracy_param.top_k
        node.attr['top_k'].CopyFrom(attr_value)
    if accuracy_param.HasField('axis'):
        attr_value = attr_value_pb2.AttrValue()
        attr_value.i = accuracy_param.axis
        node.attr['axis'].CopyFrom(attr_value)
    if accuracy_param.HasField('ignore_label'):
        attr_value = attr_value_pb2.AttrValue()
        attr_value.i = accuracy_param.ignore_label
        node.attr['ignore_label'].CopyFrom(attr_value)


@ParamToAttr.register("argmax_param")
def _argmax_param_to_attr(node, caffe_layer):
    argmax_param = caffe_layer.argmax_param
    if argmax_param.HasField('out_max_val'):
        attr_value = attr_value_pb2.AttrValue()
        attr_value.b = argmax_param.out_max_val
        node.attr['out_max_val'].CopyFrom(attr_value)
    if argmax_param.HasField('top_k'):
        attr_value = attr_value_pb2.AttrValue()
        attr_value.i = argmax_param.top_k
        node.attr['top_k'].CopyFrom(attr_value)
    if argmax_param.HasField('axis'):
        attr_value = attr_value_pb2.AttrValue()
        attr_value.i = argmax_param.axis
        node.attr['axis'].CopyFrom(attr_value)

@ParamToAttr.register("batch_norm_param")
def _batch_norm_param_to_attr(node, caffe_layer):
    bn_param = caffe_layer.batch_norm_param
    if bn_param.HasField('use_global_stats'):
        attr_value = attr_value_pb2.AttrValue()
        attr_value.b = bn_param.use_global_stats
        node.attr['use_global_stats'].CopyFrom(attr_value)
    if bn_param.HasField('moving_average_fraction'):
        attr_value = attr_value_pb2.AttrValue()
        attr_value.f = bn_param.moving_average_fraction
        node.attr['moving_average_fraction'].CopyFrom(attr_value)
    if bn_param.HasField('eps'):
        attr_value = attr_value_pb2.AttrValue()
        attr_value.f = bn_param.eps
        node.attr['eps'].CopyFrom(attr_value)

@ParamToAttr.register("bias_param")
def _bias_param_to_attr(node, caffe_layer):
    bias_param = caffe_layer.bias_param
    if bias_param.HasField('axis'):
        attr_value = attr_value_pb2.AttrValue()
        attr_value.i = bias_param.axis
        node.attr['axis'].CopyFrom(attr_value)
    if bias_param.HasField('num_axes'):
        attr_value = attr_value_pb2.AttrValue()
        attr_value.i = bias_param.num_axes
        node.attr['num_axes'].CopyFrom(attr_value)

@ParamToAttr.register("clip_param")
def _clip_param_to_attr(node, caffe_layer):
    clip_param = caffe_layer.clip_param
    if clip_param.HasField('min'):
        attr_value = attr_value_pb2.AttrValue()
        attr_value.f = clip_param.min
        node.attr['min'].CopyFrom(attr_value)
    if clip_param.HasField('max'):
        attr_value = attr_value_pb2.AttrValue()
        attr_value.f = clip_param.max
        node.attr['max'].CopyFrom(attr_value)

@ParamToAttr.register("concat_param")
def _concat_param_to_attr(node, caffe_layer):
    concat_param = caffe_layer.concat_param
    if concat_param.HasField('axis'):
        attr_value = attr_value_pb2.AttrValue()
        attr_value.i = concat_param.axis
        node.attr['axis'].CopyFrom(attr_value)
    if concat_param.HasField('concat_dim'):
        attr_value = attr_value_pb2.AttrValue()
        attr_value.i = concat_param.concat_dim
        node.attr['concat_dim'].CopyFrom(attr_value)


@ParamToAttr.register("contrastive_loss_param")
def _contrastive_loss_param_to_attr(node, caffe_layer):
    contrastive_loss_param = caffe_layer.contrastive_loss_param
    raise NotImplementedError("TODO: doesn't support ContrastiveLoss layer")

@ParamToAttr.register("convolution_param")
def _convolution_param_to_attr(node, caffe_layer):
    convolution_param = caffe_layer.convolution_param
    if convolution_param.HasField('num_output'):
        attr_value = attr_value_pb2.AttrValue()
        attr_value.i = convolution_param.num_output
        node.attr['num_output'].CopyFrom(attr_value)
    if convolution_param.HasField('bias_term'):
        attr_value = attr_value_pb2.AttrValue()
        attr_value.b = convolution_param.bias_term
        node.attr['bias_term'].CopyFrom(attr_value)
    if len(convolution_param.pad):
        list_value = attr_value_pb2.AttrValue.ListValue()
        list_value.i.extend(convolution_param.pad)
        attr_value = attr_value_pb2.AttrValue()
        attr_value.list.CopyFrom(list_value)
        node.attr['pad'].CopyFrom(attr_value)
    if len(convolution_param.kernel_size):
        kernel_size_value = attr_value_pb2.AttrValue.ListValue()
        kernel_size_value.i.extend(convolution_param.kernel_size)
        attr_value = attr_value_pb2.AttrValue()
        attr_value.list.CopyFrom(kernel_size_value)
        node.attr['kernel_size'].CopyFrom(attr_value)
    if len(convolution_param.stride):
        stride_value = attr_value_pb2.AttrValue.ListValue()
        stride_value.i.extend(convolution_param.stride)
        attr_value = attr_value_pb2.AttrValue()
        attr_value.list.CopyFrom(stride_value)
        node.attr['stride'].CopyFrom(attr_value)
    if len(convolution_param.dilation):
        dilation_value = attr_value_pb2.AttrValue.ListValue()
        dilation_value.i.extend(convolution_param.dilation)
        attr_value = attr_value_pb2.AttrValue()
        attr_value.list.CopyFrom(dilation_value)
        node.attr['dilation'].CopyFrom(attr_value)
    if convolution_param.HasField('pad_h'):
        attr_value = attr_value_pb2.AttrValue()
        attr_value.i = convolution_param.pad_h
        node.attr['pad_h'].CopyFrom(attr_value)
    if convolution_param.HasField('pad_w'):
        attr_value = attr_value_pb2.AttrValue()
        attr_value.i = convolution_param.pad_w
        node.attr['pad_w'].CopyFrom(attr_value)
    if convolution_param.HasField('kernel_h'):
        attr_value = attr_value_pb2.AttrValue()
        attr_value.i = convolution_param.kernel_h
        node.attr['kernel_h'].CopyFrom(attr_value)
    if convolution_param.HasField('kernel_w'):
        attr_value = attr_value_pb2.AttrValue()
        attr_value.i = convolution_param.kernel_w
        node.attr['kernel_w'].CopyFrom(attr_value)
    if convolution_param.HasField('stride_h'):
        attr_value = attr_value_pb2.AttrValue()
        attr_value.i = convolution_param.stride_h
        node.attr['stride_h'].CopyFrom(attr_value)
    if convolution_param.HasField('stride_w'):
        attr_value = attr_value_pb2.AttrValue()
        attr_value.i = convolution_param.stride_w
        node.attr['stride_w'].CopyFrom(attr_value)
    if convolution_param.HasField('group'):
        attr_value = attr_value_pb2.AttrValue()
        attr_value.i = convolution_param.group
        node.attr['group'].CopyFrom(attr_value)
    if convolution_param.HasField('weight_filler'):
        attr_value = attr_value_pb2.AttrValue()
        _make_filler_attr(attr_value, convolution_param.weight_filler)
        node.attr['weight_filler'].CopyFrom(attr_value)
    if convolution_param.HasField('bias_filler'):
        attr_value = attr_value_pb2.AttrValue()
        _make_filler_attr(attr_value, convolution_param.bias_filler)
        node.attr['bias_filler'].CopyFrom(attr_value)
    if convolution_param.HasField('engine'):
        attr_value = attr_value_pb2.AttrValue()
        attr_value.i = convolution_param.engine
        node.attr['engine'].CopyFrom(attr_value)
    if convolution_param.HasField('axis'):
        attr_value = attr_value_pb2.AttrValue()
        attr_value.i = convolution_param.axis
        node.attr['axis'].CopyFrom(attr_value)


@ParamToAttr.register("crop_param")
def _crop_param_to_attr(node, caffe_layer):
    crop_param = caffe_layer.crop_param
    if crop_param.HasField('axis'):
        attr_value = attr_value_pb2.AttrValue()
        attr_value.i = crop_param.axis
        node.attr['axis'].CopyFrom(attr_value)
    if len(crop_param.offset):
        offset_value = attr_value_pb2.AttrValue.ListValue()
        offset_value.i.extend(crop_param.offset)
        attr_value = attr_value_pb2.AttrValue()
        attr_value.list.CopyFrom(offset_value)
        node.attr['offset'].CopyFrom(attr_value)


@ParamToAttr.register("data_param")
def _data_param_to_attr(node, caffe_layer):
    data_param = caffe_layer.data_param
    if data_param.HasField('source'):
        attr_value = attr_value_pb2.AttrValue()
        attr_value.s = bytes(data_param.source, "utf-8")
        node.attr['source'].CopyFrom(attr_value)
    if data_param.HasField('batch_size'):
        attr_value = attr_value_pb2.AttrValue()
        attr_value.i = data_param.batch_size
        node.attr['batch_size'].CopyFrom(attr_value)
    if data_param.HasField('rand_skip'):
        attr_value = attr_value_pb2.AttrValue()
        attr_value.i = data_param.rand_skip
        node.attr['rand_skip'].CopyFrom(attr_value)
    if data_param.HasField('backend'):
        attr_value = attr_value_pb2.AttrValue()
        _make_backend_attr(attr_value, data_param.backend)
        node.attr['backend'].CopyFrom(attr_value)
    if data_param.HasField('scale'):
        attr_value = attr_value_pb2.AttrValue()
        attr_value.f = data_param.scale
        node.attr['scale'].CopyFrom(attr_value)
    if data_param.HasField('mean_file'):
        attr_value = attr_value_pb2.AttrValue()
        attr_value.s = data_param.mean_file
        node.attr['mean_file'].CopyFrom(attr_value)
    if data_param.HasField('crop_size'):
        attr_value = attr_value_pb2.AttrValue()
        attr_value.i = data_param.crop_size
        node.attr['crop_size'].CopyFrom(attr_value)
    if data_param.HasField('mirror'):
        attr_value = attr_value_pb2.AttrValue()
        attr_value.b = data_param.mirror
        node.attr['mirror'].CopyFrom(attr_value)
    if data_param.HasField('force_encoded_color'):
        attr_value = attr_value_pb2.AttrValue()
        attr_value.b = data_param.force_encoded_color
        node.attr['force_encoded_color'].CopyFrom(attr_value)
    if data_param.HasField('prefetch'):
        attr_value = attr_value_pb2.AttrValue()
        attr_value.i = data_param.prefetch
        node.attr['prefetch'].CopyFrom(attr_value)


@ParamToAttr.register("dropout_param")
def _dropout_param_to_attr(node, caffe_layer):
    dropout_param = caffe_layer.dropout_param
    if dropout_param.HasField('dropout_ratio'):
        attr_value = attr_value_pb2.AttrValue()
        attr_value.f = dropout_param.dropout_ratio
        node.attr['dropout_ratio'].CopyFrom(attr_value)


@ParamToAttr.register("dummy_data_param")
def _dummy_data_param_to_attr(node, caffe_layer):
    raise NotImplementedError("layer type DummyData doesn't be support yet ")

@ParamToAttr.register("eltwise_param")
def _eltwise_param_to_attr(node, caffe_layer):
    eltwise_param = caffe_layer.eltwise_param
    if eltwise_param.HasField('operation'):
        attr_value = attr_value_pb2.AttrValue()
        attr_value.i = eltwise_param.operation
        node.attr['operation'].CopyFrom(attr_value)
    if len(eltwise_param.coeff):
        coeff_value = attr_value_pb2.AttrValue.ListValue()
        coeff_value.f.extend(eltwise_param.coeff)
        attr_value = attr_value_pb2.AttrValue()
        attr_value.list.CopyFrom(coeff_value)
        node.attr['coeff'].CopyFrom(attr_value)
    if eltwise_param.HasField('stable_prod_grad'):
        attr_value = attr_value_pb2.AttrValue()
        attr_value.b = eltwise_param.stable_prod_grad
        node.attr['stable_prod_grad'].CopyFrom(attr_value)


@ParamToAttr.register("elu_param")
def _elu_param_to_attr(node, caffe_layer):
    elu_param = caffe_layer.elu_param
    if elu_param.HasField('alpha'):
        attr_value = attr_value_pb2.AttrValue()
        attr_value.f = elu_param.alpha
        node.attr['alpha'].CopyFrom(attr_value)


@ParamToAttr.register("embed_param")
def _embed_param_to_attr(node, caffe_layer):
    raise NotImplementedError("layer type Embed doesn't support yet ")


@ParamToAttr.register("exp_param")
def _exp_param_to_attr(node, caffe_layer):
    exp_param = caffe_layer.exp_param
    if exp_param.HasField('base'):
        attr_value = attr_value_pb2.AttrValue()
        attr_value.f = exp_param.base
        node.attr['base'].CopyFrom(attr_value)
    if exp_param.HasField('scale'):
        attr_value = attr_value_pb2.AttrValue()
        attr_value.f = exp_param.scale
        node.attr['scale'].CopyFrom(attr_value)
    if exp_param.HasField('shift'):
        attr_value = attr_value_pb2.AttrValue()
        attr_value.f = exp_param.shift
        node.attr['shift'].CopyFrom(attr_value)


@ParamToAttr.register("flatten_param")
def _flatten_param_to_attr(node, caffe_layer):
    flatten_param = caffe_layer.flatten_param
    if flatten_param.HasField('axis'):
        attr_value = attr_value_pb2.AttrValue()
        attr_value.i = flatten_param.axis
        node.attr['axis'].CopyFrom(attr_value)
    if flatten_param.HasField('end_axis'):
        attr_value = attr_value_pb2.AttrValue()
        attr_value.i = flatten_param.end_axis
        node.attr['end_axis'].CopyFrom(attr_value)


@ParamToAttr.register("hdf5_data_param")
def _hdf5_data_param_to_attr(node, caffe_layer):
    raise NotImplementedError("layer type HDF5Data doesn't support yet")


@ParamToAttr.register("hdf5_output_param")
def _hdf5_output_param_to_attr(node, caffe_layer):
    raise NotImplementedError("layer type HDF5Output doesn't support yet")


@ParamToAttr.register("hinge_loss_param")
def _hinge_loss_param_to_attr(node, caffe_layer):
    hinge_loss_param = caffe_layer.hinge_loss_param
    if hinge_loss_param.HasField('norm'):
        attr_value = attr_value_pb2.AttrValue()
        attr_value.i = hinge_loss_param.norm
        node.attr['norm'].CopyFrom(attr_value)


@ParamToAttr.register("image_data_param")
def _image_data_param_to_attr(node, caffe_layer):
    image_data_param = caffe_layer.image_data_param
    if image_data_param.HasField('source'):
        attr_value = attr_value_pb2.AttrValue()
        attr_value.s = bytes(image_data_param.source, "utf-8")
        node.attr['source'].CopyFrom(attr_value)
    if image_data_param.HasField('batch_size'):
        attr_value = attr_value_pb2.AttrValue()
        attr_value.i = image_data_param.batch_size
        node.attr['batch_size'].CopyFrom(attr_value)
    if image_data_param.HasField('rand_skip'):
        attr_value = attr_value_pb2.AttrValue()
        attr_value.i = image_data_param.rand_skip
        node.attr['rand_skip'].CopyFrom(rand_skip)
    if image_data_param.HasField('shuffle'):
        attr_value = attr_value_pb2.AttrValue()
        attr_value.b = image_data_param.shuffle
        node.attr['shuffle'].CopyFrom(attr_value)
    if image_data_param.HasField('new_height'):
        attr_value = attr_value_pb2.AttrValue()
        attr_value.i = image_data_param.new_height
        node.attr['new_height'].CopyFrom(attr_value)
    if image_data_param.HasField('is_color'):
        attr_value = attr_value_pb2.AttrValue()
        attr_value.b = image_data_param.is_color
        node.attr['is_color'].CopyFrom(attr_value)
    if image_data_param.HasField('scale'):
        attr_value = attr_value_pb2.AttrValue()
        attr_value.f = image_data_param.scale
        node.attr['scale'].CopyFrom(attr_value)
    if image_data_param.HasField('mean_file'):
        attr_value = attr_value_pb2.AttrValue()
        attr_value.b = bytes(image_data_param.mean_file, "utf-8")
        node.attr['mean_file'].CopyFrom(attr_value)
    if image_data_param.HasField('crop_size'):
        attr_value = attr_value_pb2.AttrValue()
        attr_value.i = image_data_param.crop_size
        node.attr['crop_size'].CopyFrom(attr_value)
    if image_data_param.HasField('mirror'):
        attr_value = attr_value_pb2.AttrValue()
        attr_value.b = image_data_param.mirror
        node.attr['mirror'].CopyFrom(attr_value)
    if image_data_param.HasField('root_folder'):
        attr_value = attr_value_pb2.AttrValue()
        attr_value.s = bytes(image_data_param.root_folder, "utf-8")
        node.attr['root_folder'].CopyFrom(attr_value)


@ParamToAttr.register("infogain_loss_param")
def _infogain_loss_param_to_attr(node, caffe_layer):
    raise NotImplementedError("layer type InfogainLoss doesn't support yet")


@ParamToAttr.register("inner_product_param")
def _inner_product_param_to_attr(node, caffe_layer):
    inner_product_param = caffe_layer.inner_product_param
    if inner_product_param.HasField('num_output'):
        attr_value = attr_value_pb2.AttrValue()
        attr_value.i = inner_product_param.num_output
        node.attr['num_output'].CopyFrom(attr_value)
    if inner_product_param.HasField('bias_term'):
        attr_value = attr_value_pb2.AttrValue()
        attr_value.b = inner_product_param.bias_term
        node.attr['bias_term'].CopyFrom(attr_value)
    if inner_product_param.HasField('weight_filler'):
        attr_value = attr_value_pb2.AttrValue()
        weight_filler = inner_product_param.weight_filler
        _make_filler_attr(attr_value, weight_filler)
        node.attr['weight_filler'].CopyFrom(attr_value)
    if inner_product_param.HasField('bias_filler'):
        attr_value = attr_value_pb2.AttrValue()
        bias_filler = inner_product_param.bias_filler
        _make_filler_attr(attr_value, bias_filler)
        node.attr['bias_filler'].CopyFrom(attr_value)
    if inner_product_param.HasField('axis'):
        attr_value = attr_value_pb2.AttrValue()
        attr_value.i = inner_product_param.axis
        node.attr['axis'].CopyFrom(attr_value)
    if inner_product_param.HasField('transpose'):
        attr_value = attr_value_pb2.AttrValue()
        attr_value.b = inner_product_param.transpose
        node.attr['transpose'].CopyFrom(attr_value)


@ParamToAttr.register("input_param")
def _input_param_to_attr(node, caffe_layer):
    input_param = caffe_layer.input_param
    if input_param.HasField('shape'):
        attr_value = attr_value_pb2.AttrValue()
        _make_shape_attr(attr_value, input_param.shape)
        node.attr['shape'].CopyFrom(attr_value)


@ParamToAttr.register("log_param")
def _log_param_to_attr(node, caffe_layer):
    log_param = caffe_layer.log_param
    if log_param.HasField('base'):
        attr_value = attr_value_pb2.AttrValue()
        attr_value.f = log_param.base
        node.attr['base'].CopyFrom(attr_value)
    if log_param.HasField('scale'):
        attr_value = attr_value_pb2.AttrValue()
        attr_value.f = log_param.scale
        node.attr['scale'].CopyFrom(attr_value)
    if log_param.HasField('shift'):
        attr_value = attr_value_pb2.AttrValue()
        attr_value.f = log_param.shift
        node.attr['shift'].CopyFrom(attr_value)


@ParamToAttr.register("lrn_param")
def _lrn_param_to_attr(node, caffe_layer):
    lrn_param = caffe_layer.lrn_param
    if lrn_param.HasField('local_size'):
        attr_value = attr_value_pb2.AttrValue()
        attr_value.i = lrn_param.local_size
        node.attr['local_size'].CopyFrom(attr_value)
    if lrn_param.HasField('alpha'):
        attr_value = attr_value_pb2.AttrValue()
        attr_value.f = lrn_param.alpha
        node.attr['alpha'].CopyFrom(attr_value)
    if lrn_param.HasField('beta'):
        attr_value = attr_value_pb2.AttrValue()
        attr_value.f = lrn_param.beta
        node.attr['beta'].CopyFrom(attr_value)
    if lrn_param.HasField('norm_region'):
        attr_value = attr_value_pb2.AttrValue()
        attr_value.i = lrn_param.norm_region
        node.attr['norm_region'].CopyFrom(attr_value)
    if lrn_param.HasField('k'):
        attr_value = attr_value_pb2.AttrValue()
        attr_value.f = lrn_param.k
        node.attr['k'].CopyFrom(attr_value)
    if lrn_param.HasField('engine'):
        attr_value = attr_value_pb2.AttrValue()
        _make_engine_attr(attr_value, lrn_param.engine)
        node.attr['engine'].CopyFrom(attr_value)


@ParamToAttr.register("memory_data_param")
def _memory_data_param_to_attr(node, caffe_layer):
    raise NotImplementedError("layer type MemoryData doesn't support yet")


@ParamToAttr.register("mvn_param")
def _mvn_param_to_attr(node, caffe_layer):
    raise NotImplementedError("layer type MVN doesn't support yet")


@ParamToAttr.register("parameter_param")
def _parameter_param_to_attr(node, caffe_layer):
    raise NotImplementedError("layer type Parameter doesn't support yet")


@ParamToAttr.register("pooling_param")
def _pooling_param_to_attr(node, caffe_layer):
    pooling_param = caffe_layer.pooling_param
    if pooling_param.HasField('pool'):
        attr_value = attr_value_pb2.AttrValue()
        _make_pooling_attr(attr_value, pooling_param.pool)
        node.attr['pool'].CopyFrom(attr_value)
    if pooling_param.HasField('pad'):
        attr_value = attr_value_pb2.AttrValue()
        attr_value.i = pooling_param.pad
        node.attr['pad'].CopyFrom(attr_value)
    if pooling_param.HasField('pad_h'):
        attr_value = attr_value_pb2.AttrValue()
        attr_value.i = pooling_param.pad_h
        node.attr['pad_h'].CopyFrom(attr_value)
    if pooling_param.HasField('pad_w'):
        attr_value = attr_value_pb2.AttrValue()
        attr_value.i = pooling_param.pad_w
        node.attr['pad_w'].CopyFrom(attr_value)
    if pooling_param.HasField('kernel_size'):
        attr_value = attr_value_pb2.AttrValue()
        attr_value.i = pooling_param.kernel_size
        node.attr['kernel_size'].CopyFrom(attr_value)
    if pooling_param.HasField('kernel_h'):
        attr_value = attr_value_pb2.AttrValue()
        attr_value.i = pooling_param.kernel_h
        node.attr['kernel_h'].CopyFrom(attr_value)
    if pooling_param.HasField('kernel_w'):
        attr_value = attr_value_pb2.AttrValue()
        attr_value.i = pooling_param.kernel_w
        node.attr['kernel_w'].CopyFrom(attr_value)
    if pooling_param.HasField('stride'):
        attr_value = attr_value_pb2.AttrValue()
        attr_value.i = pooling_param.stride
        node.attr['stride'].CopyFrom(attr_value)
    if pooling_param.HasField('stride_h'):
        attr_value = attr_value_pb2.AttrValue()
        attr_value.i = pooling_param.stride_h
        node.attr['stride_h'].CopyFrom(attr_value)
    if pooling_param.HasField('stride_w'):
        attr_value = attr_value_pb2.AttrValue()
        attr_value.i = pooling_param.stride_w
        node.attr['stride_w'].CopyFrom(attr_value)
    if pooling_param.HasField('engine'):
        attr_value = attr_value_pb2.AttrValue()
        _make_engine_attr(attr_value, pooling_param.engine)
        node.attr['engine'].CopyFrom(attr_value)
    if pooling_param.HasField('global_pooling'):
        attr_value = attr_value_pb2.AttrValue()
        attr_value.b = pooling_param.global_pooling
        node.attr['global_pooling'].CopyFrom(attr_value)
    if pooling_param.HasField('round_mode'):
        attr_value = attr_value_pb2.AttrValue()
        _make_round_attr(attr_value, pooling_param.round_mode)
        node.attr['round_mode'].CopyFrom(attr_value)


@ParamToAttr.register("power_param")
def _power_param_to_attr(node, caffe_layer):
    power_param = caffe_layer.power_param
    if power_param.HasField('power'):
        attr_value = attr_value_pb2.AttrValue()
        attr_value.f = power_param.power
        node.attr['power'].CopyFrom(attr_value)
    if power_param.HasField('scale'):
        attr_value = attr_value_pb2.AttrValue()
        attr_value.f = power_param.scale
        node.attr['scale'].CopyFrom(attr_value)


@ParamToAttr.register("prelu_param")
def _prelu_param_to_attr(node, caffe_layer):
    raise NotImplementedError("prelu_param doesn't support yet")


@ParamToAttr.register("python_param")
def _python_param_to_attr(node, caffe_layer):
    raise NotImplementedError("python_param doesn't support yet")


@ParamToAttr.register("recurrent_param")
def _recurrent_param_to_attr(node, caffe_layer):
    raise NotImplementedError("recurrent_param doesn't support yet")


@ParamToAttr.register("reduction_param")
def _reduction_param_to_attr(node, caffe_layer):
    raise NotImplementedError("reduction_param doesn't support yet")


@ParamToAttr.register("relu_param")
def _relu_param_to_attr(node, caffe_layer):
    relu_param = caffe_layer.relu_param
    if relu_param.HasField('negative_slope'):
        attr_value = attr_value_pb2.AttrValue()
        attr_value.f = relu_param.negative_slope
        node.attr['negative_slope'].CopyFrom(attr_value)
    if relu_param.HasField('engine'):
        attr_value = attr_value_pb2.AttrValue()
        _make_engine_attr(attr_value, relu_param.engine)
        node.attr['engine'].CopyFrom(attr_value)


@ParamToAttr.register("reshape_param")
def _reshape_param_to_attr(node, caffe_layer):
    reshape_param = caffe_layer.reshape_param
    if reshape_param.HasField('shape'):
        attr_value = attr_value_pb2.AttrValue()
        _make_shape_attr(attr_value, reshape_param.shape)
        node.attr['shape'].CopyFrom(attr_value)
    if reshape_param.HasField('axis'):
        attr_value = attr_value_pb2.AttrValue()
        attr_value.i = reshape_param.axis
        node.attr['axis'].CopyFrom(attr_value)
    if reshape_param.HasField('num_axes'):
        attr_value = attr_value_pb2.AttrValue()
        attr_value.i = reshape_param.num_axes
        node.attr['num_axes'].CopyFrom(attr_value)


@ParamToAttr.register("scale_param")
def _scale_param_to_attr(node, caffe_layer):
    scale_param = caffe_layer.scale_param
    if scale_param.HasField('axis'):
        attr_value = attr_value_pb2.AttrValue()
        attr_value.i = scale_param.axis
        node.attr['axis'].CopyFrom(attr_value)
    if scale_param.HasField('num_axes'):
        attr_value = attr_value_pb2.AttrValue()
        attr_value.i = scale_param.num_axes
        node.attr['num_axes'].CopyFrom(attr_value)
    if scale_param.HasField('filler'):
        attr_value = attr_value_pb2.AttrValue()
        _make_filler_attr(attr_value, scale_param.filler)
        node.attr['filler'].CopyFrom(attr_value)
    if scale_param.HasField('bias_term'):
        attr_value = attr_value_pb2.AttrValue()
        attr_value.b = scale_param.bias_term
        node.attr['bias_term'].CopyFrom(attr_value)
    if scale_param.HasField('bias_filler'):
        attr_value = attr_value_pb2.AttrValue()
        _make_filler_attr(attr_value, scale_param.bias_filler)
        node.attr['bias_filler'].CopyFrom(attr_value)


@ParamToAttr.register("sigmoid_param")
def _sigmoid_param_to_attr(node, caffe_layer):
    sigmoid_param = caffe_layer.sigmoid_param
    if sigmoid_param.HasField('engine'):
        attr_value = attr_value_pb2.AttrValue()
        _make_engine_attr(attr_value, sigmoid_param.engine)


@ParamToAttr.register("softmax_param")
def _softmax_param_to_attr(node, caffe_layer):
    softmax_param = caffe_layer.softmax_param
    if softmax_param.HasField('engine'):
        attr_value = attr_value_pb2.AttrValue()
        _make_engine_attr(attr_value, softmax_param.engine)
    if softmax_param.HasField('axis'):
        attr_value = attr_value_pb2.AttrValue()
        attr_value.i = softmax_param.axis
        node.attr['axis'].CopyFrom(attr_value)


@ParamToAttr.register("spp_param")
def _spp_param_to_attr(node, caffe_layer):
    spp_param = caffe_layer.spp_param
    if spp_param.HasField('pyramid_height'):
        attr_value = attr_value_pb2.AttrValue()
        attr_value.i = spp_param.pyramid_height
        node.attr['pyramid_height'].CopyFrom(attr_value)
    if spp_param.HasField('pool'):
        attr_value = attr_value_pb2.AttrValue()
        _make_pooling_attr(attr_value, spp_param.pool)
        node.attr['pool'].CopyFrom(attr_value)
    if spp_param.HasField('engine'):
        attr_value = attr_value_pb2.AttrValue()
        _make_engine_attr(attr_value, spp_param.engine)
        node.attr['engine'].CopyFrom(attr_value)


@ParamToAttr.register("slice_param")
def _slice_param_to_attr(node, caffe_layer):
    slice_param = caffe_layer.slice_param
    if slice_param.HasField('axis'):
        attr_value = attr_value_pb2.AttrValue()
        attr_value.i = slice_param.axis
        node.attr['axis'].CopyFrom(attr_value)
    if slice_param.HasField('slice_point'):
        attr_value = attr_value_pb2.AttrValue()
        attr_value.i = slice_param.slice_point
        node.attr['slice_point'].CopyFrom(attr_value)
    if slice_param.HasField('slice_dim'):
        attr_value = attr_value_pb2.AttrValue()
        attr_value.i = slice_param.slice_dim
        node.attr['slice_dim'].CopyFrom(attr_value)


@ParamToAttr.register("swish_param")
def _swish_param_to_attr(node, caffe_layer):
    swish_param = caffe_layer.swish_param
    if swish_param.HasField('beta'):
        attr_value = attr_value_pb2.AttrValue()
        attr_value.f = swish_param.beta
        node.attr['beta'].CopyFrom(attr_value)


@ParamToAttr.register("tanh_param")
def _tanh_param_to_attr(node, caffe_layer):
    tanh_param = caffe_layer.tanh_param
    if tanh_param.HasField('engine'):
        attr_value = attr_value_pb2.AttrValue()
        _make_engine_attr(attr_value, tanh_param.engine)


@ParamToAttr.register("threshold_param")
def _threshold_param_to_attr(node, caffe_layer):
    threshold_param = caffe_layer.threshold_param
    if threshold_param.HasField('threshold'):
        attr_value = attr_value_pb2.AttrValue()
        attr_value.f = threshold_param.threshold
        node.attr['threshold'].CopyFrom(attr_value)


@ParamToAttr.register("tile_param")
def _tile_param_to_attr(node, caffe_layer):
    tile_param = caffe_layer.tile_param
    if tile_param.HasField('axis'):
        attr_value = attr_value_pb2.AttrValue()
        attr_value.i = tile_param.axis
        node.attr['axis'].CopyFrom(attr_value)
    if tile_param.HasField('tiles'):
        attr_value = attr_value_pb2.AttrValue()
        attr_value.i = tile_param.tiles
        node.attr['tiles'].CopyFrom(attr_value)


@ParamToAttr.register("window_data_param")
def _window_data_param_to_attr(node, caffe_layer):
    window_data_param = caffe_layer.window_data_param
    if window_data_param.HasField('source'):
        attr_value = attr_value_pb2.AttrValue()
        attr_value.s = bytes(window_data_param.source, "utf-8")
        node.attr['source'].CopyFrom(attr_value)
    if window_data_param.HasField('scale'):
        attr_value = attr_value_pb2.AttrValue()
        attr_value.f = window_data_param.scale
        node.attr['scale'].CopyFrom(attr_value)
    if window_data_param.HasField('mean_file'):
        attr_value = attr_value_pb2.AttrValue()
        attr_value.s = bytes(window_data_param.mean_file, "utf-8")
    if window_data_param.HasField('batch_size'):
        attr_value = attr_value_pb2.AttrValue()
        attr_value.i = window_data_param.batch_size
        node.attr['batch_size'].CopyFrom(attr_value)
    if window_data_param.HasField('crop_size'):
        attr_value = attr_value_pb2.AttrValue()
        attr_value.i = window_data_param.crop_size
        node.attr['crop_size'].CopyFrom(attr_value)
    if window_data_param.HasField('mirror'):
        attr_value = attr_value_pb2.AttrValue()
        attr_value.b = window_data_param.mirror
        node.attr['mirror'].CopyFrom(attr_value)
    if window_data_param.HasField('fg_threshold'):
        attr_value = attr_value_pb2.AttrValue()
        attr_value.f = window_data_param.fg_threshold
        node.attr['fg_threshold'].CopyFrom(attr_value)
    if window_data_param.HasField('bg_threshold'):
        attr_value = attr_value_pb2.AttrValue()
        attr_value.f = window_data_param.bg_threshold
        node.attr['bg_threshold'].CopyFrom(attr_value)
    if window_data_param.HasField('fg_fraction'):
        attr_value = attr_value_pb2.AttrValue()
        attr_value.f = window_data_param.fg_fraction
        node.attr['fg_fraction'].CopyFrom(attr_value)
    if window_data_param.HasField('context_pad'):
        attr_value = attr_value_pb2.AttrValue()
        attr_value.i = window_data_param.context_pad
        node.attr['context_pad'].CopyFrom(attr_value)
    if window_data_param.HasField('crop_mode'):
        attr_value = attr_value_pb2.AttrValue()
        attr_value.s = bytes(window_data_param.crop_mode, "utf-8")
        node.attr['crop_mode'].CopyFrom(attr_value)
    if window_data_param.HasField('cache_images'):
        attr_value = attr_value_pb2.AttrValue()
        attr_value.b = window_data_param.cache_images
        node.attr['cache_images'].CopyFrom(attr_value)
    if window_data_param.HasField('root_folder'):
        attr_value = attr_value_pb2.AttrValue()
        attr_value.s = bytes(window_data_param.root_folder, "utf-8")
        node.attr['root_folder'].CopyFrom(attr_value)


@ParamToAttr.register('transform_param')
def _transform_param_to_attr(node, caffe_layer):
    transform_param = caffe_layer.transform_param
    if transform_param.HasField('scale'):
        attr_value = attr_value_pb2.AttrValue()
        attr_value.f = transform_param.scale
        node.attr['scale'].CopyFrom(attr_value)
    if transform_param.HasField('mirror'):
        attr_value = attr_value_pb2.AttrValue()
        attr_value.b = transform_param.mirror
        node.attr['mirror'].CopyFrom(attr_value)
    if transform_param.HasField('crop_size'):
        attr_value = attr_value_pb2.AttrValue()
        attr_value.i = transform_param.crop_size
        node.attr['crop_size'].CopyFrom(attr_value)
    if transform_param.HasField('mean_file'):
        attr_value = attr_value_pb2.AttrValue()
        attr_value.s = bytes(transform_param.mean_file, "utf-8")
        node.attr['mean_file'].CopyFrom(attr_value)
    if len(transform_param.mean_value):
        mean_value_list = attr_value_pb2.AttrValue.ListValue()
        mean_value_list.f.extend(transform_param.mean_value)
        attr_value = attr_value_pb2.AttrValue()
        attr_value.list.CopyFrom(mean_value_list)
        node.attr['mean_value'].CopyFrom(attr_value)
    if transform_param.HasField('force_color'):
        attr_value = attr_value_pb2.AttrValue()
        attr_value.b = transform_param.force_color
        node.attr['force_color'].CopyFrom(attr_value)
    if transform_param.HasField('force_gray'):
        attr_value = attr_value_pb2.AttrValue()
        attr_value.b = transform_param.force_gray
        node.attr['force_gray'].CopyFrom(attr_value)


@ParamToAttr.register("shuffle_channel_param")
def _shuffle_channel_param_to_attr(node, caffe_layer):
    shuffle_channel_param = caffe_layer.shuffle_channel_param
    if shuffle_channel_param.HasField('group'):
        attr_value = attr_value_pb2.AttrValue()
        attr_value.i = shuffle_channel_param.group
        node.attr['group'].CopyFrom(attr_value)


class CaffeGraph(tbgraph_base.TBGraph):
    def __init__(self, caffemodel, model_type='txt'):
        super(CaffeGraph, self).__init__()
        self._caffe_model = caffe_pb2.NetParameter()
        if model_type == "txt":
            with open(caffemodel, "r") as caffe_stream:
                text_format.Parse(caffe_stream.read(), self._caffe_model)
        elif model_type == "pb":
            with open(caffemodel, "rb") as caffe_stream:
                self._caffe_model.ParseFromString(caffe_stream.read())
        else:
            raise NotImplementedError("model type {} doesn't support yet")

        # a map from blob to version, which is used for building SSA format
        # graph
        self._blobs_version = {}
        # a map from node name to node definition
        self.nodes = {}

    def convert_to_nodes(self, caffe_layer):
        """ Convert a caffe layer into TB nodes

        Because a caffe layer may generate multiple output. Therefore, it produces
        multiple output nodes

        Arg:
            caffe_layer: a caffe layer message proto
        """
        new_node = node_def_pb2.NodeDef()
        new_node.op = caffe_layer.type
        for caffe_bottom in caffe_layer.bottom:
            if caffe_bottom not in self._blobs_version:
                # this is an weight or input data blob
                # so we want to produce an node for this blob
                in_node = node_def_pb2.NodeDef()
                self._blobs_version[caffe_bottom] = 0
                in_node.name = '{}_{}'.format(caffe_bottom, 0)
                self.nodes[in_node.name] = in_node
                self._tb_graph.node.extend([in_node])
            new_node.input.append('{}_{}'.format(caffe_bottom, self._blobs_version[caffe_bottom]))

        if caffe_layer.top[0] in self._blobs_version:
            # there is a previous blob and we need to increase blob version
            self._blobs_version[caffe_layer.top[0]] += 1
        else:
            self._blobs_version[caffe_layer.top[0]] = 0
        new_node.name = "{}_{}".format(caffe_layer.top[0], self._blobs_version[caffe_layer.top[0]])

        # handle layer parameter and convert them into attributes of the node
        for param_type in ParamToAttr._param_to_attr_:
            if caffe_layer.HasField(param_type):
                ParamToAttr.HandleParam(param_type, new_node, caffe_layer)
        self.nodes[new_node.name] = new_node
        self._tb_graph.node.extend([new_node])

        for caffe_top in caffe_layer.top[1:]:
            if caffe_top in self._blobs_version:
                self._blobs_version[caffe_top] += 1
            else:
                self._blobs_version[caffe_top] = 0
            out_node = node_def_pb2.NodeDef()
            out_node.name = '{}_{}'.format(caffe_top, self._blobs_version[caffe_top])
            out_node.input.append(new_node.name)
            self.nodes[out_node.name] = out_node
            self._tb_graph.node.extend([out_node])

    def ConvertNet(self):
        for caffe_layer in self._caffe_model.layer:
            if len(caffe_layer.include):
                if caffe_layer.include[0].phase == caffe_pb2.TEST:
                    # skip TEST layers
                    continue
            self.convert_to_nodes(caffe_layer)
