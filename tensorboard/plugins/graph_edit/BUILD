# Description:
# TensorBoard plugin for graphs

package(default_visibility = ["//tensorboard:internal"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

## GraphEdit Plugin ##
py_library(
    name = "graphedit_plugin",
    srcs = ["graphedit_plugin.py"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":keras_util",
        ":graph_util",
        ":caffe2_util",
        ":caffe_util",
        ":onnx_util",
        ":onnx_write_util",
        ":torch_util",
        "//tensorboard/backend:http_util",
        "//tensorboard/backend:process_graph",
        "//tensorboard/backend/event_processing:event_accumulator",
        "//tensorboard/compat:tensorflow",
        "//tensorboard/plugins:base_plugin",
        "@com_google_protobuf//:protobuf_python",
        "@org_pocoo_werkzeug",
        "@org_pythonhosted_six",
    ],
)

py_library(
    name = "keras_util",
    srcs = ["keras_util.py"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:private"],
    deps = [
        "//tensorboard/compat/proto:protos_all_py_pb2",
        "//tensorboard/compat/tensorflow_stub",
    ],
)

py_library(
    name = "graph_util",
    srcs = ["graph_util.py"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:private"],
    deps = [
        "//tensorboard/compat/proto:protos_all_py_pb2",
    ],
)

# base tensorboard graph representation 
py_library(
    name = "tb_base",
    srcs = ["tbgraph_base.py"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:private"],
    deps = [
        "//tensorboard/compat/proto:protos_all_py_pb2",
    ],
)

# convert caffe2 model to tb ir
py_library(
    name = "caffe2_util",
    srcs = ["c2graph_util.py"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:private"],
    deps = [
        ":tb_base",
    ],
)

# convert caffe model to tb ir 
py_library(
    name = "caffe_util",
    srcs = ["caffe_util.py"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:private"],
    deps = [
        ":tb_base",
    ],
)

# convert onnx model to tb ir 
py_library(
    name = "onnx_util",
    srcs = ["onnx_util.py"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:private"],
    deps = [
        ":tb_base",
    ],
)

# save tensorboard model in onnx ir
py_library(
    name = "onnx_write_util",
    srcs = ["onnx_write_util.py"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:private"],
    deps = [
        ":tb_base",
    ],
)

# convert pytorch model to tb ir 
py_library(
    name = "torch_util",
    srcs = ["torch_util.py"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:private"],
    deps = [
        ":tb_base",
    ],
)

