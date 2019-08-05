# Description:
# TensorBoard plugin for debugDB

package(default_visibility = ["//tensorboard:internal"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

## debugDB Plugin ##
py_library(
    name = "debugDB_plugin",
    srcs = ["debugDB_plugin.py"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":keras_util",
        ":graph_util",
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
