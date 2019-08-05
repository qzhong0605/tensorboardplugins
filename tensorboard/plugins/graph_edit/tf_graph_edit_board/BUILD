package(default_visibility = ["//tensorboard:internal"])

load("//tensorboard/defs:defs.bzl", "tensorboard_webcomponent_library")
load("//tensorboard/defs:web.bzl", "tf_web_library")

licenses(["notice"])  # Apache 2.0

tf_web_library(
    name = "tf_graph_edit_board",
    srcs = ["tf-graph-edit-board.html"],
    path = "/tf-graph-edit-board",
    deps = [
        "//tensorboard/plugins/graph_edit/tf_graph_edit",
        "//tensorboard/plugins/graph_edit/tf_graph_edit_common",
        "//tensorboard/components/tf_imports:polymer",
        "@org_polymer_paper_progress",
    ],
)

tensorboard_webcomponent_library(
    name = "legacy",
    srcs = [":tf_graph_edit_board"],
    destdir = "tf-graph-edit-board",
    deps = [
        "//tensorboard/plugins/graph_edit/tf_graph_edit:legacy",
        "//tensorboard/plugins/graph_edit/tf_graph_edit_common:legacy",
        "//third_party/javascript/polymer/v1/paper-progress:lib",
        "//third_party/javascript/polymer/v1/polymer:lib",
    ],
)

