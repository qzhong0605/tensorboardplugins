package(default_visibility = ["//tensorboard:internal"])

load("//tensorboard/defs:defs.bzl", "tensorboard_webcomponent_library")
load("//tensorboard/defs:web.bzl", "tf_web_library")

licenses(["notice"])  # Apache 2.0

tf_web_library(
    name = "tf_graph_edit_loader",
    srcs = [
        "tf-graph-edit-loader.html",
        "tf-graph-edit-loader.ts",
    ],
    path = "/tf-graph-edit-loader",
    deps = [
        "//tensorboard/plugins/graph_edit/tf_graph_edit_common",
        "//tensorboard/components/tf_imports:polymer",
    ],
)

tensorboard_webcomponent_library(
    name = "legacy",
    srcs = [":tf_graph_edit_loader"],
    destdir = "tf-graph-edit-loader",
    deps = [
        "//tensorboard/plugins/graph_edit/tf_graph_edit_common:legacy",
        "//third_party/javascript/polymer/v1/polymer:lib",
    ],
)

tf_web_library(
    name = "tf_graph_edit_dashboard_loader",
    srcs = [
        "tf-graph-edit-dashboard-loader.html",
        "tf-graph-edit-dashboard-loader.ts",
    ],
    path = "/tf-graph-edit-loader",
    deps = [
        "//tensorboard/components/tf_backend",
        "//tensorboard/plugins/graph_edit/tf_graph_edit_common",
        "//tensorboard/plugins/graph_edit/tf_graph_edit_controls",
        "//tensorboard/components/tf_imports:polymer",
    ],
)

tensorboard_webcomponent_library(
    name = "legacy_dashboard_loader",
    srcs = [":tf_graph_edit_dashboard_loader"],
    destdir = "tf-graph-edit-loader",
    deps = [
        "//tensorboard/components/tf_backend:legacy",
        "//tensorboard/plugins/graph_edit/tf_graph_edit_common:legacy",
        "//tensorboard/plugins/graph_edit/tf_graph_edit_controls:legacy",
        "//third_party/javascript/polymer/v1/polymer:lib",
    ],
)
