package(default_visibility = ["//tensorboard:internal"])

load("//tensorboard/defs:defs.bzl", "tensorboard_webcomponent_library")
load("//tensorboard/defs:web.bzl", "tf_web_library")

licenses(["notice"])  # Apache 2.0

tf_web_library(
    name = "tf_graph_edit_dashboard",
    srcs = ["tf-graph-edit-dashboard.html"],
    path = "/tf-graph-edit-dashboard",
    visibility = ["//visibility:public"],
    deps = [
        "//tensorboard/components/tf_backend",
        "//tensorboard/components/tf_dashboard_common",
        "//tensorboard/components/tf_imports:polymer",
        "//tensorboard/components/tf_storage",
        "//tensorboard/components/tf_tensorboard:registry",
        "//tensorboard/components/vz_sorting",
        "//tensorboard/plugins/graph_edit/tf_graph_edit",
        "//tensorboard/plugins/graph_edit/tf_graph_edit_board",
        "//tensorboard/plugins/graph_edit/tf_graph_edit_controls",
        "//tensorboard/plugins/graph_edit/tf_graph_edit_loader:tf_graph_edit_dashboard_loader",
        "@org_polymer_paper_input",
        "@org_polymer_paper_button",
        "@org_polymer_paper_menu",
        "@org_polymer_paper_dropdown_menu",
        "@org_polymer_iron_icon",
        "@org_polymer_paper_item",
    ],
)

tensorboard_webcomponent_library(
    name = "legacy",
    srcs = [":tf_graph_edit_dashboard"],
    destdir = "tf-graph-edit-dashboard",
    deps = [
        "//tensorboard/components/tf_backend:legacy",
        "//tensorboard/components/tf_dashboard_common:legacy",
        "//tensorboard/components/vz_sorting:legacy",
        "//tensorboard/plugins/graph_edit/tf_graph_edit:legacy",
        "//tensorboard/plugins/graph_edit/tf_graph_edit_board:legacy",
        "//tensorboard/plugins/graph_edit/tf_graph_edit_controls:legacy",
        "//tensorboard/plugins/graph_edit/tf_graph_edit_loader:legacy_dashboard_loader",
        "//third_party/javascript/polymer/v1/polymer:lib",
        "//third_party/javascript/polymer/v1/paper-input:lib",
        "//third_party/javascript/polymer/v1/paper-button:lib",
        "//third_party/javascript/polymer/v1/paper-dropdown-menu:lib",
        "//third_party/javascript/polymer/v1/paper-menu:lib",
        "//third_party/javascript/polymer/v1/paper-item:lib",
    ],
)
