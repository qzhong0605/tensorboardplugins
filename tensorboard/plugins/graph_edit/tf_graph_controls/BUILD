package(default_visibility = ["//tensorboard:internal"])

load("//tensorboard/defs:defs.bzl", "tensorboard_webcomponent_library")
load("//tensorboard/defs:web.bzl", "tf_web_library")

licenses(["notice"])  # Apache 2.0

tf_web_library(
    name = "tf_graph_controls",
    srcs = [
        "tf-graph-controls.html",
        "tf-graph-controls.ts",
    ],
    path = "/tf-graph-controls",
    deps = [
        "//tensorboard/components/tf_dashboard_common",
        "//tensorboard/components/tf_imports:polymer",
        "//tensorboard/plugins/graph/tf_graph_common",
        "//tensorboard/plugins/graph/tf_graph_node_search",
        "//tensorboard/plugins/graph/tf_graph_add_op",
        "//tensorboard/plugins/graph/tf_graph_add_edge",
        "//tensorboard/plugins/graph/tf_graph_edit_op",
        "@org_polymer_iron_icon",
        "@org_polymer_iron_ajax",
        "@org_polymer_paper_button",
        "@org_polymer_paper_dropdown_menu",
        "@org_polymer_paper_icon_button",
        "@org_polymer_paper_item",
        "@org_polymer_paper_menu",
        "@org_polymer_paper_radio_group",
        "@org_polymer_paper_toggle_button",
        "@org_polymer_paper_tooltip",
        "@org_polymer_paper_input",
        "@org_polymer_paper_dialog",
    ],
)

tensorboard_webcomponent_library(
    name = "legacy",
    srcs = [":tf_graph_controls"],
    destdir = "tf-graph-controls",
    deps = [
        "//tensorboard/components/tf_dashboard_common:legacy",
        "//tensorboard/plugins/graph/tf_graph_common:legacy",
        "//tensorboard/plugins/graph/tf_graph_node_search:legacy",
        "//tensorboard/plugins/graph/tf_graph_add_op:legacy",
        "//tensorboard/plugins/graph/tf_graph_add_edge:legacy",
        "//tensorboard/plugins/graph/tf_graph_edit_op:legacy",
        "//third_party/javascript/polymer/v1/paper-button:lib",
        "//third_party/javascript/polymer/v1/paper-dropdown-menu:lib",
        "//third_party/javascript/polymer/v1/paper-icon-button:lib",
        "//third_party/javascript/polymer/v1/paper-item:lib",
        "//third_party/javascript/polymer/v1/paper-menu:lib",
        "//third_party/javascript/polymer/v1/paper-input:lib",
        "//third_party/javascript/polymer/v1/paper-radio-group:lib",
        "//third_party/javascript/polymer/v1/paper-toggle-button:lib",
        "//third_party/javascript/polymer/v1/paper-tooltip:lib",
        "//third_party/javascript/polymer/v1/paper-dialog:lib",
        "//third_party/javascript/polymer/v1/polymer:lib",
    ],
)
