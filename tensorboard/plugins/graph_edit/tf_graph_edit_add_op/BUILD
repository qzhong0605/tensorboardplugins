package(default_visibility = ["//tensorboard:internal"])

load("//tensorboard/defs:defs.bzl", "tensorboard_webcomponent_library")
load("//tensorboard/defs:web.bzl", "tf_web_library")

licenses(["notice"])  # Apache 2.0

tf_web_library(
    name = "tf_graph_edit_add_op",
    srcs = ["tf-graph-edit-add-op.html"],
    path = "/tf-graph-edit-add-op",
    deps = [
        "//tensorboard/plugins/graph_edit/tf_graph_edit",
        "//tensorboard/plugins/graph_edit/tf_graph_edit_common",
        "//tensorboard/components/tf_imports:polymer",
        "//tensorboard/components/tf_imports:d3",
        "//tensorboard/components/tf_imports:katex-js",
        "@org_polymer_paper_progress",
        "@org_polymer_paper_input",
        "@org_polymer_paper_dropdown_menu",
        "@org_polymer_paper_icon_button",
        "@org_polymer_paper_menu",
        "@org_polymer_paper_item",
        "@org_polymer_iron_list",
        "@org_polymer_paper_button",
        "@org_polymer_paper_listbox",
        "@org_polymer_paper_menu_button",
    ],
)

tensorboard_webcomponent_library(
    name = "legacy",
    srcs = [":tf_graph_edit_add_op"],
    destdir = "tf-graph-edit-add-op",
    deps = [
        "//tensorboard/plugins/graph_edit/tf_graph_edit:legacy",
        "//tensorboard/plugins/graph_edit/tf_graph_edit_common:legacy",
        "//third_party/javascript/polymer/v1/paper-progress:lib",
        "//third_party/javascript/polymer/v1/polymer:lib",
        "//third_party/javascript/polymer/v1/paper-input:lib",
        "//third_party/javascript/polymer/v1/paper-dropdown-menu:lib",
        "//third_party/javascript/polymer/v1/paper-icon-button:lib",
        "//third_party/javascript/polymer/v1/paper-item:lib",
        "//third_party/javascript/polymer/v1/paper-menu:lib",
        "//third_party/javascript/polymer/v1/iron-list:lib",
        "//third_party/javascript/polymer/v1/paper-button:lib",
        "//third_party/javascript/polymer/v1/paper-menu-button:lib",
    ],
)

