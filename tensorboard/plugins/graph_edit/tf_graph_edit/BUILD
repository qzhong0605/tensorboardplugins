package(default_visibility = ["//tensorboard:internal"])

load("//tensorboard/defs:defs.bzl", "tensorboard_webcomponent_library")
load("//tensorboard/defs:web.bzl", "tf_web_library")

licenses(["notice"])  # Apache 2.0

tf_web_library(
    name = "tf_graph_edit",
    srcs = [
        "tf-graph-edit.html",
        "tf-graph-edit-minimap.html",
        "tf-graph-edit-scene.html",
    ],
    path = "/tf-graph-edit",
    deps = [
        "//tensorboard/components/tf_dashboard_common",
        "//tensorboard/plugins/graph_edit/tf_graph_edit_common",
        "//tensorboard/components/tf_imports:polymer",
        "@org_polymer_iron_flex_layout",
        "@org_polymer_iron_icons",
        "@org_polymer_paper_button",
        "@org_polymer_paper_dropdown_menu",
        "@org_polymer_paper_input",
        "@org_polymer_paper_menu",
        "@org_polymer_paper_radio_group",
        "@org_polymer_paper_toggle_button",
        "@org_polymer_paper_tooltip",
    ],
)

tensorboard_webcomponent_library(
    name = "legacy",
    srcs = [":tf_graph_edit"],
    destdir = "tf-graph-edit",
    deps = [
        "//tensorboard/components/tf_dashboard_common:legacy",
        "//tensorboard/plugins/graph_edit/tf_graph_edit_common:legacy",
        "//third_party/javascript/polymer/v1/iron-flex-layout:lib",
        "//third_party/javascript/polymer/v1/iron-icons:lib",
        "//third_party/javascript/polymer/v1/paper-button:lib",
        "//third_party/javascript/polymer/v1/paper-dropdown-menu:lib",
        "//third_party/javascript/polymer/v1/paper-input:lib",
        "//third_party/javascript/polymer/v1/paper-menu:lib",
        "//third_party/javascript/polymer/v1/paper-radio-group:lib",
        "//third_party/javascript/polymer/v1/paper-toggle-button:lib",
        "//third_party/javascript/polymer/v1/paper-tooltip:lib",
        "//third_party/javascript/polymer/v1/polymer:lib",
    ],
)

