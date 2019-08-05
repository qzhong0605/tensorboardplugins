package(default_visibility = ["//tensorboard:internal"])

load("//tensorboard/defs:defs.bzl", "tensorboard_webcomponent_library")
load("//tensorboard/defs:web.bzl", "tf_web_library")

licenses(["notice"])  # Apache 2.0

tf_web_library(
    name = "tf_convert_info",
    srcs = [
        "tf-convert-info.html",
        "tf-convert-node-info.html",
        "tf-convert-node-list-item.html",
    ],
    path = "/tf-convert-info",
    deps = [
        ":tf_convert_icon",
        "//tensorboard/components/tf_dashboard_common",
        "//tensorboard/components/tf_imports:polymer",
        "//tensorboard/plugins/convert/tf_convert_common",
        "@org_polymer_iron_collapse",
        "@org_polymer_iron_list",
        "@org_polymer_paper_button",
        "@org_polymer_paper_icon_button",
        "@org_polymer_paper_item",
        "@org_polymer_paper_material",
        "@org_polymer_paper_slider",
        "@org_polymer_paper_spinner",
    ],
)

tf_web_library(
    name = "tf_convert_icon",
    srcs = [
        "tf-convert-icon.html",
    ],
    path = "/tf-convert-info",
    deps = [
        "//tensorboard/components/tf_dashboard_common",
        "//tensorboard/components/tf_imports:polymer",
    ],
)

tensorboard_webcomponent_library(
    name = "legacy",
    srcs = [":tf_convert_info"],
    destdir = "tf-convert-info",
    deps = [
        "//tensorboard/components/tf_dashboard_common:legacy",
        "//tensorboard/plugins/convert/tf_convert_common:legacy",
        "//third_party/javascript/polymer/v1/iron-collapse:lib",
        "//third_party/javascript/polymer/v1/iron-list:lib",
        "//third_party/javascript/polymer/v1/paper-icon-button:lib",
        "//third_party/javascript/polymer/v1/paper-item:lib",
        "//third_party/javascript/polymer/v1/paper-material:paper-material-html",
        "//third_party/javascript/polymer/v1/paper-slider:lib",
        "//third_party/javascript/polymer/v1/paper-spinner:lib",
        "//third_party/javascript/polymer/v1/polymer:lib",
    ],
)

tensorboard_webcomponent_library(
    name = "tf_convert_icon_legacy",
    srcs = [":tf_convert_icon"],
    destdir = "tf-convert-info",
    deps = [
        "//tensorboard/components/tf_dashboard_common:legacy",
        "//third_party/javascript/polymer/v1/iron-collapse:lib",
        "//third_party/javascript/polymer/v1/iron-list:lib",
        "//third_party/javascript/polymer/v1/paper-icon-button:lib",
        "//third_party/javascript/polymer/v1/paper-item:lib",
        "//third_party/javascript/polymer/v1/paper-slider:lib",
        "//third_party/javascript/polymer/v1/paper-spinner:lib",
        "//third_party/javascript/polymer/v1/polymer:lib",
    ],
)
