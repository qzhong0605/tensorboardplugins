package(default_visibility = ["//tensorboard:internal"])

load("//tensorboard/defs:defs.bzl", "tensorboard_webcomponent_library")
load("//tensorboard/defs:web.bzl", "tf_web_library")

licenses(["notice"])  # Apache 2.0

tf_web_library(
    name = "tf_convert_controls",
    srcs = [
        "tf-convert-controls.html",
        "tf-convert-controls.ts",
    ],
    path = "/tf-convert-controls",
    deps = [
        "//tensorboard/components/tf_dashboard_common",
        "//tensorboard/components/tf_imports:polymer",
        "//tensorboard/plugins/convert/tf_convert_common",
        "//tensorboard/plugins/convert/tf_convert_info",
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
    ],
)

tensorboard_webcomponent_library(
    name = "legacy",
    srcs = [":tf_convert_controls"],
    destdir = "tf-convert-controls",
    deps = [
        "//tensorboard/components/tf_dashboard_common:legacy",
        "//tensorboard/plugins/convert/tf_convert_common:legacy",
        "//tensorboard/plugins/convert/tf_convert_info:legacy",
        "//third_party/javascript/polymer/v1/paper-button:lib",
        "//third_party/javascript/polymer/v1/paper-dropdown-menu:lib",
        "//third_party/javascript/polymer/v1/paper-icon-button:lib",
        "//third_party/javascript/polymer/v1/paper-item:lib",
        "//third_party/javascript/polymer/v1/paper-menu:lib",
        "//third_party/javascript/polymer/v1/paper-radio-group:lib",
        "//third_party/javascript/polymer/v1/paper-toggle-button:lib",
        "//third_party/javascript/polymer/v1/paper-tooltip:lib",
        "//third_party/javascript/polymer/v1/paper-input:lib",
        "//third_party/javascript/polymer/v1/polymer:lib",
    ],
)
