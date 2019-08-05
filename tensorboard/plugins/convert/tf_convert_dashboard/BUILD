package(default_visibility = ["//tensorboard:internal"])

load("//tensorboard/defs:defs.bzl", "tensorboard_webcomponent_library")
load("//tensorboard/defs:web.bzl", "tf_web_library")

licenses(["notice"])  # Apache 2.0

tf_web_library(
    name = "tf_convert_dashboard",
    srcs = ["tf-convert-dashboard.html"],
    path = "/tf-convert-dashboard",
    visibility = ["//visibility:public"],
    deps = [
        "//tensorboard/components/tf_backend",
        "//tensorboard/components/tf_dashboard_common",
        "//tensorboard/components/tf_imports:polymer",
        "//tensorboard/components/tf_storage",
        "//tensorboard/components/tf_tensorboard:registry",
        "//tensorboard/components/vz_sorting",
        "//tensorboard/plugins/convert/tf_convert",
        "//tensorboard/plugins/convert/tf_convert_board",
        "//tensorboard/plugins/convert/tf_convert_controls",
        "//tensorboard/plugins/convert/tf_convert_loader:tf_convert_dashboard_loader",
    ],
)

tensorboard_webcomponent_library(
    name = "legacy",
    srcs = [":tf_convert_dashboard"],
    destdir = "tf-convert-dashboard",
    deps = [
        "//tensorboard/components/tf_backend:legacy",
        "//tensorboard/components/tf_dashboard_common:legacy",
        "//tensorboard/components/vz_sorting:legacy",
        "//tensorboard/plugins/convert/tf_convert:legacy",
        "//tensorboard/plugins/convert/tf_convert_board:legacy",
        "//tensorboard/plugins/convert/tf_convert_controls:legacy",
        "//tensorboard/plugins/convert/tf_convert_loader:legacy_dashboard_loader",
        "//third_party/javascript/polymer/v1/polymer:lib",
    ],
)
