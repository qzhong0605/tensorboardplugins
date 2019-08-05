package(default_visibility = ["//tensorboard:internal"])

load("//tensorboard/defs:defs.bzl", "tensorboard_webcomponent_library")
load("//tensorboard/defs:web.bzl", "tf_web_library")

licenses(["notice"])  # Apache 2.0

tf_web_library(
    name = "tf_convert_loader",
    srcs = [
        "tf-convert-loader.html",
        "tf-convert-loader.ts",
    ],
    path = "/tf-convert-loader",
    deps = [
        "//tensorboard/plugins/convert/tf_convert_common",
        "//tensorboard/components/tf_imports:polymer",
    ],
)

tensorboard_webcomponent_library(
    name = "legacy",
    srcs = [":tf_convert_loader"],
    destdir = "tf-convert-loader",
    deps = [
        "//tensorboard/plugins/convert/tf_convert_common:legacy",
        "//third_party/javascript/polymer/v1/polymer:lib",
    ],
)

tf_web_library(
    name = "tf_convert_dashboard_loader",
    srcs = [
        "tf-convert-dashboard-loader.html",
        "tf-convert-dashboard-loader.ts",
    ],
    path = "/tf-convert-loader",
    deps = [
        "//tensorboard/components/tf_backend",
        "//tensorboard/plugins/convert/tf_convert_common",
        "//tensorboard/plugins/convert/tf_convert_controls",
        "//tensorboard/components/tf_imports:polymer",
    ],
)

tensorboard_webcomponent_library(
    name = "legacy_dashboard_loader",
    srcs = [":tf_convert_dashboard_loader"],
    destdir = "tf-convert-loader",
    deps = [
        "//tensorboard/components/tf_backend:legacy",
        "//tensorboard/plugins/convert/tf_convert_common:legacy",
        "//tensorboard/plugins/convert/tf_convert_controls:legacy",
        "//third_party/javascript/polymer/v1/polymer:lib",
    ],
)
