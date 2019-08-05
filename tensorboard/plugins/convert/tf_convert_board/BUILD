package(default_visibility = ["//tensorboard:internal"])

load("//tensorboard/defs:defs.bzl", "tensorboard_webcomponent_library")
load("//tensorboard/defs:web.bzl", "tf_web_library")

licenses(["notice"])  # Apache 2.0

tf_web_library(
    name = "tf_convert_board",
    srcs = ["tf-convert-board.html"],
    path = "/tf-convert-board",
    deps = [
        "//tensorboard/plugins/convert/tf_convert",
        "//tensorboard/plugins/convert/tf_convert_common",
        "//tensorboard/components/tf_imports:polymer",
        "@org_polymer_paper_progress",
    ],
)

tensorboard_webcomponent_library(
    name = "legacy",
    srcs = [":tf_convert_board"],
    destdir = "tf-convert-board",
    deps = [
        "//tensorboard/plugins/convert/tf_convert:legacy",
        "//tensorboard/plugins/convert/tf_convert_common:legacy",
        "//third_party/javascript/polymer/v1/paper-progress:lib",
        "//third_party/javascript/polymer/v1/polymer:lib",
    ],
)

