package(default_visibility = ["//tensorboard:internal"])

load("//tensorboard/defs:defs.bzl", "tensorboard_webcomponent_library")
load("//tensorboard/defs:web.bzl", "tf_web_library")

licenses(["notice"])  # Apache 2.0

tf_web_library(
    name = "tf_debugdb_board",
    srcs = ["tf-debugdb-board.html"],
    path = "/tf-debugdb-board",
    deps = [
        "//tensorboard/plugins/debug_db/tf_debugdb",
        "//tensorboard/plugins/debug_db/tf_debugdb_common",
        "//tensorboard/components/tf_imports:polymer",
        "@org_polymer_paper_progress",
    ],
)

tensorboard_webcomponent_library(
    name = "legacy",
    srcs = [":tf_debugdb_board"],
    destdir = "tf-debugdb-board",
    deps = [
        "//tensorboard/plugins/debug_db/tf_debugdb:legacy",
        "//tensorboard/plugins/debug_db/tf_debugdb_common:legacy",
        "//third_party/javascript/polymer/v1/paper-progress:lib",
        "//third_party/javascript/polymer/v1/polymer:lib",
    ],
)

