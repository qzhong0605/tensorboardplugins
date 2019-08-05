package(default_visibility = ["//tensorboard:internal"])

load("//tensorboard/defs:defs.bzl", "tensorboard_webcomponent_library")
load("//tensorboard/defs:web.bzl", "tf_web_library")

licenses(["notice"])  # Apache 2.0

tf_web_library(
    name = "tf_debugdb_loader",
    srcs = [
        "tf-debugdb-loader.html",
        "tf-debugdb-loader.ts",
    ],
    path = "/tf-debugdb-loader",
    deps = [
        "//tensorboard/plugins/debug_db/tf_debugdb_common",
        "//tensorboard/components/tf_imports:polymer",
    ],
)

tensorboard_webcomponent_library(
    name = "legacy",
    srcs = [":tf_debugdb_loader"],
    destdir = "tf-debugdb-loader",
    deps = [
        "//tensorboard/plugins/debug_db/tf_debugdb_common:legacy",
        "//third_party/javascript/polymer/v1/polymer:lib",
    ],
)

tf_web_library(
    name = "tf_debugdb_dashboard_loader",
    srcs = [
        "tf-debugdb-dashboard-loader.html",
        "tf-debugdb-dashboard-loader.ts",
    ],
    path = "/tf-debugdb-loader",
    deps = [
        "//tensorboard/components/tf_backend",
        "//tensorboard/plugins/debug_db/tf_debugdb_common",
        "//tensorboard/plugins/debug_db/tf_debugdb_controls",
        "//tensorboard/components/tf_imports:polymer",
    ],
)

tensorboard_webcomponent_library(
    name = "legacy_dashboard_loader",
    srcs = [":tf_debugdb_dashboard_loader"],
    destdir = "tf-debugdb-loader",
    deps = [
        "//tensorboard/components/tf_backend:legacy",
        "//tensorboard/plugins/debug_db/tf_debugdb_common:legacy",
        "//tensorboard/plugins/debug_db/tf_debugdb_controls:legacy",
        "//third_party/javascript/polymer/v1/polymer:lib",
    ],
)
