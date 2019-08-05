package(default_visibility = ["//tensorboard:internal"])

load("//tensorboard/defs:defs.bzl", "tensorboard_webcomponent_library")
load("//tensorboard/defs:web.bzl", "tf_web_library")

licenses(["notice"])  # Apache 2.0

tf_web_library(
    name = "tf_debugDB_loader",
    srcs = [
        "tf-debugDB-loader.html",
        "tf-debugDB-loader.ts",
    ],
    path = "/tf-debugDB-loader",
    deps = [
        "//tensorboard/plugins/debugDB/tf_debugDB_common",
        "//tensorboard/components/tf_imports:polymer",
    ],
)

tensorboard_webcomponent_library(
    name = "legacy",
    srcs = [":tf_debugDB_loader"],
    destdir = "tf-debugDB-loader",
    deps = [
        "//tensorboard/plugins/debugDB/tf_debugDB_common:legacy",
        "//third_party/javascript/polymer/v1/polymer:lib",
    ],
)

tf_web_library(
    name = "tf_debugDB_dashboard_loader",
    srcs = [
        "tf-debugDB-dashboard-loader.html",
        "tf-debugDB-dashboard-loader.ts",
    ],
    path = "/tf-debugDB-loader",
    deps = [
        "//tensorboard/components/tf_backend",
        "//tensorboard/plugins/debugDB/tf_debugDB_common",
        "//tensorboard/plugins/debugDB/tf_debugDB_controls",
        "//tensorboard/components/tf_imports:polymer",
    ],
)

tensorboard_webcomponent_library(
    name = "legacy_dashboard_loader",
    srcs = [":tf_debugDB_dashboard_loader"],
    destdir = "tf-debugDB-loader",
    deps = [
        "//tensorboard/components/tf_backend:legacy",
        "//tensorboard/plugins/debugDB/tf_debugDB_common:legacy",
        "//tensorboard/plugins/debugDB/tf_debugDB_controls:legacy",
        "//third_party/javascript/polymer/v1/polymer:lib",
    ],
)
