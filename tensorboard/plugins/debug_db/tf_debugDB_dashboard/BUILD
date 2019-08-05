package(default_visibility = ["//tensorboard:internal"])

load("//tensorboard/defs:defs.bzl", "tensorboard_webcomponent_library")
load("//tensorboard/defs:web.bzl", "tf_web_library")

licenses(["notice"])  # Apache 2.0

tf_web_library(
    name = "tf_debugdb_dashboard",
    srcs = ["tf-debugdb-dashboard.html"],
    path = "/tf-debugdb-dashboard",
    visibility = ["//visibility:public"],
    deps = [
        "//tensorboard/components/tf_backend",
        "//tensorboard/components/tf_dashboard_common",
        "//tensorboard/components/tf_imports:polymer",
        "//tensorboard/components/tf_storage",
        "//tensorboard/components/tf_tensorboard:registry",
        "//tensorboard/components/vz_sorting",
        "//tensorboard/plugins/debug_db/tf_debugdb",
        "//tensorboard/plugins/debug_db/tf_debugdb_board",
        "//tensorboard/plugins/debug_db/tf_debugdb_controls",
        "//tensorboard/plugins/debug_db/tf_debugdb_loader:tf_debugdb_dashboard_loader",
        "@org_polymer_paper_tabs",
        "@org_polymer_paper_icon_button",
    ],
)

tensorboard_webcomponent_library(
    name = "legacy",
    srcs = [":tf_debugdb_dashboard"],
    destdir = "tf-debugdb-dashboard",
    deps = [
        "//tensorboard/components/tf_backend:legacy",
        "//tensorboard/components/tf_dashboard_common:legacy",
        "//tensorboard/components/vz_sorting:legacy",
        "//tensorboard/plugins/debug_db/tf_debugdb:legacy",
        "//tensorboard/plugins/debug_db/tf_debugdb_board:legacy",
        "//tensorboard/plugins/debug_db/tf_debugdb_controls:legacy",
        "//tensorboard/plugins/debug_db/tf_debugdb_loader:legacy_dashboard_loader",
        "//third_party/javascript/polymer/v1/polymer:lib",
        "//third_party/javascript/polymer/v1/paper-tabs:lib",
        "//third_party/javascript/polymer/v1/paper-icon-button:lib",
    ],
)
