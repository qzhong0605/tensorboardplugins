package(default_visibility = ["//tensorboard:internal"])

load("//tensorboard/defs:defs.bzl", "tensorboard_webcomponent_library")
load("//tensorboard/defs:web.bzl", "tf_web_library")

licenses(["notice"])  # Apache 2.0

tf_web_library(
    name = "tf_debugDB_dashboard",
    srcs = ["tf-debugDB-dashboard.html"],
    path = "/tf-debugDB-dashboard",
    visibility = ["//visibility:public"],
    deps = [
        "//tensorboard/components/tf_backend",
        "//tensorboard/components/tf_dashboard_common",
        "//tensorboard/components/tf_imports:polymer",
        "//tensorboard/components/tf_storage",
        "//tensorboard/components/tf_tensorboard:registry",
        "//tensorboard/components/vz_sorting",
        "//tensorboard/plugins/debugDB/tf_debugDB",
        "//tensorboard/plugins/debugDB/tf_debugDB_board",
        "//tensorboard/plugins/debugDB/tf_debugDB_controls",
        "//tensorboard/plugins/debugDB/tf_debugDB_loader:tf_debugDB_dashboard_loader",
        "@org_polymer_paper_tabs",
        "@org_polymer_paper_icon_button",
    ],
)

tensorboard_webcomponent_library(
    name = "legacy",
    srcs = [":tf_debugDB_dashboard"],
    destdir = "tf-debugDB-dashboard",
    deps = [
        "//tensorboard/components/tf_backend:legacy",
        "//tensorboard/components/tf_dashboard_common:legacy",
        "//tensorboard/components/vz_sorting:legacy",
        "//tensorboard/plugins/debugDB/tf_debugDB:legacy",
        "//tensorboard/plugins/debugDB/tf_debugDB_board:legacy",
        "//tensorboard/plugins/debugDB/tf_debugDB_controls:legacy",
        "//tensorboard/plugins/debugDB/tf_debugDB_loader:legacy_dashboard_loader",
        "//third_party/javascript/polymer/v1/polymer:lib",
        "//third_party/javascript/polymer/v1/paper-tabs:lib",
        "//third_party/javascript/polymer/v1/paper-icon-button:lib",
    ],
)
