package(default_visibility = ["//tensorboard:internal"])

load("//tensorboard/defs:defs.bzl", "tensorboard_webcomponent_library")
load("//tensorboard/defs:web.bzl", "tf_web_library")

licenses(["notice"])  # Apache 2.0

tf_web_library(
    name = "tf_debugDB_board",
    srcs = ["tf-debugDB-board.html"],
    path = "/tf-debugDB-board",
    deps = [
        "//tensorboard/plugins/debugDB/tf_debugDB",
        "//tensorboard/plugins/debugDB/tf_debugDB_common",
        "//tensorboard/components/tf_imports:polymer",
        "@org_polymer_paper_progress",
    ],
)

tensorboard_webcomponent_library(
    name = "legacy",
    srcs = [":tf_debugDB_board"],
    destdir = "tf-debugDB-board",
    deps = [
        "//tensorboard/plugins/debugDB/tf_debugDB:legacy",
        "//tensorboard/plugins/debugDB/tf_debugDB_common:legacy",
        "//third_party/javascript/polymer/v1/paper-progress:lib",
        "//third_party/javascript/polymer/v1/polymer:lib",
    ],
)

