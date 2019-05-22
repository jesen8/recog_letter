load("//tensorflow:tensorflow.bzl", "tf_cc_binary")

cc_library(
    name = "recog_letter_c",
    srcs = ["RecogLetter.cc"],
    hdrs = ["RecogLetter.h"],
    deps = [
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:client_session",
        "//tensorflow/core:tensorflow",
        "@opencv//:opencv",
    ],
    
    linkstatic = True,
)

tf_cc_binary(
    name = "recog",
    srcs = ["main.cc"],
    deps = [
        ":recog_letter_c",
    ],
)

tf_cc_binary(
    name = "librecog_letter.so",
    srcs = ["dll.cc"],
    deps = [
        ":recog_letter_c",
        
    ],
    linkshared = 1,
)

# bazel build --config=opt --config=monolithic --config=noaws //tensorflow/cc/work/recog_letter:librecog_letter.so