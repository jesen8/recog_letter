cmake_minimum_required(VERSION 2.8)

set(tensoflow_source_dir /home/swls/work_dir/github/tensorflow-1.13.0-rc2)

find_package(OpenCV REQUIRED) 

SET(CMAKE_CXX_FLAGS "-std=c++11")

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${tensoflow_source_dir}
    ${tensoflow_source_dir}/bazel-out/k8-opt/genfiles
    ${tensoflow_source_dir}/tensorflow/contrib/makefile/gen/proto
    ${tensoflow_source_dir}/tensorflow/contrib/makefile/gen/proto_text
    ${tensoflow_source_dir}/tensorflow/contrib/makefile/gen/host_obj
    ${tensoflow_source_dir}/tensorflow/contrib/makefile/gen/protobuf/include
	${tensoflow_source_dir}/tensorflow/contrib/makefile/downloads/eigen
    ${tensoflow_source_dir}/tensorflow/contrib/makefile/downloads/nsync/public
    ${tensoflow_source_dir}/tensorflow/contrib/makefile/downloads/absl
    ${tensoflow_source_dir}/tensorflow/contrib/makefile/downloads/googletest/googletest/include
    )
	# ${tensoflow_source_dir}/include)

# link_directories(
#     ${OpenCV_LIBRARY_DIRS}
#     ${tensoflow_source_dir}/bazel-bin/tensorflow
#     ${tensoflow_source_dir}/tensorflow/contrib/makefile/downloads/nsync/builds/default.linux.c++11
#     ${tensoflow_source_dir}/tensorflow/contrib/makefile/gen/lib
#     ${tensoflow_source_dir}/tensorflow/contrib/makefile/gen/protobuf/lib
#     )


link_directories(
    ${OpenCV_LIBRARY_DIRS}
    )
# set(DEEP_CTR_LINKER_LIBS "")
# list(APPEND DEEP_CTR_LINKER_LIBS
#     libtensorflow-core.a
#     # tensorflow_framework
#     libprotobuf.a
#     libnsync.a
#     pthread
#     m 
#     z)

# /home/swls/work_dir/github/tensorflow-1.13.0-rc2/tensorflow/cc/work/recog_letter/build/librecog_letter.a
SET(param "-std=c++11 -Wall -L${tensoflow_source_dir}/tensorflow/contrib/makefile/gen/protobuf-host/lib -Wl,--allow-multiple-definition -Wl,--whole-archive  ${tensoflow_source_dir}/tensorflow/contrib/makefile/gen/lib/libtensorflow-core.a -Wl,--no-whole-archive ${tensoflow_source_dir}/tensorflow/contrib/makefile/downloads/nsync/builds/default.linux.c++11/nsync.a ${tensoflow_source_dir}/tensorflow/cc/work/recog_letter/build/librecog_letter.a -lstdc++ -l:libprotobuf.a -lz -lm -ldl -lpthread -lrt")


# SET(LDFLAGS "-msse4.1 -fPIC -O3 -ldl")
# SET(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} ${LDFLAGS})
# -nostartfiles -ggdb -Wall -Wsign-compare -finline-functions -Wl,--allow-multiple-definition -Wl,--whole-archive


# executable
set(SIMPLE_MODEL "recog_letter_test")
add_executable(${SIMPLE_MODEL} "main.cc")
# add_executable(${SIMPLE_MODEL} dll.cc RecogLetter.cc RecogLetter.h)
# add_library(${SIMPLE_MODEL} STATIC RecogLetter.cc RecogLetter.h)
# set(EXECUTABLE_OUTPUT_PATH "${PROJECT_SOURCE_DIR}/bin")



# target_link_libraries(${SIMPLE_MODEL} /home/swls/work_dir/github/tensorflow-1.13.0-rc2/tensorflow/cc/work/recog_letter/librecog_letter.a ${CMAKE_CXX_FLAGS})

target_link_libraries(${SIMPLE_MODEL} ${OpenCV_LIBS})

target_link_libraries(${SIMPLE_MODEL} ${param})
# target_link_libraries(${SIMPLE_MODEL} ${OpenCV_LIBS})

# target_link_libraries(${SIMPLE_MODEL} ${DEEP_CTR_LINKER_LIBS} ${CMAKE_CXX_FLAGS})

# target_link_libraries(${SIMPLE_MODEL} ${OpenCV_LIBS})

# # executable
# set(DEEP_MODEL "deep_model.bin")
# add_executable(${DEEP_MODEL} "deep_model.cc")
# set(EXECUTABLE_OUTPUT_PATH "${PROJECT_SOURCE_DIR}/bin")
# target_link_libraries(${DEEP_MODEL} ${DEEP_CTR_LINKER_LIBS})



# CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

# project(tensorflow-predictor-cpp)

# SET(TENSORFLOW_INCLUDE_PATH ${CMAKE_SOURCE_DIR}/tensorflow/include)
# SET(TENSORFLOW_LIBARY ${CMAKE_SOURCE_DIR}/tensorflow/lib/libtensorflow-core.a)
# # MESSAGE(STATUS "TENSORFLOW_INCLUDE_PATH ${TENSORFLOW_INCLUDE_PATH}")
# # MESSAGE(STATUS "TENSORFLOW_LIBARY ${TENSORFLOW_LIBARY}")

# SET(TENSORFLOW_PROTOBUF_INCLUDE_PATH ${CMAKE_SOURCE_DIR}/tensorflow/protobuf/include)
# SET(TENSORFLOW_PROTOBUF_LIBRARY_PATH ${CMAKE_SOURCE_DIR}/tensorflow/protobuf/lib)
# SET(TENSORFLOW_PROTOBUF_LIBRARY ${TENSORFLOW_PROTOBUF_LIBRARY_PATH}/libprotobuf.a)
# SET(TENSORFLOW_PROTOBUF_LITE_LIBRARY ${TENSORFLOW_PROTOBUF_LIBRARY_PATH}/libprotobuf-lite.a)
# SET(TENSORFLOW_PROTOC_LIBRARY ${TENSORFLOW_PROTOBUF_LIBRARY_PATH}/libprotoc.a)
# # MESSAGE(STATUS "TENSORFLOW_PROTOBUF_INCLUDE_PATH ${TENSORFLOW_PROTOBUF_INCLUDE_PATH}")
# # MESSAGE(STATUS "TENSORFLOW_PROTOBUF_LIBRARY_PATH ${TENSORFLOW_PROTOBUF_LIBRARY_PATH}")

# SET(TENSORFLOW_NSYNC_INCLUDE_PATH ${CMAKE_SOURCE_DIR}/tensorflow/nsync/include)
# SET(TENSORFLOW_NSYNC_LIBRARY_PATH ${CMAKE_SOURCE_DIR}/tensorflow/nsync/lib)
# # MESSAGE(STATUS "TENSORFLOW_NSYNC_INCLUDE_PATH ${TENSORFLOW_NSYNC_INCLUDE_PATH}")
# # MESSAGE(STATUS "TENSORFLOW_NSYNC_LIBRARY_PATH ${TENSORFLOW_NSYNC_LIBRARY_PATH}")
# SET(TENSORFLOW_NSYNC_LIBRARY ${TENSORFLOW_NSYNC_LIBRARY_PATH}/libnsync.a)

# SET(TENSORFLOW_PROTO_INCLUDE_PATH ${CMAKE_SOURCE_DIR}/tensorflow/proto)
# SET(TENSORFLOW_PROTO_TEXT_INCLUDE_PATH ${CMAKE_SOURCE_DIR}/tensorflow/proto_text)
# SET(TENSORFLOW_HOST_OBJ_INCLUDE_PATH ${CMAKE_SOURCE_DIR}/tensorflow/host_obj)
# SET(TENSORFLOW_EIGEN_INCLUDE_PATH ${CMAKE_SOURCE_DIR}/tensorflow/eigen3)
# SET(TENSORFLOW_ABSL_INCLUDE_PATH ${CMAKE_SOURCE_DIR}/tensorflow/absl)
# SET(TENSORFLOW_THIRD_PARTY_INCLUDE_PATH ${CMAKE_SOURCE_DIR}/tensorflow/tensorflow_third_party)
# # MESSAGE(STATUS "TENSORFLOW_PROTO_INCLUDE_PATH ${TENSORFLOW_PROTO_INCLUDE_PATH}")
# # MESSAGE(STATUS "TENSORFLOW_PROTO_TEXT_INCLUDE_PATH ${TENSORFLOW_PROTO_TEXT_INCLUDE_PATH}")
# # MESSAGE(STATUS "TENSORFLOW_HOST_OBJ_INCLUDE_PATH ${TENSORFLOW_HOST_OBJ_INCLUDE_PATH}")
# # MESSAGE(STATUS "TENSORFLOW_EIGEN_INCLUDE_PATH ${TENSORFLOW_EIGEN_INCLUDE_PATH}")
# # MESSAGE(STATUS "TENSORFLOW_ABSL_INCLUDE_PATH ${TENSORFLOW_ABSL_INCLUDE_PATH}")
# # MESSAGE(STATUS "TENSORFLOW_THIRD_PARTY_INCLUDE_PATH ${TENSORFLOW_THIRD_PARTY_INCLUDE_PATH}")

# INCLUDE_DIRECTORIES(${TENSORFLOW_PROTOBUF_INCLUDE_PATH})
# INCLUDE_DIRECTORIES(${TENSORFLOW_INCLUDE_PATH})
# INCLUDE_DIRECTORIES(${TENSORFLOW_PROTO_INCLUDE_PATH})
# INCLUDE_DIRECTORIES(${TENSORFLOW_PROTO_TEXT_INCLUDE_PATH})
# INCLUDE_DIRECTORIES(${TENSORFLOW_HOST_OBJ_INCLUDE_PATH})
# INCLUDE_DIRECTORIES(${TENSORFLOW_EIGEN_INCLUDE_PATH})
# INCLUDE_DIRECTORIES(${TENSORFLOW_ABSL_INCLUDE_PATH})
# INCLUDE_DIRECTORIES(${TENSORFLOW_NSYNC_INCLUDE_PATH})
# INCLUDE_DIRECTORIES(${TENSORFLOW_THIRD_PARTY_INCLUDE_PATH})


# # ADD_EXECUTABLE(load_model load_model.cpp)
# add_executable(main load_model.cpp)

# SET(LOAD_MODEL_LIBRARIES
#     # ${TENSORFLOW_PROTOBUF_LIBRARY}
#     ${TENSORFLOW_PROTOC_LIBRARY}
#     # ${TENSORFLOW_NSYNC_LIBRARY}
#     ${TENSORFLOW_LIBARY})

# SET(LDFLAGS "-std=c++11 -msse4.1 -fPIC -O3 -ggdb -Wall -finline-functions -Wl,--allow-multiple-definition -Wl,--whole-archive")
# # -undefined  dynamic_lookup -all_load
# SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${LDFLAGS}")

# link_directories(${CMAKE_SOURCE_DIR}/tensorflow/protobuf/lib)
# target_link_libraries(main protobuf)

# # MESSAGE(STATUS "CMAKE_CXX_COMPILER: ${CMAKE_CXX_COMPILER}")
# # MESSAGE(STATUS "CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")
# TARGET_LINK_LIBRARIES(main ${LOAD_MODEL_LIBRARIES} ${CMAKE_CXX_FLAGS})