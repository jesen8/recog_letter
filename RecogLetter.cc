

// #include <cstddef>
// #include <opencv2/opencv.hpp>
#include <stdio.h>
// #include <jpeglib.h>
// #include <setjmp.h>
#include <iostream>
#include <fstream>

#include "RecogLetter.h"


#include "tensorflow/core/framework/graph.pb.h"
// #include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
// #include "tensorflow/core/platform/init_main.h"

// #include "tensorflow/core/platform/types.h"
// #include "tensorflow/core/public/session.h"
// #include "tensorflow/core/util/command_line_flags.h"

#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/image_ops.h"
// #include "tensorflow/cc/ops/standard_ops.h"
// #include "tensorflow/core/lib/strings/str_util.h"


// These are all common classes it's handy to reference with no namespace.
// using tensorflow::Flag;
using tensorflow::int32;
using tensorflow::Status;
using tensorflow::string;
using tensorflow::Tensor;
// using namespace cv;

RecogLetter::RecogLetter()
{

}
RecogLetter::~RecogLetter()
{

}




// Reads a model graph definition from disk, and creates a session object you
// can use to run it.
Status RecogLetter::LoadGraph(string graph_file_name,
                 std::unique_ptr<tensorflow::Session>* session) {
  tensorflow::GraphDef graph_def;
  Status load_graph_status =
      ReadBinaryProto(tensorflow::Env::Default(), graph_file_name, &graph_def);
  if (!load_graph_status.ok()) {
    return tensorflow::errors::NotFound("Failed to load compute graph at '",
                                        graph_file_name, "'");
  }
  session->reset(tensorflow::NewSession(tensorflow::SessionOptions()));
  Status session_create_status = (*session)->Create(graph_def);
  if (!session_create_status.ok()) {
    return session_create_status;
  }
  return Status::OK();
}

// Analyzes the output of the Inception graph to retrieve the highest scores and
// their positions in the tensor, which correspond to categories.
Status RecogLetter::GetTopLabels(const std::vector<Tensor>& outputs, int how_many_labels,
                    Tensor* out_indices, Tensor* out_scores) {
  const Tensor& unsorted_scores_tensor = outputs[0];
  auto unsorted_scores_flat = unsorted_scores_tensor.flat<float>();
  std::vector<std::pair<int, float>> scores;
  for (int i = 0; i < unsorted_scores_flat.size(); ++i) {
    scores.push_back(std::pair<int, float>({i, unsorted_scores_flat(i)}));
  }
  std::sort(scores.begin(), scores.end(),
            [](const std::pair<int, float>& left,
               const std::pair<int, float>& right) {
              return left.second > right.second;
            });
  scores.resize(how_many_labels);
  Tensor sorted_indices(tensorflow::DT_INT32, {how_many_labels});
  Tensor sorted_scores(tensorflow::DT_FLOAT, {how_many_labels});
  for (int i = 0; i < scores.size(); ++i) {
    sorted_indices.flat<int>()(i) = scores[i].first;
    sorted_scores.flat<float>()(i) = scores[i].second;
  }
  *out_indices = sorted_indices;
  *out_scores = sorted_scores;
  return Status::OK();
}

// Given the output of a model run, and the name of a file containing the labels
// this prints out the top five highest-scoring values.
Status RecogLetter::PrintTopLabels(const std::vector<Tensor>& outputs, 
                  std::string labels_file_name, char* label_index, float* score) 
{


  const int how_many_labels = _how_many_labels;
  Tensor indices;
  Tensor scores;
  TF_RETURN_IF_ERROR(GetTopLabels(outputs, how_many_labels, &indices, &scores));
  tensorflow::TTypes<float>::Flat scores_flat = scores.flat<float>();
  tensorflow::TTypes<int32>::Flat indices_flat = indices.flat<int32>();
  // for (int pos = 0; pos < how_many_labels; ++pos) {
  // int label_index0 = indices_flat(pos);
  // float score0 = scores_flat(pos);
  //LOG(INFO) << indices_flat(0) << " (" <<  " " << scores_flat(0);
  // }
  // LOG(INFO) <<  "-------------";
  *label_index = mapping[indices_flat(0)];
  *score = scores_flat(0);
  // LOG(INFO) <<  "label_index: " << label_index << " score: " << score;
  // LOG(INFO) <<  "+++++++++++++++";
  return Status::OK();
}

// This is a testing function that returns whether the top label index is the
// one that's expected.
Status RecogLetter::CheckTopLabel(const std::vector<Tensor>& outputs, int expected,
                     bool* is_expected) {
  *is_expected = false;
  Tensor indices;
  Tensor scores;
  const int how_many_labels = 2;
  TF_RETURN_IF_ERROR(GetTopLabels(outputs, how_many_labels, &indices, &scores));
  tensorflow::TTypes<int32>::Flat indices_flat = indices.flat<int32>();
  if (indices_flat(0) != expected) {
    LOG(ERROR) << "Expected label #" << expected << " but got #"
               << indices_flat(0);
    *is_expected = false;
  } else {
    *is_expected = true;
  }
  return Status::OK();
}

int RecogLetter::init(const std::string &model_path, const char* txtmap_path)
{
    // First we load and initialize the model.
  
  string graph_path = tensorflow::io::JoinPath(root_dir, model_path);
  Status load_graph_status = LoadGraph(graph_path, &session);
  if (!load_graph_status.ok()) {
    LOG(ERROR) << load_graph_status;
    return -1;
  }

  if (init_dictionary(txtmap_path) != 0){
    return -1;
  }

  return 0;
}

int RecogLetter::init_dictionary(const std::string& filename)
{
  std::ifstream inf(filename, std::ios::in);
  if(!inf.is_open())
  { return -1; }

  //LOG(INFO) <<"read dictionary file "<<filename;
  std::string line;
  std::vector<std::string> splits;
  while(!inf.eof()){
    inf>>line;

    splits = tensorflow::str_util::Split(line, "," );
    
    this->mapping[std::stoi(splits[0])] = splits[1][0];
  }
  inf.close();
  return 0;
}

int RecogLetter::recog(cv::Mat *cv_image, char* label_index, float* score)
{
  if (input_width != cv_image->cols && input_height != cv_image->rows){
    return -1;
  }
  Tensor inputImg(tensorflow::DT_FLOAT, tensorflow::TensorShape({1,input_height,input_width,input_channel}));

  float *p = inputImg.flat<float>().data();
  cv::Mat cameraImg(input_height, input_width, CV_32FC1, p);

  cv_image->convertTo(cameraImg, CV_32FC1);
  const Tensor& resized_tensor = inputImg;

  std::vector<Tensor> outputs;
  Status run_status = session->Run({{input_layer, resized_tensor}},
                                   {output_layer}, {}, &outputs);
  if (!run_status.ok()) {
    LOG(ERROR) << "Running model failed: " << run_status;
    return -1;
  } 

  Status print_status = PrintTopLabels(outputs, labels, label_index, score);
  if (!print_status.ok()) {
    LOG(ERROR) << "Running print failed: " << print_status;
    return -1;
  }
  return 0;
}
