#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/types.h"

// using tensorflow::string;
using tensorflow::Status;
using tensorflow::Tensor;
using tensorflow::int32;
class RecogLetter
{
    public:
    RecogLetter();
    ~RecogLetter();

    public:
    int init(const std::string &model_path, const char* txtmap_path);
    int recog(cv::Mat *cv_image, char* label_index, float* score);

    private:
    int init_dictionary(const std::string& filename);

    Status LoadGraph(std::string graph_file_name,
                 std::unique_ptr<tensorflow::Session>* session);
    Status GetTopLabels(const std::vector<Tensor>& outputs, int how_many_labels,
                    Tensor* out_indices, Tensor* out_scores) ;
    Status PrintTopLabels(const std::vector<Tensor>& outputs, 
                    std::string labels_file_name, char* label_index, float* score); 
    Status CheckTopLabel(const std::vector<Tensor>& outputs, int expected,
                     bool* is_expected);

    private:
    std::unique_ptr<tensorflow::Session> session;
    std::string labels =  "";

    int32 _how_many_labels = 26;
    int32 input_channel = 1;
    int32 input_width = 32;
    int32 input_height = 32;
    int32 input_mean = 0;
    int32 input_std = 255;
    std::string input_layer = "input_node";
    std::string output_layer = "output_node";
    bool self_test = false;
    std::string root_dir = "";

    std::unordered_map<int, char> mapping;
};