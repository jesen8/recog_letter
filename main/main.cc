
#include "RecogLetter.h"

int main(int argc, char* argv[]) {
  
  RecogLetter recog_letter;

  int ret_code = recog_letter.init(argv[1], argv[2]);
  
  return ret_code;
}
//   std::vector<tensorflow::Flag> flag_list = {
//       Flag("image", &image, "image to be processed"),
//       Flag("graph", &graph, "graph to be executed"),
//       Flag("labels", &labels, "name of file containing labels"),
//       Flag("input_width", &input_width, "resize image to this width in pixels"),
//       Flag("input_height", &input_height,
//            "resize image to this height in pixels"),
//       Flag("input_mean", &input_mean, "scale pixel values to this mean"),
//       Flag("input_std", &input_std, "scale pixel values to this std deviation"),
//       Flag("input_layer", &input_layer, "name of input layer"),
//       Flag("output_layer", &output_layer, "name of output layer"),
//       Flag("self_test", &self_test, "run a self test"),
//       Flag("root_dir", &root_dir,
//            "interpret image and graph file names relative to this directory"),
//   };
//   string usage = tensorflow::Flags::Usage(argv[0], flag_list);
//   const bool parse_result = tensorflow::Flags::Parse(&argc, argv, flag_list);
//   if (!parse_result) {
//     LOG(ERROR) << "\n" << usage;
//     return -1;
//   }

  // We need to call this to set up global state for TensorFlow.
//   tensorflow::port::InitMain(usage.c_str(), &argc, &argv);
//   if (argc > 2) {
//     LOG(ERROR) << "Unknown argument " << argv[1] << "\n" << usage;
//     return -1;
//   }

  



  

//   return 0;
