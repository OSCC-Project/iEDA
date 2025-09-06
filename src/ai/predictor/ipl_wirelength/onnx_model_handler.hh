// ***************************************************************************************
// Copyright (c) 2023-2025 Peng Cheng Laboratory
// Copyright (c) 2023-2025 Institute of Computing Technology, Chinese Academy of Sciences
// Copyright (c) 2023-2025 Beijing Institute of Open Source Chip
//
// iEDA is licensed under Mulan PSL v2.
// You can use this software according to the terms and conditions of the Mulan PSL v2.
// You may obtain a copy of Mulan PSL v2 at:
// http://license.coscl.org.cn/MulanPSL2
//
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
// EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
// MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
//
// See the Mulan PSL v2 for more details.
// ***************************************************************************************
#ifndef ONNX_MODEL_HANDLER_HH
#define ONNX_MODEL_HANDLER_HH

#include <string>
#include <vector>
#include <memory>
#include <iostream>

#include "onnxruntime_cxx_api.h"

namespace ipl {

class ONNXModelHandler {
public:
    ONNXModelHandler();
    ~ONNXModelHandler();

    // Load ONNX model from file
    bool loadModel(const std::string& model_path);

    // Predict using the loaded ONNX model
    std::vector<float> predict(const std::vector<float>& input);

    // Get the number of input features expected by the model
    int getInputFeatureCount() const;

    // Get the number of output features produced by the model
    int getOutputFeatureCount() const;

private:
    Ort::Env _env;
    Ort::SessionOptions _session_options;
    std::unique_ptr<Ort::Session> _session;
    int _input_feature_count = 0;
    int _output_feature_count = 0;
    std::vector<std::string> _input_names;
    std::vector<std::string> _output_names;
    std::vector<std::vector<int64_t>> _input_shapes;
    std::vector<std::vector<int64_t>> _output_shapes;
};

} // namespace ipl

#endif // ONNX_MODEL_HANDLER_HH
