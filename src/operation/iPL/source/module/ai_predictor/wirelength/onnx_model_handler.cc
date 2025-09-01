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
#include "onnx_model_handler.hh"

#include <iostream>

namespace ipl {

ONNXModelHandler::ONNXModelHandler() : _env(ORT_LOGGING_LEVEL_WARNING, "ONNXModelHandler") {
    // Initialize ONNX Runtime environment
    _session_options.SetIntraOpNumThreads(1);
    _session_options.SetInterOpNumThreads(1);
    _session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);
}

ONNXModelHandler::~ONNXModelHandler() {
    // Clean up - smart pointers handle this automatically
}

bool ONNXModelHandler::loadModel(const std::string& model_path) {
    try {
        // Create session from model file
        _session = std::make_unique<Ort::Session>(_env, model_path.c_str(), _session_options);

        // Get allocator
        Ort::AllocatorWithDefaultOptions allocator;

        // Get input information
        size_t num_input_nodes = _session->GetInputCount();
        if (num_input_nodes == 0) {
            std::cerr << "Model has no input nodes" << std::endl;
            return false;
        }

        // Get input names and shapes
        _input_names.clear();
        _input_shapes.clear();
        
        for (size_t i = 0; i < num_input_nodes; i++) {
            // Get input name using the correct API
            Ort::AllocatedStringPtr input_name_ptr = _session->GetInputNameAllocated(i, allocator);
            _input_names.push_back(std::string(input_name_ptr.get()));

            // Get input type info
            Ort::TypeInfo input_type_info = _session->GetInputTypeInfo(i);
            auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
            
            // Get input shape
            std::vector<int64_t> input_shape = input_tensor_info.GetShape();
            _input_shapes.push_back(input_shape);
        }

        // Get output information
        size_t num_output_nodes = _session->GetOutputCount();
        if (num_output_nodes == 0) {
            std::cerr << "Model has no output nodes" << std::endl;
            return false;
        }

        // Get output names and shapes
        _output_names.clear();
        _output_shapes.clear();
        
        for (size_t i = 0; i < num_output_nodes; i++) {
            // Get output name
            Ort::AllocatedStringPtr output_name_ptr = _session->GetOutputNameAllocated(i, allocator);
            _output_names.push_back(std::string(output_name_ptr.get()));

            // Get output type info
            Ort::TypeInfo output_type_info = _session->GetOutputTypeInfo(i);
            auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
            
            // Get output shape
            std::vector<int64_t> output_shape = output_tensor_info.GetShape();
            _output_shapes.push_back(output_shape);
        }

        // Validate shapes for our use case
        if (_input_shapes[0].size() != 2) {
            std::cerr << "Unexpected input shape dimension: " << _input_shapes[0].size() << std::endl;
            return false;
        }

        if (_output_shapes[0].size() != 2) {
            std::cerr << "Unexpected output shape dimension: " << _output_shapes[0].size() << std::endl;
            return false;
        }

        // Set feature counts (assuming batch dimension is dynamic or 1)
        _input_feature_count = static_cast<int>(_input_shapes[0][1]);
        _output_feature_count = static_cast<int>(_output_shapes[0][1]);

        std::cout << "Successfully loaded ONNX model from " << model_path << std::endl;
        std::cout << "Input name: " << _input_names[0] << std::endl;
        std::cout << "Output name: " << _output_names[0] << std::endl;
        std::cout << "Input feature count: " << _input_feature_count << std::endl;
        std::cout << "Output feature count: " << _output_feature_count << std::endl;

        return true;
    } catch (const Ort::Exception& e) {
        std::cerr << "ONNX exception: " << e.what() << std::endl;
        return false;
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return false;
    }
}

std::vector<float> ONNXModelHandler::predict(const std::vector<float>& input) {
    if (!_session) {
        std::cerr << "Model not loaded" << std::endl;
        return {};
    }

    if (input.size() != static_cast<size_t>(_input_feature_count)) {
        std::cerr << "Input feature count mismatch: expected " << _input_feature_count
                  << ", got " << input.size() << std::endl;
        return {};
    }

    try {
        // Create input tensor
        const std::vector<int64_t> input_shape = {1, _input_feature_count}; // Batch size 1
        
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
            OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info, const_cast<float*>(input.data()), input.size(), 
            input_shape.data(), input_shape.size());

        if (!input_tensor.IsTensor()) {
            std::cerr << "Failed to create input tensor" << std::endl;
            return {};
        }

        // Prepare input and output names
        std::vector<const char*> input_names_cstr;
        std::vector<const char*> output_names_cstr;
        
        for (const auto& name : _input_names) {
            input_names_cstr.push_back(name.c_str());
        }
        for (const auto& name : _output_names) {
            output_names_cstr.push_back(name.c_str());
        }

        // Run inference
        std::vector<Ort::Value> output_tensors = _session->Run(
            Ort::RunOptions{nullptr}, 
            input_names_cstr.data(), &input_tensor, 1,
            output_names_cstr.data(), output_names_cstr.size());

        if (output_tensors.empty()) {
            std::cerr << "Failed to get output tensors" << std::endl;
            return {};
        }

        // Get output data
        float* output_data = output_tensors[0].GetTensorMutableData<float>();
        if (!output_data) {
            std::cerr << "Failed to get output data" << std::endl;
            return {};
        }

        // Get the actual output size
        auto output_tensor_info = output_tensors[0].GetTensorTypeAndShapeInfo();
        std::vector<int64_t> output_shape = output_tensor_info.GetShape();
        
        size_t output_size = 1;
        for (int64_t dim : output_shape) {
            output_size *= static_cast<size_t>(dim);
        }

        // Copy output data to vector
        std::vector<float> output(output_data, output_data + output_size);
        return output;
        
    } catch (const Ort::Exception& e) {
        std::cerr << "ONNX exception during inference: " << e.what() << std::endl;
        return {};
    } catch (const std::exception& e) {
        std::cerr << "Exception during inference: " << e.what() << std::endl;
        return {};
    }
}

int ONNXModelHandler::getInputFeatureCount() const {
    return _input_feature_count;
}

int ONNXModelHandler::getOutputFeatureCount() const {
    return _output_feature_count;
}

} // namespace ipl
