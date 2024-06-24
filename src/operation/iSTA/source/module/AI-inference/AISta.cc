/**
 * @file AISta.cc
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The top layer for deploying the AI model used in sta.
 * @version 0.1
 * @date 2024-06-06
 *
 * @copyright Copyright (c) 2024
 *
 */
#include <array>
#include <fstream>

#include "AISta.hh"
#include "sta/StaArc.hh"

namespace ista {

/**
 * @brief for AI calibrate task, we need tokenization for cell name and pin
 * name.
 *
 * @param file_path
 * @return std::map<std::string, unsigned>
 */
std::map<std::string, unsigned> AICalibratePathDelay::tokenization(
    std::string file_path) {
  std::map<std::string, unsigned> token_map;
  // read file, each line is a token, then add it to the token map, id plus
  // one.
  std::ifstream file(file_path);
  while (!file.eof()) {
    std::string token;
    file >> token;
    token_map.insert(std::make_pair(token, token_map.size() + 1));
  }

  return token_map;
}

/**
 * @brief init the AI calibrate
 *
 * @return unsigned
 */
unsigned AICalibratePathDelay::init() {
  _cell_to_id = tokenization(_cell_list_path);
  _pin_to_id = tokenization(_pin_list_path);

  return 1;
}

/**
 * @brief preprocess data for AI calibrate task
 *
 * @param seq_path_data
 */
auto AICalibratePathDelay::preprocessData(StaSeqPathData* seq_path_data) {
  std::map<std::string, std::array<float, MAX_SEQ_LEN>> feature_vecs{
      {"pin_name", {}},  {"cell_name", {}},  {"fanout", {}},
      {"rise_fall", {}}, {"is_net", {}},     {"capacitance", {}},
      {"slew", {}},      {"incr_delay", {}}, {"arrive_time", {}}};

  std::stack<StaPathDelayData*> path_stack = seq_path_data->getPathDelayData();

  double launch_edge = FS_TO_NS(seq_path_data->getLaunchEdge());
  auto* path_delay_data = path_stack.top();
  auto* launch_clock_data = path_delay_data->get_launch_clock_data();
  auto launch_network_time = FS_TO_NS(launch_clock_data->get_arrive_time());
  double clock_path_arrive_time = launch_edge + launch_network_time;

  double last_arrive_time = 0;
  StaVertex* last_vertex = nullptr;
  unsigned vertex_index = 0;
  while (!path_stack.empty()) {
    auto* path_delay_data = path_stack.top();
    auto* own_vertex = path_delay_data->get_own_vertex();
    auto trans_type = path_delay_data->get_trans_type();
    auto* obj = own_vertex->get_design_obj();

    // for net node.
    if (last_vertex &&
        ((obj->isPin() && obj->isInput() && !own_vertex->is_assistant()) ||
         (obj->isPort() && obj->isOutput() && own_vertex->is_assistant()))) {
      auto snk_arcs = last_vertex->getSnkArc(own_vertex);
      LOG_FATAL_IF(snk_arcs.size() != 1)
          << last_vertex->getName() << " " << own_vertex->getName()
          << " net arc found " << snk_arcs.size() << " arc.";
      if (snk_arcs.size() == 1) {
        auto* net_arc = dynamic_cast<StaNetArc*>(snk_arcs.front());
        auto* net = net_arc->get_net();
        unsigned fanout_num = net->getFanouts();

        feature_vecs["pin_name"][vertex_index] = _pin_to_id.at("net");
        feature_vecs["cell_name"][vertex_index] = _cell_to_id.at("net");
        feature_vecs["fanout"][vertex_index] = fanout_num;
        feature_vecs["rise_fall"][vertex_index] =
            3;  // hard code net rise fall with 3.
        feature_vecs["is_net"][vertex_index] = 1;
        ++vertex_index;
      }
    }

    // for cell node.
    auto* pin_name = obj->get_name();
    auto* cell_name = obj->get_own_instance()->get_inst_cell()->get_cell_name();
    std::string cell_pin_name = Str::printf("%s:%s", cell_name, pin_name);
    feature_vecs["pin_name"][vertex_index] = _pin_to_id.at(cell_pin_name);
    if (obj->get_own_instance()) {
      auto* cell_name =
          obj->get_own_instance()->get_inst_cell()->get_cell_name();
      feature_vecs["cell_name"][vertex_index] = _cell_to_id.at(cell_name);
    }

    if (own_vertex->is_clock()) {
      trans_type =
          own_vertex->isRisingTriggered() ? TransType::kRise : TransType::kFall;
    }

    feature_vecs["rise_fall"][vertex_index] =
        (trans_type == TransType::kRise) ? 1 : 2;

    auto arrive_time = FS_TO_NS(path_delay_data->get_arrive_time());
    auto incr_time = arrive_time - last_arrive_time;
    last_arrive_time = arrive_time;

    auto vertex_load =
        own_vertex->getLoad(path_delay_data->get_delay_type(), trans_type);
    auto vertex_slew =
        own_vertex->getSlewNs(path_delay_data->get_delay_type(), trans_type);

    feature_vecs["capacitance"][vertex_index] = vertex_load;
    feature_vecs["slew"][vertex_index] = vertex_slew.value_or(0.0);
    feature_vecs["incr_delay"][vertex_index] = incr_time;
    feature_vecs["arrive_time"][vertex_index] =
        arrive_time + clock_path_arrive_time;

    last_vertex = own_vertex;
    path_stack.pop();
    ++vertex_index;
  }

  return feature_vecs;
}

/**
 * @brief create input tensor for AI calibration task.
 *
 * @param seq_path_data
 */
Ort::Value AICalibratePathDelay::createInputTensor(
    StaSeqPathData* seq_path_data) {
  auto feature_vecs = preprocessData(seq_path_data);
  std::vector<std::string> feature_names = {
      "pin_name",    "cell_name", "fanout",     "rise_fall",  "is_net",
      "capacitance", "slew",      "incr_delay", "arrive_time"};

  std::vector<float> input_data;
  for (auto& feature_name : feature_names) {
    std::array<float, MAX_SEQ_LEN> feature_vec = feature_vecs[feature_name];
    for (const auto feature_value : feature_vec) {
      input_data.push_back(feature_value);
    }
  }

  std::vector<int64_t> input_dims = {1,
                                     static_cast<int64_t>(input_data.size())};

  Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
      Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault),
      input_data.data(), input_data.size(), input_dims.data(),
      input_dims.size());

  return input_tensor;
}

/**
 * @brief Infer the AI calibration task.
 *
 * @param input_tensor
 */
std::vector<Ort::Value> AICalibratePathDelay::infer(Ort::Value& input_tensor) {
  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "AISta");
  Ort::SessionOptions session_options;
  std::string model_path = _model_to_path[_model_type];
  Ort::Session session(env, model_path.c_str(), session_options);

  Ort::AllocatorWithDefaultOptions ort_alloc;
  Ort::AllocatedStringPtr input_name =
      session.GetInputNameAllocated(0, ort_alloc);
  Ort::AllocatedStringPtr output_name =
      session.GetOutputNameAllocated(0, ort_alloc);
  const char* input_names[] = {input_name.get()};
  char* output_names[] = {output_name.get()};
  std::vector<Ort::Value> output_tensors = session.Run(
      Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);

  return output_tensors;
}

/**
 * @brief Get the output result of infer.
 *
 * @param output_tensor
 * @return std::vector<float>
 */
std::vector<float> AICalibratePathDelay::getOutputResult(
    std::vector<Ort::Value>& output_tensors) {
  std::vector<float> result;

  for(auto& output_tensor: output_tensors) {
      float* output = output_tensor.GetTensorMutableData<float>();
      result.emplace_back(*output);  
  }
  
  return result;
}

}  // namespace ista