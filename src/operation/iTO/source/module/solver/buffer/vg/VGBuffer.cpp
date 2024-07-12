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

#include "VGBuffer.h"

#include "data_manager.h"

namespace ito {

BufferedOptionSeq VGBuffer::VGBuffering(TreeBuild* tree) {
  LOG_ERROR_IF(_available_lib_cell_sizes.empty()) << "No buffer cell sizes available for VGBuffering";
  int driver_id = tree->get_root()->get_id();
  tree->updateBranch();
  return findBufferSolution(tree, tree->left(driver_id), driver_id);
}

BufferedOptionSeq VGBuffer::findBufferSolution(TreeBuild* tree, int curr_id, int prev_id)
{
  if (curr_id == TreeBuild::_null_pt) {
    return {};
  }
  auto* obj_pin = tree->get_pin(curr_id);
  Point curr_loc = tree->get_location(curr_id);
  Point prev_loc = tree->get_location(prev_id);

  if (obj_pin) {
    auto* pin = dynamic_cast<Pin*>(obj_pin);

    if (timingEngine->get_sta_engine()->isLoad(pin->getFullName().c_str())) {
      BufferedOption* buffered_option = new BufferedOption(BufferedOptionType::kSink);
      StaVertex* vertex = timingEngine->get_sta_engine()->findVertex(pin->getFullName().c_str());
      auto req_ns_r = vertex->getReqTimeNs(AnalysisMode::kMax, TransType::kRise);
      double req_r = req_ns_r ? *req_ns_r : 0.0;
      auto req_ns_f = vertex->getReqTimeNs(AnalysisMode::kMax, TransType::kFall);
      double req_f = req_ns_f ? *req_ns_f : 0.0;
      double req = min(req_r, req_f);

      BufferedOptionSeq buf_option_seq;
      buffered_option->set_location(curr_loc);
      buffered_option->set_cap(pin->cap());
      buffered_option->set_pin_loaded(pin);
      buffered_option->set_req(req);
      buf_option_seq.emplace_back(buffered_option);

      buf_option_seq = addWire(buf_option_seq, curr_loc, prev_loc);
      // Determine if a buffer needs to be inserted after adding wire
      if (!buf_option_seq.empty()) {
        BufferedOptionSeq buf_options = addBuffer(buf_option_seq, prev_loc);
        for (BufferedOption* buf_opt : buf_options) {
          buf_option_seq.push_back(buf_opt);
        }
        buf_option_seq.insert(buf_option_seq.end(), buf_options.begin(), buf_options.end());
      }
      return buf_option_seq;
    }
  }
  // curr -> steiner point
  else if (obj_pin == nullptr) {
    BufferedOptionSeq buf_opt_left = findBufferSolution(tree, tree->left(curr_id), curr_id);
    BufferedOptionSeq buf_opt_mid = findBufferSolution(tree, tree->middle(curr_id), curr_id);

    BufferedOptionSeq buf_opt_merger = mergeBranch(buf_opt_left, buf_opt_mid, curr_loc);

    buf_opt_merger = addWire(buf_opt_merger, curr_loc, prev_loc);
    // Determine if a buffer needs to be inserted after adding wire
    if (!buf_opt_merger.empty()) {
      BufferedOptionSeq buf_options = addBuffer(buf_opt_merger, prev_loc);
      buf_opt_merger.insert(buf_opt_merger.end(), buf_options.begin(), buf_options.end());
    }
    return buf_opt_merger;
  }

  return {};
}

BufferedOptionSeq VGBuffer::mergeBranch(BufferedOptionSeq buf_opt_left, BufferedOptionSeq buf_opt_right, Point curr_loc)
{
  BufferedOptionSeq buf_opt_merger;
  for (auto left : buf_opt_left) {
    for (auto right : buf_opt_right) {
      BufferedOption* buffered_option = new BufferedOption(BufferedOptionType::kBranch);
      TORequired left_req = left->get_required_arrival_time();
      float left_cap = left->get_cap();
      TORequired right_req = right->get_required_arrival_time();
      float right_cap = right->get_cap();

      BufferedOption* min_opt = approximatelyLess(left_req, right_req) ? left : right;

      buffered_option->set_location(curr_loc);
      buffered_option->set_cap(left_cap + right_cap);
      buffered_option->set_delay_required(min_opt->get_delay_required());
      buffered_option->set_left(left);
      buffered_option->set_right(right);
      buffered_option->set_req(min_opt->get_req());
      buf_opt_merger.emplace_back(buffered_option);
    }
  }
  sort(buf_opt_merger.begin(), buf_opt_merger.end(), [=](BufferedOption* opt1, BufferedOption* opt2) {
    return approximatelyGreater(opt1->get_required_arrival_time(), opt2->get_required_arrival_time());
  });
  size_t i = 0;
  while (i < buf_opt_merger.size()) {
    BufferedOption* buffer_option_1 = buf_opt_merger[i];
    float buffer_option_1_cap = buffer_option_1->get_cap();
    size_t j = i + 1;

    while (j < buf_opt_merger.size()) {
      BufferedOption* buffer_option_2 = buf_opt_merger[j];
      float buffer_option_2_cap = buffer_option_2->get_cap();

      if (approximatelyLess(buffer_option_2_cap, buffer_option_1_cap)) {
        j++;
      } else {
        buf_opt_merger.erase(buf_opt_merger.begin() + j);
      }
    }
    i++;
  }
  return buf_opt_merger;
}

BufferedOptionSeq VGBuffer::addWire(BufferedOptionSeq buf_opt_seq, Point curr_loc, Point prev_loc)
{
  int wire_length = abs(curr_loc.get_x() - prev_loc.get_x()) + abs(curr_loc.get_y() - prev_loc.get_y());
  std::optional<double> width = std::nullopt;
  double wire_length_cap = timingEngine->get_sta_adapter()->getCapacitance(1, (double) wire_length / toDmInst->get_dbu(), width);
  double wire_length_res = timingEngine->get_sta_adapter()->getResistance(1, (double) wire_length / toDmInst->get_dbu(), width);
  // double wire_delay = wire_length_res * wire_length_cap;

  BufferedOptionSeq buf_option_seq;
  for (BufferedOption* buf_opt : buf_opt_seq) {
    BufferedOption* buffered_option = new BufferedOption(BufferedOptionType::kWire);
    double wire_delay = wire_length_res * (wire_length_cap / 2 + buf_opt->get_cap());

    float update_cap = buf_opt->get_cap() + wire_length_cap;
    TODelay update_req_delay = buf_opt->get_delay_required() + wire_delay;

    buffered_option->set_location(prev_loc);
    buffered_option->set_cap(update_cap);
    buffered_option->set_delay_required(update_req_delay);
    buffered_option->set_left(buf_opt);
    buffered_option->set_req(buf_opt->get_req());

    buf_option_seq.push_back(buffered_option);
  }
  return buf_option_seq;
}

BufferedOptionSeq VGBuffer::addBuffer(BufferedOptionSeq buf_opt_seq, Point prev_loc)
{
  BufferedOptionSeq new_options;

  for (LibCell* buffer_cell : _available_lib_cell_sizes) {
    std::vector<std::pair<TORequired, BufferedOption*>> valid_options;

    for (BufferedOption* option : buf_opt_seq) {
      TODelay delay = timingEngine->calcSetupDelayOfBuffer(option->get_cap(), buffer_cell);
      TORequired req_time = option->get_required_arrival_time() - delay;
      valid_options.emplace_back(req_time, option);
    }

    auto max_req_option
        = std::max_element(valid_options.begin(), valid_options.end(), [](const auto& a, const auto& b) { return a.first < b.first; });

    if (max_req_option != valid_options.end()) {
      auto& [max_req_time, best_option] = *max_req_option;

      LibPort *input_buffer_port, *output_buffer_port;
      buffer_cell->bufferPorts(input_buffer_port, output_buffer_port);
      float input_buffer_port_cap = input_buffer_port->get_port_cap();
      TODelay delay = timingEngine->calcSetupDelayOfBuffer(best_option->get_cap(), buffer_cell);
      TORequired req_time = best_option->get_required_arrival_time() - delay;

      if (std::none_of(new_options.begin(), new_options.end(), [input_buffer_port_cap, req_time](BufferedOption* option) {
            return option->get_cap() <= input_buffer_port_cap && option->get_required_arrival_time() >= req_time;
          })) {
        TODelay updated_req_delay = best_option->get_delay_required() + delay;
        auto buffered_option = new BufferedOption(BufferedOptionType::kBuffer);
        buffered_option->set_location(prev_loc);
        buffered_option->set_cap(input_buffer_port_cap);
        buffered_option->set_delay_required(updated_req_delay);
        buffered_option->set_lib_cell_size(buffer_cell);
        buffered_option->set_left(best_option);
        buffered_option->set_req(best_option->get_req());
        new_options.push_back(buffered_option);
      }
    }
  }
  return new_options;
}

}  // namespace ito
