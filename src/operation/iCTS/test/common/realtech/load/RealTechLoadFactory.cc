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
/**
 * @file RealTechLoadFactory.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 * @date 2026-04-11
 * @brief Load-construction helpers for real-tech and synthetic-stand-in tests.
 */

#include "common/realtech/load/RealTechLoadFactory.hh"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "IdbDesign.h"
#include "IdbGeometry.h"
#include "IdbNet.h"
#include "IdbPins.h"
#include "common/data/TestDataGenerator.hh"
#include "common/dataset/TestDataset.hh"
#include "database/design/Inst.hh"
#include "database/design/Pin.hh"
#include "database/spatial/Point.hh"
#include "idm.h"

namespace icts_test::common::realtech::load {
namespace {

constexpr int kSyntheticLoadCanvasWidth = 180000;
constexpr int kSyntheticLoadCanvasHeight = 140000;

auto PickBestNetName(std::string& net_name, bool& is_clock_net) -> bool
{
  auto* idb_design = dmInst->get_idb_design();
  if (idb_design == nullptr || idb_design->get_net_list() == nullptr) {
    return false;
  }

  const auto& nets = idb_design->get_net_list()->get_net_list();
  std::size_t best_clock_loads = 0;
  std::size_t best_any_loads = 0;
  std::string best_clock_name;
  std::string best_any_name;

  for (auto* net : nets) {
    if (net == nullptr) {
      continue;
    }
    const auto load_count = net->get_load_pins().size();
    if (load_count < 2) {
      continue;
    }
    if (net->is_clock() && load_count > best_clock_loads) {
      best_clock_loads = load_count;
      best_clock_name = net->get_net_name();
    }
    if (load_count > best_any_loads) {
      best_any_loads = load_count;
      best_any_name = net->get_net_name();
    }
  }

  if (!best_clock_name.empty()) {
    net_name = best_clock_name;
    is_clock_net = true;
    return true;
  }
  if (!best_any_name.empty()) {
    net_name = best_any_name;
    is_clock_net = false;
    return true;
  }
  return false;
}

}  // namespace

auto MakeRealDesignLoads(std::size_t target_count, std::string& source_label, unsigned seed) -> GeneratedPins
{
  GeneratedPins result;
  if (target_count == 0) {
    return result;
  }

  auto* idb_design = dmInst->get_idb_design();
  auto* net_list = idb_design != nullptr ? idb_design->get_net_list() : nullptr;
  if (idb_design == nullptr || net_list == nullptr) {
    return result;
  }

  std::string selected_net_name;
  bool is_clock_net = false;
  if (!PickBestNetName(selected_net_name, is_clock_net)) {
    return result;
  }

  auto* selected_net = net_list->find_net(selected_net_name);
  if (selected_net == nullptr) {
    return result;
  }

  std::vector<idb::IdbPin*> valid_candidate_pins;
  for (auto* idb_pin : selected_net->get_load_pins()) {
    if (idb_pin == nullptr || idb_pin->get_average_coordinate() == nullptr) {
      continue;
    }
    valid_candidate_pins.push_back(idb_pin);
  }
  if (valid_candidate_pins.size() < 2) {
    return result;
  }

  const std::size_t sample_count = std::min(target_count, valid_candidate_pins.size());
  std::vector<std::size_t> sample_index(valid_candidate_pins.size());
  std::iota(sample_index.begin(), sample_index.end(), 0);
  std::mt19937 generator(seed);
  std::shuffle(sample_index.begin(), sample_index.end(), generator);
  if (sample_index.size() > sample_count) {
    sample_index.resize(sample_count);
  }
  std::ranges::sort(sample_index);

  result.storage.reserve(sample_index.size());
  result.loads.reserve(sample_index.size());

  int min_x = std::numeric_limits<int>::max();
  int min_y = std::numeric_limits<int>::max();
  int max_x = std::numeric_limits<int>::min();
  int max_y = std::numeric_limits<int>::min();

  for (std::size_t index = 0; index < sample_index.size(); ++index) {
    auto* idb_pin = valid_candidate_pins.at(sample_index.at(index));
    auto* coord = idb_pin->get_average_coordinate();
    if (coord == nullptr) {
      return {};
    }

    const int x_coord = coord->get_x();
    const int y_coord = coord->get_y();
    min_x = std::min(min_x, x_coord);
    min_y = std::min(min_y, y_coord);
    max_x = std::max(max_x, x_coord);
    max_y = std::max(max_y, y_coord);

    std::ostringstream pin_name;
    pin_name << selected_net_name << "_" << idb_pin->get_pin_name() << "_" << index;
    auto pin = std::make_unique<icts::Pin>(pin_name.str(), icts::PinType::kClock, icts::Point<int>(x_coord, y_coord));
    result.loads.push_back(pin.get());
    result.storage.push_back(std::move(pin));
  }

  if (result.loads.empty()) {
    return {};
  }

  result.width = std::max(1, max_x - min_x);
  result.height = std::max(1, max_y - min_y);
  source_label = (is_clock_net ? "clock_net:" : "net:") + selected_net_name;
  return result;
}

auto MakeSyntheticLoads(std::size_t target_count, std::string& source_label, unsigned seed) -> GeneratedPins
{
  source_label = "synthetic_standin:gaussian_mixture";
  return data::MakeGaussianMixture(target_count, {.width = kSyntheticLoadCanvasWidth, .height = kSyntheticLoadCanvasHeight}, seed);
}

auto CountPinsWithExactCapContext(const std::vector<icts::Pin*>& loads) -> std::size_t
{
  return static_cast<std::size_t>(std::ranges::count_if(loads, [](const icts::Pin* pin) -> bool {
    return pin != nullptr && pin->get_inst() != nullptr && pin->get_net() != nullptr && !pin->get_name().empty()
           && !pin->get_inst()->get_name().empty();
  }));
}

}  // namespace icts_test::common::realtech::load
