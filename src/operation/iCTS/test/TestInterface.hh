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
 * @file TestInterface.hh
 * @author Dawn Li (dawnli619215645@gmail.com)
 */

#include <random>

#include "../../platform/data_manager/idm.h"
#include "CTSAPI.hh"
#include "LocalLegalization.hh"
#include "TimingPropagator.hh"
#include "TreeBuilder.hh"
namespace {
using icts::Inst;
using icts::LocalLegalization;
using icts::Pin;
using icts::Point;
using icts::RCPattern;
using icts::TimingPropagator;
using icts::TreeBuilder;

struct EnvInfo
{
  int min_x;
  int max_x;
  int min_y;
  int max_y;
  size_t min_num;
  size_t max_num;
  double min_cap;
  double max_cap;
  double min_delay;
  double max_delay;
};

class TestInterface
{
 public:
  TestInterface(const std::string& db_config_path, const std::string& cts_config_path)
  {
    if (db_config_path.empty() && cts_config_path.empty()) {
      return;
    }
    dmInst->init(db_config_path);
    CTSAPIInst.init(cts_config_path);
    LocalLegalization::setIgnoreCore(true);
  }
  virtual ~TestInterface() = default;

 protected:
  std::vector<Inst*> genRandomBuffers(const EnvInfo& env_info, const int& seed = 0) const
  {
    std::random_device rd;
    std::mt19937 gen(static_cast<std::mt19937::result_type>(seed));
    std::set<Point> locs;
    size_t pin_num = std::uniform_int_distribution<>(env_info.min_num, env_info.max_num)(gen);
    while (locs.size() < pin_num) {
      auto x = std::uniform_int_distribution<>(env_info.min_x / 1000, env_info.max_x / 1000)(gen) * 1000;
      auto y = std::uniform_int_distribution<>(env_info.min_y / 1000, env_info.max_y / 1000)(gen) * 1000;
      locs.insert(Point(x, y));
    }
    std::vector<Inst*> load_bufs;
    size_t i = 0;
    for (auto loc : locs) {
      auto* buf = TreeBuilder::genBufInst(CTSAPIInst.toString("buf_", i++), loc);
      buf->set_cell_master(TimingPropagator::getMinSizeCell());
      load_bufs.push_back(buf);
      auto* load_pin = buf->get_load_pin();
      auto pattern = static_cast<RCPattern>(1 + std::rand() % 2);
      load_pin->set_pattern(pattern);
      TimingPropagator::updatePinCap(load_pin);
      TimingPropagator::initLoadPinDelay(load_pin);
    }
    if (env_info.min_delay > 0 && env_info.max_delay > 0 && env_info.max_delay >= env_info.min_delay) {
      std::ranges::for_each(load_bufs, [&](Inst* buf) {
        auto* driver_pin = buf->get_driver_pin();
        auto delay = std::uniform_real_distribution<>(env_info.min_delay, env_info.max_delay)(gen);
        driver_pin->set_min_delay(delay);
        driver_pin->set_max_delay(delay);
      });
    }
    if (env_info.min_cap > 0 && env_info.max_cap > 0 && env_info.max_cap >= env_info.min_cap) {
      std::ranges::for_each(load_bufs, [&](Inst* buf) {
        auto* driver_pin = buf->get_driver_pin();
        auto cap = std::uniform_real_distribution<>(env_info.min_cap, env_info.max_cap)(gen);
        auto* temp_inst = TreeBuilder::genBufInst("temp", buf->get_location());
        driver_pin->add_child(temp_inst->get_load_pin());
        driver_pin->set_cap_load(cap);
        auto* load_pin = buf->get_load_pin();
        TimingPropagator::initLoadPinDelay(load_pin);
      });
    }
    return load_bufs;
  }

  std::vector<Pin*> genRandomPins(const EnvInfo& env_info, const int& seed = 0) const
  {
    auto load_bufs = genRandomBuffers(env_info, seed);
    std::vector<Pin*> load_pins;
    std::ranges::transform(load_bufs, std::back_inserter(load_pins), [](Inst* buf) { return buf->get_load_pin(); });
    return load_pins;
  }
};
}  // namespace
