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

#include "ViolationOptimizer.h"

#include "Master.h"
#include "Placer.h"
#include "Reporter.h"
#include "ToConfig.h"
#include "api/TimingEngine.hh"
#include "api/TimingIDBAdapter.hh"
#include "timing_engine.h"

#include <algorithm>

namespace ito {

ViolationOptimizer* ViolationOptimizer::_instance = nullptr;

void ViolationOptimizer::fixViolations()
{
  // step 1. init
  init();
  int begin_buffer_num = toDmInst->get_buffer_num();
  int begin_resize_num = toDmInst->get_resize_num();

  timingEngine->set_eval_data();

  // step 2. check and repair
  checkAndRepair();

  // step 2.1. check if there are still violations and repair them
  iterCheckAndRepair();

  // step 3. overview report if need
#ifdef REPORT_TO_TXT
  toRptInst->get_ofstream() << "\nTO: Total insert " << toDmInst->get_buffer_num() - begin_buffer_num << " buffers when fix DRV."
                            << "\nTO: Total resize " << toDmInst->get_resize_num() - begin_resize_num << " instances when fix DRV.\n";
  toRptInst->reportTime(false);
#endif
}

void ViolationOptimizer::fixSpecialNet(const char* net_name)
{
  // step 1. init
  init();

  Netlist* design_nl = timingEngine->get_sta_engine()->get_netlist();
  ista::Net* net = design_nl->findNet(net_name);

  LOG_ERROR_IF(!net) << "Cannot find net: " << net_name << " in design.\n";

  double cap_load_allowed_max = kInf;
  while (isNeedRepair(net, cap_load_allowed_max)) {
    optimizeViolationNet(net, cap_load_allowed_max);
    _slew_2_cap_factor *= 0.5;
  }
  timingEngine->get_sta_engine()->updateTiming();
  if (isNeedRepair(net, cap_load_allowed_max)) {
    toRptInst->get_ofstream() << "Failed optimize DRV in NET: " << net_name << endl;
    toRptInst->get_ofstream().close();
  }
}

void ViolationOptimizer::checkAndRepair()
{
  int number_driver_vertices = timingEngine->get_driver_vertices().size();
  TOVertexSeq sorted_driver_vertices = timingEngine->get_driver_vertices();

  int net_connect_port = 0;

  for (int i = number_driver_vertices - 1; i >= 0; --i) {
    if ((number_driver_vertices - i) % 1000 == 0 || i == 0) {
      LOG_INFO << "The 1-th Check and repair: " << (number_driver_vertices - i) << "/" << number_driver_vertices << "("
               << (double(number_driver_vertices - i) / number_driver_vertices) * 100 << "%) nets\n";
    }
    StaVertex* driver = sorted_driver_vertices[i];

    auto* design_obj = driver->get_design_obj();
    auto* net = design_obj->get_net();

    // do not fix clock net
    if (net->isClockNet()) {
      continue;
    }

    if (netConnectToPort(net)) {
      net_connect_port++;
      continue;
    }

    if (driver->is_clock() || driver->is_const()) {
      continue;
    }

    double cap_load_allowed_max = kInf;
    if (isNeedRepair(net, cap_load_allowed_max)) {
      optimizeViolationNet(net, cap_load_allowed_max);
    }
  }

#ifdef REPORT_TO_TXT
  toRptInst->reportDRVResult(_number_slew_violation_net, _number_cap_violation_net, true);
#endif
}

void ViolationOptimizer::iterCheckAndRepair()
{
  int max_iter = toConfig->get_drv_optimize_iter_number();
  int iter = 1;
  if (iter == max_iter) {
    return;
  }

  checkViolations();

  int prev_violation_num = _violation_nets_map.size();
  while (!_violation_nets_map.empty()) {
    // If there are still a violation nets, the secondary fix is performed.
    int which_id = 1;
    for (auto [net, cap_load_allowed_max] : _violation_nets_map) {
      int batch_size = std::max(prev_violation_num / 100, 10);
      if (which_id % batch_size == 0 || which_id == prev_violation_num) {
        LOG_INFO << "The " << iter + 1 << "-th Check and repair: " << which_id << "/" << prev_violation_num << "("
                 << (double(which_id) / prev_violation_num) * 100 << "%) nets\n";
      }
      which_id++;
      optimizeViolationNet(net, cap_load_allowed_max);
    }
    int last_violation_num = _violation_nets_map.size();
    _violation_nets_map.clear();
    checkViolations();

    dereaseSlewCapFactor(prev_violation_num, last_violation_num);
    prev_violation_num = last_violation_num;
    if (++iter >= max_iter) {
      break;
    }
  }
}

void ViolationOptimizer::dereaseSlewCapFactor(int prev_violation_num, int last_violation_num) {
  auto getHighestDigit = [](int number) {
    int digits = static_cast<int>(log10(number));
    int divisor = static_cast<int>(pow(10, digits));
    int highestDigit = number / divisor;
    return highestDigit;
  };

  double factor = (prev_violation_num - last_violation_num + 1) / static_cast<double>(getHighestDigit(prev_violation_num));
  factor = std::clamp(factor, 0.3, 1.0);
  _slew_2_cap_factor *= factor;
}

}  // namespace ito