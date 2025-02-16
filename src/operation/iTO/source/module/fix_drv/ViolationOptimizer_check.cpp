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
#include "Master.h"
#include "Placer.h"
#include "Reporter.h"
#include "ToConfig.h"
#include "ViolationOptimizer.h"
#include "api/TimingEngine.hh"
#include "api/TimingIDBAdapter.hh"
#include "timing_engine.h"

namespace ito {
bool ViolationOptimizer::isNeedRepair(ista::Net* net, double& cap_load_allowed_max)
{
  auto driver = net->getDriver();
  if (driver == nullptr) {
    return false;
  }

  // check if exit cap violation
  bool is_cap_vio = checkCapacitanceViolation(cap_load_allowed_max, driver);

  // check if exit slew violation
  bool is_slew_vio = checkSlewViolation(cap_load_allowed_max, driver);
  if (is_slew_vio) {
    cap_load_allowed_max *= _slew_2_cap_factor;  // 0.05;
  }

  return (is_cap_vio || is_slew_vio);
}

bool ViolationOptimizer::checkCapacitanceViolation(double& max_driver_load_cap, DesignObject* driver_pin)
{
  double cap_load;
  std::optional<double> exit_cap_restraint_max;
  double cap_slack;

  TransType rf = TransType::kRise;
  timingEngine->get_sta_engine()->validateCapacitance(driver_pin->getFullName().c_str(), AnalysisMode::kMax, rf, cap_load,
                                                      exit_cap_restraint_max, cap_slack);
  if (exit_cap_restraint_max) {
    max_driver_load_cap = *exit_cap_restraint_max;
    if (cap_slack < 0) {
      _number_cap_violation_net++;
      return true;
    }
  }
  return false;
}

void ViolationOptimizer::checkViolations()
{
  _number_slew_violation_net = 0;
  _number_cap_violation_net = 0;
  timingEngine->get_sta_engine()->updateTiming();

  Netlist* design_nl = timingEngine->get_sta_engine()->get_netlist();
  ista::Net* net;
  FOREACH_NET(design_nl, net)
  {
    if (net->isClockNet() || netConnectToPort(net)) {
      continue;
    }

    double cap_load_allowed_max = kInf;
    if (isNeedRepair(net, cap_load_allowed_max)) {
      _violation_nets_map[net] = cap_load_allowed_max;
    }
  }
#ifdef REPORT_TO_TXT
  toRptInst->reportDRVResult(_number_slew_violation_net, _number_cap_violation_net, false);
#endif
}

bool ViolationOptimizer::checkSlewViolation(double& max_driver_load_cap, DesignObject* driver_pin)
{
  float slew_margin = kInf;
  float slew_constraint_max = kInf;

  Net* net = driver_pin->get_net();
  DesignObject* pin;
  FOREACH_NET_PIN(net, pin)
  {
    double slew_tmp;
    std::optional<double> exit_slew_max_restraint;
    double slack_tmp;

    timingEngine->get_sta_engine()->validateSlew(pin->getFullName().c_str(), AnalysisMode::kMax, TransType::kRise, slew_tmp,
                                                 exit_slew_max_restraint, slack_tmp);
    if (exit_slew_max_restraint && slack_tmp < slew_margin) {
      slew_margin = slack_tmp;
      slew_constraint_max = *exit_slew_max_restraint;
    }
  }
  if (slew_margin < 0.0) {
    LibPort* driver_port = nullptr;
    if (driver_pin->isPin()) {
      driver_port = dynamic_cast<Pin*>(driver_pin)->get_cell_port();
    }

    if (driver_port) {
      // Identify the maximum load capacitance that corresponds to the maximum slew.
      double slew_2_cap = calcLoadCap(driver_port, slew_constraint_max);
      max_driver_load_cap = min(max_driver_load_cap, slew_2_cap);
      _number_slew_violation_net++;
      return true;
    }
  }
  return false;
}

/**
 * @brief Calculata the output port load capacitance that causes the current
 * slew.
 *
 * @param driver_port
 * @param slew
 * @return double
 */
double ViolationOptimizer::calcLoadCap(ista::LibPort* driver_port, double slew)
{
  double lower_cap = 0.0;
  double upper_cap = slew / driver_port->driveResistance() * 2.0;

  // 扩展搜索范围，确保初始值包含零差异点
  auto slew_diff = calcSlew(driver_port, upper_cap) - slew;
  while (slew_diff < 0.0) {
    lower_cap = upper_cap;
    upper_cap *= 2.0;
    slew_diff = calcSlew(driver_port, upper_cap) - slew;
  }

  // 二分法查找最佳电容值
  while (abs(lower_cap - upper_cap) > max(lower_cap, upper_cap) * 0.01) {
    double mid_cap = (lower_cap + upper_cap) / 2.0;
    double diff_mid = calcSlew(driver_port, mid_cap) - slew;

    if (diff_mid < 0.0) {
      lower_cap = mid_cap;
    } else {
      upper_cap = mid_cap;
    }
  }

  // 返回接近零差异点的电容值
  return (lower_cap + upper_cap) / 2.0;
}

double ViolationOptimizer::calcSlew(LibPort* driver_port, double cap_load)
{
  TOSlew rise_fall_slew[2];
  timingEngine->calcGateRiseFallSlews(rise_fall_slew, cap_load, driver_port);
  TOSlew gate_slew = max(rise_fall_slew[TYPE_RISE], rise_fall_slew[TYPE_FALL]);
  return gate_slew;
}

/**
 * @brief net's loads contain a port.
 *
 * @param net
 * @return true
 * @return false
 */
bool ViolationOptimizer::netConnectToPort(ista::Net* net)
{
  auto load_pin_ports = net->getLoads();
  for (auto pin_port : load_pin_ports) {
    if (pin_port->isPort()) {
      return true;
    }
  }
  return false;
}

int ViolationOptimizer::portFanoutLoadNum(ista::LibPort* port)
{
  auto& fanout_load = port->get_fanout_load();
  if (!fanout_load.has_value()) {
    ista::LibCell* cell = port->get_ower_cell();
    ista::LibLibrary* lib = cell->get_owner_lib();
    fanout_load = lib->get_default_fanout_load();
  }
  if (fanout_load) {
    return *fanout_load;
  } else {
    return 1;
  }
}

}  // namespace ito