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
 * @file AnnotateData.cc
 * @author shaozheqing (707005020@qq.com)
 * @brief Annotate data implemention.
 * @version 0.1
 * @date 2023-01-11
 */

#include "AnnotateData.hh"

#include <ranges>

#include "log/Log.hh"

namespace ipower {

/**
 * @brief print t0,t1,tx,tz
 *
 * @param out
 */
void AnnotateTime::printAnnotateTime(std::ostream& out) {
  out << "(T0 " << _T0.get_ui() << ") ";
  out << "(T1 " << _T1.get_ui() << ") ";
  out << "(TX " << _TX.get_ui() << ") ";
  out << "(TZ " << _TZ.get_ui() << ") " << std::endl;
}

/**
 * @brief print toggle and glitch
 *
 * @param out
 */
void AnnotateToggle::printAnnotateToggle(std::ostream& out) {
  out << "(TC " << _TC.get_ui() << ") ";
  out << "(TB " << _TB.get_ui() << ") ";
  out << "(TG " << _TG.get_ui() << ") ";
  out << "(IG " << _IG.get_ui() << ") ";
  out << "(IK " << _IK.get_ui() << ") ";
}

/**
 * @brief Print each instance message by hierarchy
 *
 * @param out
 */
void AnnotateInstance::printAnnotateInstance(std::ostream& out) {
  out << "(INSTANCE " << _module_instance_name << std::endl;
  for (auto& signal : _signals) {
    signal.second->printAnnotateSignal(out);
  }

  for (auto& children_instance : _children_instances) {
    children_instance.second->printAnnotateInstance(out);
  }

  out << ") " << std::endl;
}

/**
 * @brief calculate Instances' TC and SP.
 *
 * @param duration
 */
void AnnotateInstance::calcInstancesTcSP(int64_t duration) {
  for (auto& [signal_name, signal] : _signals) {
    double tc_data;
    double sp_data;
    std::tie(tc_data, sp_data) = signal->get_signal_tc_sp();
    tc_data /= duration;

    auto signal_tc_sp = std::make_unique<AnnotateSignalToggleSPData>(
        signal_name, tc_data, sp_data);
    _signals_tc_sp.emplace_back(std::move(signal_tc_sp));
  }

  for (auto& children_instance : _children_instances) {
    children_instance.second->calcInstancesTcSP(duration);
  }
}

/**
 * @brief print annotate database
 *
 * @param out
 */
void AnnotateDB::printAnnotateDB(std::ostream& out) {
  out << "(DURATION " << _simulation_duration.get_ui() << ") " << std::endl;
  out << "(TIMESCALE " << _timescale.get_time_scale() << ") " << std::endl;

  _top_instance->printAnnotateInstance(out);
}

/**
 * @brief find the signal according the parent instances and signal name.
 *
 * @param parent_instance_names the instance names is from bottom to up, bottom
 * first.
 * @param signal_name
 * @return AnnotateRecord*
 */
AnnotateRecord* AnnotateDB::findSignalRecord(
    std::vector<std::string_view>& parent_instance_names,
    std::string_view signal_name) {
  auto* parent_instance = _top_instance.get();
  // reverse because top is in the last.
  for (auto& instance_name : parent_instance_names | std::views::reverse) {
    parent_instance = parent_instance->findInstance(instance_name);
  }
  LOG_FATAL_IF(!parent_instance) << "instance is not found.";
  auto* found_signal = parent_instance->findSignal(signal_name);
  LOG_FATAL_IF(!found_signal) << "signal is not found.";
  auto* signal_record = found_signal->get_record_data();
  return signal_record;
}

}  // namespace ipower
