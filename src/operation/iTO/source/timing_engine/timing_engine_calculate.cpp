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

#include "../utility/Reporter.h"
#include "ToConfig.h"
#include "data_manager.h"
#include "timing_engine.h"

namespace ito {

void ToTimingEngine::calcGateRiseFallDelays(TODelay rise_fall_delay[], float cap_load, LibPort* driver_port)
{
  for (int rf_index = 0; rf_index < 2; rf_index++) {
    rise_fall_delay[rf_index] = -kInf;
  }

  LibCell* cell = driver_port->get_ower_cell();
  // get all cell arcset
  auto& cell_arcset = cell->get_cell_arcs();
  for (auto& arcset : cell_arcset) {
    ieda::Vector<std::unique_ptr<ista::LibArc>>& arcs = arcset->get_arcs();
    for (auto& arc : arcs) {
      if (arc->isDelayArc()) {
        if ((arc->get_timing_type() == LibArc::TimingType::kComb) || (arc->get_timing_type() == LibArc::TimingType::kCombRise)
            || (arc->get_timing_type() == LibArc::TimingType::kRisingEdge)
            || (arc->get_timing_type() == LibArc::TimingType::kDefault)) {
          calGateRiseFallDelay(rise_fall_delay, cap_load, TransType::kRise, arc.get());
        }

        if ((arc->get_timing_type() == LibArc::TimingType::kComb) || (arc->get_timing_type() == LibArc::TimingType::kCombFall)
            || (arc->get_timing_type() == LibArc::TimingType::kFallingEdge)
            || (arc->get_timing_type() == LibArc::TimingType::kDefault)) {
          calGateRiseFallDelay(rise_fall_delay, cap_load, TransType::kFall, arc.get());
        }
      }
    }
  }
}

void ToTimingEngine::calcGateRiseFallSlews(TOSlew rise_fall_slew[], float cap_load, LibPort* driver_port)
{
  for (int rf_index = 0; rf_index < 2; rf_index++) {
    rise_fall_slew[rf_index] = -kInf;
  }

  LibCell* cell = driver_port->get_ower_cell();
  // get all cell arcset
  auto& cell_arcset = cell->get_cell_arcs();
  for (auto& arcset : cell_arcset) {
    ieda::Vector<std::unique_ptr<ista::LibArc>>& arcs = arcset->get_arcs();
    for (auto& arc : arcs) {
      if (arc->isDelayArc()) {
        if ((arc->get_timing_type() == LibArc::TimingType::kComb) || (arc->get_timing_type() == LibArc::TimingType::kCombRise)
            || (arc->get_timing_type() == LibArc::TimingType::kRisingEdge)
            || (arc->get_timing_type() == LibArc::TimingType::kDefault)) {
          calGateRiseFallSlew(rise_fall_slew, cap_load, TransType::kRise, arc.get());
        }

        if ((arc->get_timing_type() == LibArc::TimingType::kComb) || (arc->get_timing_type() == LibArc::TimingType::kCombFall)
            || (arc->get_timing_type() == LibArc::TimingType::kFallingEdge)
            || (arc->get_timing_type() == LibArc::TimingType::kDefault)) {
          calGateRiseFallSlew(rise_fall_slew, cap_load, TransType::kFall, arc.get());
        }
      }
    }
  }
}

void ToTimingEngine::calGateRiseFallDelay(TODelay rise_fall_delay[], float cap_load, TransType rf, LibArc* arc)
{
  int rise_fall = (int) rf - 1;
  float in_slew = get_target_slews()[rise_fall];
  TODelay gate_delay = arc->getDelayOrConstrainCheckNs(rf, in_slew, cap_load);
  rise_fall_delay[rise_fall] = max(rise_fall_delay[rise_fall], gate_delay);
}

void ToTimingEngine::calGateRiseFallSlew(TOSlew rise_fall_slew[], float cap_load, TransType rf, LibArc* arc)
{
  int rise_fall = (int) rf - 1;
  float in_slew = get_target_slews()[rise_fall];
  TOSlew driver_slew = arc->getSlewNs(rf, in_slew, cap_load);
  rise_fall_slew[rise_fall] = max(rise_fall_slew[rise_fall], driver_slew);
}

double ToTimingEngine::calcDelayOfBuffer(float load, LibCell* buffer_cell)
{
  LibPort *input, *output;
  buffer_cell->bufferPorts(input, output);
  TODelay gate_delays[2];
  calcGateRiseFallDelays(gate_delays, load, output);
  return max(gate_delays[TYPE_RISE], gate_delays[TYPE_FALL]);
}

TOSlack ToTimingEngine::getWorstSlack(StaVertex* vertex, AnalysisMode mode)
{
  auto rise_slack = vertex->getSlackNs(mode, TransType::kRise);
  TOSlack rise = rise_slack ? *rise_slack : kInf;
  auto fall_slack = vertex->getSlackNs(mode, TransType::kFall);
  TOSlack fall = fall_slack ? *fall_slack : kInf;
  TOSlack slack = min(rise, fall);
  return slack;
}

TOSlack ToTimingEngine::getWorstSlack(AnalysisMode mode)
{
  StaSeqPathData* worst_path_rise = _timing_engine->getWorstSeqData(mode, TransType::kRise);
  StaSeqPathData* worst_path_fall = _timing_engine->getWorstSeqData(mode, TransType::kFall);
  TOSlack worst_slack_rise = worst_path_rise->getSlackNs();
  TOSlack worst_slack_fall = worst_path_fall->getSlackNs();
  TOSlack slack = min(worst_slack_rise, worst_slack_fall);

  StaSeqPathData* worst_path = worst_slack_rise > worst_slack_fall ? worst_path_fall : worst_path_rise;
  string capture_name = worst_path->get_capture_clock_data()->get_own_vertex()->getName();
  string launch_name = worst_path->get_launch_clock_data()->get_own_vertex()->getName();
  if (mode == AnalysisMode::kMin) {
    toRptInst->report("\nWorst Hold Path Launch : " + launch_name);
    toRptInst->report("Worst Hold Path Capture: " + capture_name);
  } else {
    toRptInst->report("\nWorst Setup Path Launch : " + launch_name);
    toRptInst->report("Worst Setup Path Capture: " + capture_name);
  }

  return slack;
}

float ToTimingEngine::calcSetupDelayOfBuffer(float cap_load, LibCell* buffer_cell)
{
  auto delay_rise = calcSetupDelayOfBuffer(cap_load, TransType::kRise, buffer_cell);
  auto delay_fall = calcSetupDelayOfBuffer(cap_load, TransType::kFall, buffer_cell);
  return max(delay_rise, delay_fall);
}

float ToTimingEngine::calcSetupDelayOfBuffer(float cap_load, TransType rf, LibCell* buffer_cell)
{
  LibPort *input, *output;
  buffer_cell->bufferPorts(input, output);
  TODelay gate_delays[2];
  calcGateRiseFallDelays(gate_delays, cap_load, output);
  int rise_fall = (int)rf - 1;
  return gate_delays[rise_fall];
}

float ToTimingEngine::calcSetupDelayOfGate(float cap_load, TransType rf, LibPort* driver_port)
{
  TODelay rise_fall_delay[2];
  calcGateRiseFallDelays(rise_fall_delay, cap_load, driver_port);
  int rise_fall = (int) rf - 1;
  return rise_fall_delay[rise_fall];
}

float ToTimingEngine::calcSetupDelayOfGate(float cap_load, LibPort* driver_port)
{
  TODelay rise_fall_delay[2];
  calcGateRiseFallDelays(rise_fall_delay, cap_load, driver_port);
  return max(rise_fall_delay[TYPE_FALL], rise_fall_delay[TYPE_RISE]);
}

std::optional<TOSlack> ToTimingEngine::getNodeWorstSlack(StaVertex *node) {
  std::optional<TOSlack> worst_slack = std::nullopt;
  StaSeqPathData        *worst_path_rise =
      _timing_engine->getWorstSeqData(node, AnalysisMode::kMax, TransType::kRise);
  StaSeqPathData *worst_path_fall =
      _timing_engine->getWorstSeqData(node, AnalysisMode::kMax, TransType::kFall);
  if (!worst_path_fall || !worst_path_rise) {
    return worst_slack;
  }

  TOSlack worst_slack_rise = worst_path_rise->getSlackNs();
  TOSlack worst_slack_fall = worst_path_fall->getSlackNs();
  auto    worst =
      (worst_slack_rise > worst_slack_fall) ? worst_slack_fall : worst_slack_rise;
  worst_slack = std::make_optional(worst);
  return worst_slack;
}

StaSeqPathData* ToTimingEngine::getNodeWorstPath(StaVertex* node)
{
  StaSeqPathData* worst_path_rise = _timing_engine->getWorstSeqData(node, AnalysisMode::kMax, TransType::kRise);
  StaSeqPathData* worst_path_fall = _timing_engine->getWorstSeqData(node, AnalysisMode::kMax, TransType::kFall);
  TOSlack worst_slack_rise = worst_path_rise ? worst_path_rise->getSlackNs() : -kInf;
  TOSlack worst_slack_fall = worst_path_fall ? worst_path_fall->getSlackNs() : -kInf;
  return (worst_slack_rise > worst_slack_fall) ? worst_path_fall : worst_path_rise;
}

}  // namespace ito