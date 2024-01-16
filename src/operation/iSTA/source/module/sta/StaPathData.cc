// ***************************************************************************************
// Copyright (c) 2023-2025 Peng Cheng Laboratory
// Copyright (c) 2023-2025 Institute of Computing Technology, Chinese Academy of
// Sciences Copyright (c) 2023-2025 Beijing Institute of Open Source Chip
//
// iEDA is licensed under Mulan PSL v2.
// You can use this software according to the terms and conditions of the Mulan
// PSL v2. You may obtain a copy of Mulan PSL v2 at:
// http://license.coscl.org.cn/MulanPSL2
//
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
// NON-INFRINGEMENT, MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
//
// See the Mulan PSL v2 for more details.
// ***************************************************************************************
/**
 * @file StaPathData.cc
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The implemention of timing path data.
 * @version 0.1
 * @date 2021-03-15
 */
#include "StaPathData.hh"

#include <utility>

#include "StaArc.hh"
#include "StaReport.hh"

namespace ista {

StaClockPair::StaClockPair(int setup_launch_clock_edge,
                           int setup_capture_clock_edge,
                           int hold_launch_clock_edge,
                           int hold_capture_clock_edge)
    : _setup_launch_clock_edge(setup_launch_clock_edge),
      _setup_capture_clock_edge(setup_capture_clock_edge),
      _hold_launch_clock_edge(hold_launch_clock_edge),
      _hold_capture_clock_edge(hold_capture_clock_edge) {}

StaClockPair::StaClockPair(StaClockPair&& other) noexcept
    : _setup_launch_clock_edge(other._setup_launch_clock_edge),
      _setup_capture_clock_edge(other._setup_capture_clock_edge),
      _hold_launch_clock_edge(other._hold_launch_clock_edge),
      _hold_capture_clock_edge(other._hold_capture_clock_edge) {}

StaClockPair& StaClockPair::operator=(StaClockPair&& rhs) noexcept {
  _setup_launch_clock_edge = rhs._setup_launch_clock_edge;
  _setup_capture_clock_edge = rhs._setup_capture_clock_edge;
  _hold_launch_clock_edge = rhs._hold_launch_clock_edge;
  _hold_capture_clock_edge = rhs._hold_capture_clock_edge;

  return *this;
}

StaPathData::StaPathData(StaPathDelayData* delay_data,
                         StaClockData* launch_clock_data,
                         StaClockData* capture_clock_data)
    : _delay_data(delay_data),
      _launch_clock_data(launch_clock_data),
      _capture_clock_data(capture_clock_data) {}

StaSeqPathData::StaSeqPathData(StaPathDelayData* delay_data,
                               StaClockData* launch_clock_data,
                               StaClockData* capture_clock_data,
                               StaClockPair&& clock_pair,
                               std::optional<int> cppr, int constrain_value)
    : StaPathData(delay_data, launch_clock_data, capture_clock_data),
      _capture_clock(capture_clock_data->get_prop_clock()),
      _clock_pair(std::move(clock_pair)),
      _cppr(cppr),
      _constrain_value(constrain_value) {}

/**
 * @brief Get seq path data launch clock edge.
 *
 * @return int
 */
int64_t StaSeqPathData::getLaunchEdge() {
  auto analysis_mode = getDelayType();
  auto launch_edge = (analysis_mode == AnalysisMode::kMax)
                         ? _clock_pair.getSetupLaunchClockEdge()
                         : _clock_pair.getHoldLaunchClockEdge();
  int64_t launch_edge_fs = PS_TO_FS(launch_edge);

  return launch_edge_fs;
}

/**
 * @brief Get seq path data capture clock edge.
 *
 *
 * @return int
 */
int64_t StaSeqPathData::getCaptureEdge() {
  auto analysis_mode = getDelayType();
  auto capture_edge = (analysis_mode == AnalysisMode::kMax)
                          ? _clock_pair.getSetupCaptureClockEdge()
                          : _clock_pair.getHoldCaptureClockEdge();
  int64_t capture_edge_fs = PS_TO_FS(capture_edge);

  return capture_edge_fs;
}

/**
 * @brief Get the seq path arrive time.
 *
 * @return int The arrive time.
 */
int64_t StaSeqPathData::getArriveTime() {
  auto* delay_data = get_delay_data();
  auto path_arrive_time = delay_data->get_arrive_time();
  auto* launch_clock_data = get_launch_clock_data();
  auto clock_arrive_time = launch_clock_data->get_arrive_time();
  auto launch_edge = getLaunchEdge();

  int64_t arrive_time = path_arrive_time + clock_arrive_time + launch_edge;

  return arrive_time;
}

/**
 * @brief Get the seq path cell delay and net delay of arrive time.
 *
 * @return int64_t
 */
std::pair<int64_t, int64_t> StaSeqPathData::getCellAndNetDelayOfArriveTime() {
  int64_t cell_delay = 0;
  int64_t net_delay = 0;
  auto accumulate_cell_and_net_delay =
      [&cell_delay, &net_delay]<typename T>(std::stack<T>& data_stack) {
        auto* path_start_point = data_stack.top();
        int64_t last_arrive_time = path_start_point->get_arrive_time();
        StaVertex* last_vertex = path_start_point->get_own_vertex();
        data_stack.pop();

        while (!data_stack.empty()) {
          auto* path_data = data_stack.top();
          int64_t arrive_time = path_data->get_arrive_time();
          int64_t incr_time = arrive_time - last_arrive_time;

          auto* own_vertex = path_data->get_own_vertex();
          auto snk_arcs = last_vertex->getSnkArc(own_vertex);
          auto* snk_arc = snk_arcs.front();

          if (snk_arc->isInstArc()) {
            cell_delay += incr_time;
          } else {
            net_delay += incr_time;
          }

          last_arrive_time = arrive_time;
          last_vertex = own_vertex;

          data_stack.pop();
        }
      };

  std::stack<StaPathDelayData*> data_path_stack = getPathDelayData();
  auto* path_start_point = data_path_stack.top();
  auto* launch_clock_data = path_start_point->get_launch_clock_data();
  auto launch_clock_path_data_stack = launch_clock_data->getPathData();
  accumulate_cell_and_net_delay(launch_clock_path_data_stack);
  accumulate_cell_and_net_delay(data_path_stack);

  return std::make_pair(cell_delay, net_delay);  // unit is fs.
}

/**
 * @brief Get the seq path req time.
 *
 * @return int The require time.
 */
int64_t StaSeqPathData::getRequireTime() {
  auto* capture_clock_data = get_capture_clock_data();
  auto clock_arrive_time = capture_clock_data->get_arrive_time();
  auto constrain_value = get_constrain_value();
  auto uncertainty_value = get_uncertainty();
  auto uncertainty = uncertainty_value.value_or(0);
  auto capture_edge = getCaptureEdge();
  auto analysis_mode = getDelayType();
  auto cppr_value = _cppr.value_or(0);
  return analysis_mode == AnalysisMode::kMax
             ? (clock_arrive_time + capture_edge - constrain_value -
                uncertainty + cppr_value)
             : (clock_arrive_time + capture_edge + constrain_value +
                uncertainty - cppr_value);
}

/**
 * @brief Get the seq path slack.
 *
 * @return int The slack.
 */
int StaSeqPathData::getSlack() {
  int arrive_time = getArriveTime();
  int req_time = getRequireTime();
  auto analysis_mode = getDelayType();
  return analysis_mode == AnalysisMode::kMax ? (req_time - arrive_time)
                                             : (arrive_time - req_time);
}

/**
 * @brief Get the seq path skew.
 *
 * @return int
 */
int StaSeqPathData::getSkew() {
  auto* launch_clock_data = get_launch_clock_data();
  int launch_arrive_time = launch_clock_data->get_arrive_time();

  auto* capture_clock_data = get_capture_clock_data();
  int capture_arrive_time = capture_clock_data->get_arrive_time();

  AnalysisMode analysis_mode = getDelayType();

  int skew = capture_arrive_time - launch_arrive_time;

  auto cppr = get_cppr();
  if (cppr) {
    skew = (analysis_mode == AnalysisMode::kMax) ? skew + cppr.value()
                                                 : skew - cppr.value();
  }

  return skew;
}

/**
 * @brief Get the delay data of the path.
 *
 * @return std::stack<StaPathDelayData*>
 */
std::stack<StaPathDelayData*> StaSeqPathData::getPathDelayData() {
  std::stack<StaPathDelayData*> path_stack;
  auto* end_delay_data = get_delay_data();

  path_stack.push(end_delay_data);

  auto* bwd_data = dynamic_cast<StaPathDelayData*>(end_delay_data->get_bwd());
  while (bwd_data) {
    path_stack.push(bwd_data);
    bwd_data = dynamic_cast<StaPathDelayData*>(bwd_data->get_bwd());
  }

  return path_stack;
}

/**
 * @brief Generate the seq path report.
 *
 * @return unsigned
 */
unsigned StaSeqPathData::reportPath(const char* rpt_file_name) {
  StaReportPathSummary report_path(rpt_file_name, getDelayType());
  unsigned is_ok = report_path(this);
  return is_ok;
}

StaPortSeqPathData::StaPortSeqPathData(StaPathDelayData* delay_data,
                                       StaClockData* launch_clock_data,
                                       StaClockData* capture_clock_data,
                                       StaClockPair&& clock_pair,
                                       std::optional<int> cppr,
                                       int constrain_value, Port* output_port)
    : StaSeqPathData(delay_data, launch_clock_data, capture_clock_data,
                     std::move(clock_pair), cppr, constrain_value),
      _output_port(output_port) {}

StaClockGatePathData::StaClockGatePathData(StaPathDelayData* delay_data,
                                           StaClockData* launch_clock_data,
                                           StaClockData* capture_clock_data,
                                           StaClockPair&& clock_pair,
                                           std::optional<int> cppr,
                                           int constrain_value)
    : StaSeqPathData(delay_data, launch_clock_data, capture_clock_data,
                     std::move(clock_pair), cppr, constrain_value) {}

StaPathEnd::StaPathEnd(StaVertex* end_vertex) : _end_vertex(end_vertex) {}

StaPathEnd::StaPathEnd(StaPathEnd&& other) noexcept
    : _end_vertex(other._end_vertex),
      _max_timing_data(std::move(other._max_timing_data)),
      _min_timing_data(std::move(other._min_timing_data)) {}

StaPathEnd& StaPathEnd::operator=(StaPathEnd&& rhs) noexcept {
  if (this != &rhs) {
    _end_vertex = rhs._end_vertex;
    _max_timing_data = std::move(rhs._max_timing_data);
    _min_timing_data = std::move(rhs._min_timing_data);
  }

  return *this;
}

/**
 * @brief Insert the path data.
 *
 * @param seq_data
 * @return unsigned
 */
unsigned StaPathEnd::insertPathData(StaPathData* seq_data) {
  if (seq_data->getDelayType() == AnalysisMode::kMax) {
    _max_timing_data.emplace_back(seq_data);
  } else {
    _min_timing_data.emplace_back(seq_data);
  }

  _delay_data_to_path_data[seq_data->get_delay_data()] = seq_data;

  return 1;
}

/**
 * @brief Find the path data accord delay data.
 *
 * @param delay_data
 * @return StaPathData*
 */
StaPathData* StaPathEnd::findPathData(StaPathDelayData* delay_data) {
  if (auto it = _delay_data_to_path_data.find(delay_data);
      it != _delay_data_to_path_data.end()) {
    return it->second;
  }
  return nullptr;
}

StaPathEndIterator::StaPathEndIterator(StaPathEnd* path_end,
                                       AnalysisMode analysis_mode) {
  if (analysis_mode == AnalysisMode::kMax) {
    _iter = path_end->_max_timing_data.begin();
    _end = path_end->_max_timing_data.end();
  } else {
    _iter = path_end->_min_timing_data.begin();
    _end = path_end->_min_timing_data.end();
  }
}

bool StaPathEndIterator::hasNext() { return _iter != _end; }

StaPathData* StaPathEndIterator::next() { return _iter++->get(); }

StaPathGroup::StaPathGroup(StaPathGroup&& other) noexcept
    : _end_data(std::move(other._end_data)) {}

StaPathGroup& StaPathGroup::operator=(StaPathGroup&& rhs) noexcept {
  if (this != &rhs) {
    _end_data = std::move(rhs._end_data);
  }

  return *this;
}

/**
 * @brief Insert path data to path group.
 *
 * @param end_vertex
 * @param seq_data
 * @return unsigned
 */
unsigned StaPathGroup::insertPathData(StaVertex* end_vertex,
                                      StaPathData* seq_data) {
  auto p = _end_data.find(end_vertex);
  if (p == _end_data.end()) {
    auto path_end = std::make_unique<StaPathEnd>(end_vertex);
    path_end->insertPathData(seq_data);
    _end_data[end_vertex] = std::move(path_end);
  } else {
    p->second->insertPathData(seq_data);
  }
  return 1;
}

StaPathGroupIterator::StaPathGroupIterator(StaPathGroup* path_group)
    : _iter(path_group->_end_data.begin()), _end(path_group->_end_data.end()) {}

bool StaPathGroupIterator::hasNext() { return _iter != _end; }

StaPathEnd* StaPathGroupIterator::next() { return _iter++->second.get(); }

StaSeqPathGroup::StaSeqPathGroup(StaClock* capture_clock)
    : _capture_clock(capture_clock) {}

StaSeqPathGroup::StaSeqPathGroup(StaSeqPathGroup&& other) noexcept
    : StaPathGroup(std::move(other)), _capture_clock(other._capture_clock) {}

StaSeqPathGroup& StaSeqPathGroup::operator=(StaSeqPathGroup&& rhs) noexcept {
  if (this != &rhs) {
    StaPathGroup::operator=(std::move(rhs));
    _capture_clock = std::move(rhs._capture_clock);
  }
  return *this;
}

StaClockGatePathGroup::StaClockGatePathGroup(const char* clock_group)
    : StaSeqPathGroup(), _clock_group(clock_group) {}

StaClockGatePathGroup::StaClockGatePathGroup(
    StaClockGatePathGroup&& other) noexcept
    : StaSeqPathGroup(std::move(other)) {}

StaClockGatePathGroup& StaClockGatePathGroup::operator=(
    StaClockGatePathGroup&& rhs) noexcept {
  if (this != &rhs) {
    StaSeqPathGroup::operator=(std::move(rhs));
  }
  return *this;
}

}  // namespace ista
