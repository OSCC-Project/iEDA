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
 * @file StaAnalyze.cc
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The implemention of static timing analysis.
 * @version 0.1
 * @date 2021-03-14
 */
#include "StaAnalyze.hh"

#include <numeric>
#include <queue>
#include <utility>

#include "StaCppr.hh"
#include "Type.hh"

namespace ista {

/**
 * @brief Analyze the launch clock and capture clock relationship.
 *
 * @param launch_clock_data
 * @param capture_clock_data
 * @return unsigned
 */
StaClockPair StaAnalyze::analyzeClockRelation(
    StaClockData* launch_clock_data, StaClockData* capture_clock_data) {
  StaClock* launch_clock = launch_clock_data->get_prop_clock();
  StaClock* capture_clock = capture_clock_data->get_prop_clock();

  auto& launch_wave_form = launch_clock->get_wave_form();
  auto& capture_wave_form = capture_clock->get_wave_form();

  int launch_period = launch_clock->get_period();
  int capture_period = capture_clock->get_period();

  // for launch and capture period, if they are not interger times, we need
  // judge whether use the approximate interger value to avoid the edge
  // calculation.
  int the_max_period = std::max(launch_period, capture_period);
  int the_min_period = std::min(launch_period, capture_period);
  double ratio = static_cast<double>(the_max_period) / the_min_period;
  int ratio_n = std::round(ratio);
  constexpr double ratio_accuracy =
      0.002;  // if the ratio decimal less than this value, we choose the
              // interger.

  // Firstly, we get the period the lowest common multiple.
  // For the float period, may be need fixed by fraction, fix me.
  int period_lcm = 0;
  if (std::fabs(ratio - ratio_n) < ratio_accuracy) {
    period_lcm = ratio_n * the_min_period;
  } else {
    period_lcm = std::lcm(launch_period, capture_period);
  }

  // Secondly, we expand the waveform edge list.
  std::vector<int> launch_edges = {launch_wave_form.getRisingEdge(),
                                   launch_wave_form.getFallingEdge()};

  std::vector<int> capture_edges = {capture_wave_form.getRisingEdge(),
                                    capture_wave_form.getFallingEdge()};

  auto expand_edge = [](std::vector<int>& edges, int period_num, int period) {
    std::vector<int> tmp;

    for (int count = 0; count <= period_num; ++count) {
      for (auto edge : edges) {
        tmp.push_back(edge + (count * period));
      }
    }

    std::swap(tmp, edges);
  };

  int period_num = std::ceil(period_lcm / launch_period);
  // add more one period to avoid beyond edge.
  expand_edge(launch_edges, period_num + 1, launch_period);

  period_num = std::ceil(period_lcm / capture_period);
  expand_edge(capture_edges, period_num + 1, capture_period);

  // Thirdly, we find the nearest setup/hold clock pair.
  auto launch_trans_type = launch_clock_data->get_clock_wave_type();
  auto capture_trans_type = capture_clock_data->get_clock_wave_type();

  size_t init_launch_edge_index =
      (launch_trans_type == TransType::kRise) ? 0 : 1;
  size_t init_capture_edge_index =
      (capture_trans_type == TransType::kRise) ? (capture_edges[0] == 0) ? 2 : 0
      : (launch_trans_type == TransType::kFall) ? 3
                                                : 1;

  while ((((init_capture_edge_index + 2) < capture_edges.size()) &&
          launch_edges[init_launch_edge_index] >=
              capture_edges[init_capture_edge_index])) {
    init_capture_edge_index += 2;
  }

  int launch_edge = launch_edges[init_launch_edge_index];
  int capture_edge = capture_edges[init_capture_edge_index];

  LOG_FATAL_IF(launch_edge > capture_edge) << "not found init edge.";

  /**
   * @brief make hold pair according to setup pair, that exist two kind of hold
   * pair, step launch edge of setup pair, or backward capture edge of setup
   * pair.
   *
   */
  auto make_hold_pair = [&launch_edges, &capture_edges](
                            int launch_edge_index, int capture_edge_index,
                            int launch_edge, int capture_edge) {
    std::optional<std::pair<int, int>> hold_pair;
    // for hold pair, scenario 1
    if ((launch_edge_index + 2) < static_cast<int>(launch_edges.size())) {
      int next_launch_edge = launch_edges[launch_edge_index + 2];
      // int diff1 = next_launch_edge - capture_edge;
      hold_pair = std::make_pair(next_launch_edge, capture_edge);
    }

    // for hold pair, scenario 2
    if (capture_edge_index >= 2) {
      int previous_capture_edge = capture_edges[capture_edge_index - 2];
      int diff2 = launch_edge - previous_capture_edge;

      /*choose the most critical from the two scenario.*/
      if (!hold_pair || (diff2 <= (hold_pair->first - hold_pair->second))) {
        hold_pair = std::make_pair(launch_edge, previous_capture_edge);
      }
    }

    return hold_pair;
  };

  /**
   * @brief The clock pair choose criteria should be different of launch and
   * capture edge.
   *
   */
  auto cmp = [](StaClockPair& left, StaClockPair& right) -> bool {
    int left_diff = left.getSetupDiff();
    int right_diff = right.getSetupDiff();
    return left_diff > right_diff;
  };

  // choose analyze setup and hold pair.
  std::priority_queue<StaClockPair, std::vector<StaClockPair>, decltype(cmp)>
      clock_pairs(cmp);  // clock pair priority queue.
  auto hold_pair =
      make_hold_pair(init_launch_edge_index, init_capture_edge_index,
                     launch_edge, capture_edge);
  clock_pairs.emplace(launch_edge, capture_edge, hold_pair->first,
                      hold_pair->second);

  size_t launch_edge_index = init_launch_edge_index + 2;
  size_t capture_edge_index = init_capture_edge_index + 2;

  // we continue to search launch/capture edge pair from the second period.
  while ((launch_edge_index < launch_edges.size()) &&
         (capture_edge_index < capture_edges.size())) {
    // search next launch edge
    while (((launch_edge_index + 2) < launch_edges.size()) &&
           (launch_edges[launch_edge_index + 2] <
            capture_edges[capture_edge_index])) {
      launch_edge_index += 2;
    }

    // search next capture edge
    while ((((capture_edge_index + 2) < capture_edges.size()) &&
            launch_edges[launch_edge_index] >=
                capture_edges[capture_edge_index])) {
      capture_edge_index += 2;
    }

    if (launch_edges[launch_edge_index] > capture_edges[capture_edge_index]) {
      break;
    }

    launch_edge = launch_edges[launch_edge_index];
    capture_edge = capture_edges[capture_edge_index];

    auto hold_pair = make_hold_pair(launch_edge_index, capture_edge_index,
                                    launch_edge, capture_edge);
    LOG_FATAL_IF(!hold_pair) << "hold pair is not exist.";

    clock_pairs.emplace(launch_edge, capture_edge, hold_pair->first,
                        hold_pair->second);

    capture_edge_index += 2;
  }

  // Finally, we choose the most critical pair.
  const StaClockPair& result_clock_pair = clock_pairs.top();
  return result_clock_pair;
}

/**
 * @brief Analyze setup hold check.
 *
 * @param end_vertex
 * @param check_arc
 * @param analysis_mode
 * @return unsigned 1 if success, 0 else fail.
 */
unsigned StaAnalyze::analyzeSetupHold(StaVertex* end_vertex, StaArc* check_arc,
                                      AnalysisMode analysis_mode) {
  // LOG_INFO << "Analyze the endpoint " << end_vertex->getName();
  Sta* ista = getSta();
  auto* the_constrain = ista->getConstrain();

  TransType clock_trans_type =
      check_arc->isRisingEdgeCheck() ? TransType::kRise : TransType::kFall;
  StaData* capture_clock_data = nullptr;
  StaClock* capture_clock;

  unsigned is_ok = 1;

  auto* clock_vertex = check_arc->get_src();

  // The capture use the opposite mode.
  AnalysisMode capture_analysis_mode = (analysis_mode == AnalysisMode::kMax)
                                           ? AnalysisMode::kMin
                                           : AnalysisMode::kMax;
  
  unsigned end_path_index = 0;
  StaData* clock_data;
  FOREACH_CLOCK_DATA(clock_vertex, clock_data) {
    if ((clock_data->get_trans_type() == clock_trans_type) &&
        (capture_analysis_mode == clock_data->get_delay_type())) {
      capture_clock_data = clock_data;
      capture_clock =
          (dynamic_cast<StaClockData*>(capture_clock_data))->get_prop_clock();

      StaData* delay_data;
      FOREACH_DELAY_DATA(end_vertex, delay_data) {
        if (analysis_mode == delay_data->get_delay_type()) {
          StaClockData* launch_clock_data =
              (dynamic_cast<StaPathDelayData*>(delay_data))
                  ->get_launch_clock_data();

          auto* launch_clock = (dynamic_cast<StaClockData*>(launch_clock_data))
                                   ->get_prop_clock();

          std::optional<int> cppr;
          if (launch_clock == capture_clock) {
            StaCppr find_cppr(launch_clock_data,
                              dynamic_cast<StaClockData*>(capture_clock_data));
            if (capture_clock->exec(find_cppr)) {
              cppr = find_cppr.get_cppr();
            }
          }

          if ((launch_clock != capture_clock)) {
            std::string launch_clock_name = launch_clock->get_clock_name();
            std::string capture_clock_name = capture_clock->get_clock_name();

            if (the_constrain->isInAsyncGroup(launch_clock_name,
                                              capture_clock_name)) {
              continue;
            }
          }

          auto clock_pair = analyzeClockRelation(
              launch_clock_data,
              dynamic_cast<StaClockData*>(capture_clock_data));

          int constrain_value = check_arc->get_arc_delay(
              analysis_mode, delay_data->get_trans_type());

          StaSeqPathData* seq_data = new StaSeqPathData(
              dynamic_cast<StaPathDelayData*>(delay_data), launch_clock_data,
              dynamic_cast<StaClockData*>(capture_clock_data),
              std::move(clock_pair), cppr, constrain_value);

          seq_data->set_check_arc(check_arc);

          // add the data to path group.
          Sta* ista = getSta();
          ++end_path_index;
          ista->insertPathData(capture_clock, end_vertex, seq_data);
        }
      }
    }
  }

  LOG_INFO_EVERY_N(100) << "add path data num " << end_path_index  << " to " << end_vertex->getName();

  if ((!capture_clock_data)) {
    LOG_ERROR << "end vertex " << end_vertex->getName()
              << " has no clock data.";
  }

  return is_ok;
}

/**
 * @brief Analyze the output port delay setup/hold check.
 *
 * @param port_vertex
 * @param analysis_mode
 * @return unsigned
 */
unsigned StaAnalyze::analyzePortSetupHold(StaVertex* port_vertex,
                                          AnalysisMode analysis_mode) {
  Sta* ista = getSta();
  auto* the_constrain = ista->getConstrain();

  /*find the io constrain, create capture clock data and io constrain value*/
  auto io_constrains = ista->getIODelayConstrain(port_vertex);
  if (io_constrains.empty()) {
    DLOG_ERROR_FIRST_N(10) << "The output port " << port_vertex->getName()
                           << " is not constrained";
    return 1;
  }
  auto find_io_constrain = [&io_constrains, port_vertex](auto analysis_mode,
                                            auto trans_type) -> SdcSetIODelay* {
    auto it = std::find_if(
        io_constrains.begin(), io_constrains.end(),
        [=](SdcSetIODelay* io_constrain) {
          if (IS_MAX(analysis_mode) && IS_RISE(trans_type)) {
            return io_constrain->isMax() && io_constrain->isRise();
          } else if (IS_MAX(analysis_mode) && IS_FALL(trans_type)) {
            return io_constrain->isMax() && io_constrain->isFall();
          } else if (IS_MIN(analysis_mode) && IS_RISE(trans_type)) {
            return io_constrain->isMin() && io_constrain->isRise();
          } else {
            return io_constrain->isMin() && io_constrain->isFall();
          }
        });

    if (it == io_constrains.end()) {
      LOG_ERROR << ""
                << "The output port " << port_vertex->getName()
                << " has no io constrain for "
                << (IS_MAX(analysis_mode) ? "max" : "min") << " and "
                << (IS_RISE(trans_type) ? "rise" : "fall");
      return nullptr;
    }

    return *it;
  };

  /*iterator the path delay data, built the seq data.*/
  StaData* delay_data;
  FOREACH_DELAY_DATA(port_vertex, delay_data) {
    if (analysis_mode == delay_data->get_delay_type()) {
      StaClockData* launch_clock_data =
          (dynamic_cast<StaPathDelayData*>(delay_data))
              ->get_launch_clock_data();

      auto* launch_clock =
          (dynamic_cast<StaClockData*>(launch_clock_data))->get_prop_clock();

      auto trans_type = delay_data->get_trans_type();

      auto* output_delay = find_io_constrain(analysis_mode, trans_type);
      if (output_delay == nullptr) {
        // no output delay for this port and analysis mode.;
        continue;  
      }

      TransType clock_trans_type =
          output_delay->isClockFall() ? TransType::kFall : TransType::kRise;

      // The capture use the opposite mode.
      AnalysisMode capture_analysis_mode = (analysis_mode == AnalysisMode::kMax)
                                               ? AnalysisMode::kMin
                                               : AnalysisMode::kMax;

      std::string clock_name = output_delay->get_clock_name();
      auto* capture_clock = ista->findClock(clock_name.c_str());

      if ((launch_clock != capture_clock)) {
        std::string launch_clock_name = launch_clock->get_clock_name();
        std::string capture_clock_name = capture_clock->get_clock_name();

        if (the_constrain->isInAsyncGroup(launch_clock_name,
                                          capture_clock_name)) {
          continue;
        }
      }

      auto constrain_value = NS_TO_FS(output_delay->get_delay_value());
      if (analysis_mode == AnalysisMode::kMin) {
        // for output delay, we need minus output delay value in hold
        // analysis.
        constrain_value = -constrain_value;
      }

      auto& capture_clock_vertexes = capture_clock->get_clock_vertexes();

      if (!capture_clock_vertexes.empty()) {
        auto* capture_clock_vertex = *(capture_clock_vertexes.begin());

        StaClockData* capture_clock_data = nullptr;

        // add the data to path group.
        Sta* ista = getSta();
        StaData* capture_clock_vertex_clock_data;
        FOREACH_CLOCK_DATA(capture_clock_vertex,
                           capture_clock_vertex_clock_data) {
          capture_clock_data =
              dynamic_cast<StaClockData*>(capture_clock_vertex_clock_data);

          if ((capture_clock_data->get_trans_type() == clock_trans_type) &&
              (capture_analysis_mode == capture_clock_data->get_delay_type()) &&
              (capture_clock_data->get_prop_clock() == capture_clock)) {
            auto clock_pair =
                analyzeClockRelation(launch_clock_data, capture_clock_data);

            auto* port_seq_data = new StaPortSeqPathData(
                dynamic_cast<StaPathDelayData*>(delay_data), launch_clock_data,
                capture_clock_data, std::move(clock_pair), std::nullopt,
                constrain_value,
                dynamic_cast<Port*>(port_vertex->get_design_obj()));

            ista->insertPathData(capture_clock, port_vertex, port_seq_data);
          }
        }
      } else {
        // virtual clock

        auto* capture_clock_data =
            new StaClockData(capture_analysis_mode, clock_trans_type, 0,
                             port_vertex, capture_clock);
        capture_clock_data->set_clock_wave_type(clock_trans_type);
        port_vertex->addData(capture_clock_data);

        auto clock_pair =
            analyzeClockRelation(launch_clock_data, capture_clock_data);

        auto* port_seq_data = new StaPortSeqPathData(
            dynamic_cast<StaPathDelayData*>(delay_data), launch_clock_data,
            capture_clock_data, std::move(clock_pair), std::nullopt,
            constrain_value,
            dynamic_cast<Port*>(port_vertex->get_design_obj()));

        ista->insertPathData(capture_clock, port_vertex, port_seq_data);
      }
    }
  }
  return 1;
}

unsigned StaAnalyze::analyzeClockGateCheck(StaVertex* end_vertex,
                                           StaArc* check_arc,
                                           AnalysisMode analysis_mode) {
  // LOG_INFO << "Analyze the endpoint " << end_vertex->getName();
  Sta* ista = getSta();
  auto* the_constrain = ista->getConstrain();

  TransType clock_trans_type =
      check_arc->isRisingEdgeCheck() ? TransType::kRise : TransType::kFall;
  StaData* capture_clock_data = nullptr;
  StaClock* capture_clock;

  unsigned is_ok = 1;

  auto* clock_vertex = check_arc->get_src();

  // The capture use the opposite mode.
  AnalysisMode capture_analysis_mode = (analysis_mode == AnalysisMode::kMax)
                                           ? AnalysisMode::kMin
                                           : AnalysisMode::kMax;

  StaData* clock_data;
  FOREACH_CLOCK_DATA(clock_vertex, clock_data) {
    if ((clock_data->get_trans_type() == clock_trans_type) &&
        (capture_analysis_mode == clock_data->get_delay_type())) {
      capture_clock_data = clock_data;
      capture_clock =
          (dynamic_cast<StaClockData*>(capture_clock_data))->get_prop_clock();

      StaData* delay_data;
      FOREACH_DELAY_DATA(end_vertex, delay_data) {
        if (analysis_mode == delay_data->get_delay_type()) {
          StaClockData* launch_clock_data =
              (dynamic_cast<StaPathDelayData*>(delay_data))
                  ->get_launch_clock_data();

          auto* launch_clock = (dynamic_cast<StaClockData*>(launch_clock_data))
                                   ->get_prop_clock();

          std::optional<int> cppr;
          if (launch_clock == capture_clock) {
            StaCppr find_cppr(launch_clock_data,
                              dynamic_cast<StaClockData*>(capture_clock_data));
            if (capture_clock->exec(find_cppr)) {
              cppr = find_cppr.get_cppr();
            }
          }

          if ((launch_clock != capture_clock)) {
            std::string launch_clock_name = launch_clock->get_clock_name();
            std::string capture_clock_name = capture_clock->get_clock_name();

            if (the_constrain->isInAsyncGroup(launch_clock_name,
                                              capture_clock_name)) {
              continue;
            }
          }

          auto clock_pair = analyzeClockRelation(
              launch_clock_data,
              dynamic_cast<StaClockData*>(capture_clock_data));

          int constrain_value = check_arc->get_arc_delay(
              analysis_mode, delay_data->get_trans_type());

          StaClockGatePathData* clock_gate_data = new StaClockGatePathData(
              dynamic_cast<StaPathDelayData*>(delay_data), launch_clock_data,
              dynamic_cast<StaClockData*>(capture_clock_data),
              std::move(clock_pair), cppr, constrain_value);

          clock_gate_data->set_check_arc(check_arc);

          // add the data to path group.
          Sta* ista = getSta();
          ista->insertPathData(end_vertex, clock_gate_data);
        }
      }
    }
  }

  if ((!capture_clock_data)) {
    LOG_ERROR << "end vertex " << end_vertex->getName()
              << " has no clock data.";
  }

  return is_ok;
}

/**
 * @brief Analyze the end vertex to do timing constrain check.
 *
 * @param the_graph
 * @return unsigned 1 if success, 0 else fail.
 */
unsigned StaAnalyze::operator()(StaGraph* the_graph) {
  LOG_INFO << "analyze timing path start";

  AnalysisMode analysis_mode = get_analysis_mode();
  StaVertex* end_vertex;
  unsigned is_ok = 1;
  unsigned index = 0;
  FOREACH_END_VERTEX(the_graph, end_vertex) {
    ++index;
    LOG_INFO_EVERY_N(1000) << "analyze timing path end vertex " << index
                           << " total " << the_graph->get_end_vertexes().size() << " start";

    if (end_vertex->is_start() && end_vertex->is_end()) {
      // for clk vertex, maybe need check recovery time, skip this now.
      continue;
    }

    if (IS_MAX(analysis_mode)) {
      if (end_vertex->is_port()) {
        is_ok &= analyzePortSetupHold(end_vertex, AnalysisMode::kMax);
      } else if (end_vertex->is_end() && end_vertex->is_clock_gate_end()) {
        StaArc* check_arc = end_vertex->getCheckArc(AnalysisMode::kMax);
        is_ok &=
            analyzeClockGateCheck(end_vertex, check_arc, AnalysisMode::kMax);
      } else {
        StaArc* check_arc = end_vertex->getCheckArc(AnalysisMode::kMax);
        is_ok &= analyzeSetupHold(end_vertex, check_arc, AnalysisMode::kMax);
      }
    }

    if (IS_MIN(analysis_mode)) {
      if (end_vertex->is_port()) {
        is_ok &= analyzePortSetupHold(end_vertex, AnalysisMode::kMin);
      } else if (end_vertex->is_end() && end_vertex->is_clock_gate_end()) {
        StaArc* check_arc = end_vertex->getCheckArc(AnalysisMode::kMin);
        is_ok &=
            analyzeClockGateCheck(end_vertex, check_arc, AnalysisMode::kMin);
      } else {
        StaArc* check_arc = end_vertex->getCheckArc(AnalysisMode::kMin);
        is_ok &= analyzeSetupHold(end_vertex, check_arc, AnalysisMode::kMin);
      }
    }

    LOG_INFO_EVERY_N(10) << "analyze timing path end vertex " << index
    << " total " << the_graph->get_end_vertexes().size() << " end";
  }

  LOG_INFO << "analyze timing path end";

  return is_ok;
}

}  // namespace ista
