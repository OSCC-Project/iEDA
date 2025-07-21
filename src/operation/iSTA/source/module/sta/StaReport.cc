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
 * @file StaReport.cc
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The implemention of report timing path.
 * @version 0.1
 * @date 2021-04-23
 */
#include "StaReport.hh"

#include <algorithm>
#include <filesystem>
#include <optional>
#include <queue>
#include <stack>
#include <string>
#include <vector>

#include "Sta.hh"
#include "StaDump.hh"
#include "StaFunc.hh"
#include "StaVertex.hh"
#include "include/Version.hh"
#include "sta/StaPathData.hh"
#include "time/Time.hh"

namespace ista {

StaReportPathSummary::StaReportPathSummary(const char* rpt_file_name,
                                           AnalysisMode analysis_mode,
                                           unsigned n_worst)
    : _rpt_file_name(Str::copy(rpt_file_name)),
      _analysis_mode(analysis_mode),
      _n_worst(n_worst) {}

StaReportPathSummary::~StaReportPathSummary() { Str::free(_rpt_file_name); }

/**
 * @brief create the report table.
 *
 * @param tbl_name The report table name.
 * @return std::unique_ptr<StaReportTable> The created table.
 */
std::unique_ptr<StaReportTable> StaReportPathSummary::createReportTable(
    const char* tbl_name) {
  auto report_tbl = std::make_unique<StaReportTable>(tbl_name);

  (*report_tbl) << TABLE_HEAD;
  /* Fill each cell with operator[] */
  (*report_tbl)[0][0] = "Endpoint";
  (*report_tbl)[0][1] = "Clock Group";
  (*report_tbl)[0][2] = "Delay Type";
  (*report_tbl)[0][3] = "Path Delay";
  (*report_tbl)[0][4] = "Path Required";
  (*report_tbl)[0][5] = "CPPR";
  (*report_tbl)[0][6] = "Slack";
  (*report_tbl)[0][7] = "Freq(MHz)";
  (*report_tbl) << TABLE_ENDLINE;

  return report_tbl;
}

/**
 * @brief Report the sequential timing path.
 *
 * @param seq_path_data
 * @return unsigned 1 if success, 0 else failed.
 */
unsigned StaReportPathSummary::operator()(StaSeqPathData* seq_path_data) {
  unsigned is_ok = 1;
  Sta* ista = Sta::getOrCreateSta();
  auto& report_tbl_summary = ista->get_report_tbl_summary();

  auto* capture_clock = seq_path_data->get_capture_clock();

  auto delay_type = seq_path_data->getDelayType();
  const char* delay_type_str =
      (delay_type == AnalysisMode::kMax) ? "max" : "min";

  auto* delay_data = seq_path_data->get_delay_data();
  auto* endpoint = delay_data->get_own_vertex();

  auto arrive_time = seq_path_data->getArriveTime();
  auto trans_type = delay_data->get_trans_type();
  const char* type_str = trans_type == TransType::kRise ? "r" : "f";

  std::string fix_str = "%." + std::to_string(get_significant_digits()) + "f";
  auto fix_point_str = [fix_str](double data) {
    return Str::printf(fix_str.c_str(), data);
  };

  std::string arrive_time_str = fix_point_str(FS_TO_NS(arrive_time));
  arrive_time_str += type_str;

  auto req_time = seq_path_data->getRequireTime();
  std::string req_time_str = fix_point_str(FS_TO_NS(req_time));
  auto cppr = seq_path_data->get_cppr();
  std::string cppr_str = fix_point_str(FS_TO_NS(cppr.value_or(0)));
  int slack = seq_path_data->getSlack();
  std::string slack_str = fix_point_str(FS_TO_NS(slack));

  if (!seq_path_data->isStaClockGatePathData()) {
    std::string freq_str = "NA";
    // only consider capture clock is same with launch clock.
    if ((seq_path_data->getDelayType() == AnalysisMode::kMax) &&
        (seq_path_data->get_capture_clock() ==
         seq_path_data->get_launch_clock_data()->get_prop_clock())) {
      double clock_period = capture_clock->getPeriodNs();
      double slack = seq_path_data->getSlackNs();

      // freq is period inverse.
      double freq_MHz = 1000 / (clock_period - slack);
      freq_str = fix_point_str(freq_MHz);
    }

    (*report_tbl_summary) << endpoint->getName()
                          << capture_clock->get_clock_name() << delay_type_str
                          << arrive_time_str << req_time_str << cppr_str
                          << slack_str << freq_str << TABLE_ENDLINE;
  } else {
    const char* capture_clock_str = "**clock_gating_default**";
    (*report_tbl_summary) << endpoint->getName() << capture_clock_str
                          << delay_type_str << arrive_time_str << req_time_str
                          << cppr_str << slack_str << "NA" << TABLE_ENDLINE;
  }

  return is_ok;
}

/**
 * @brief Report the group of the sequential timing path.
 *
 * @param seq_path_group
 * @return unsigned
 */
unsigned StaReportPathSummary::operator()(StaSeqPathGroup* seq_path_group) {
  unsigned is_ok = 1;

  auto cmp = [](StaPathData* left, StaPathData* right) -> bool {
    int left_slack = left->getSlack();
    int right_slack = right->getSlack();
    return left_slack > right_slack;
  };

  std::priority_queue<StaPathData*, std::vector<StaPathData*>, decltype(cmp)>
      seq_data_queue(cmp);

  StaPathEnd* path_end;
  StaPathData* path_data;
  AnalysisMode analysis_mode = _analysis_mode;
  FOREACH_PATH_GROUP_END(seq_path_group, path_end)
  FOREACH_PATH_END_DATA(path_end, analysis_mode, path_data) {
    seq_data_queue.push(path_data);
  }

  unsigned i = 0;
  while (!seq_data_queue.empty() && i < get_n_worst()) {
    auto* seq_path_data = dynamic_cast<StaSeqPathData*>(seq_data_queue.top());
    is_ok = (*this)(seq_path_data);
    if (!is_ok) {
      break;
    }

    seq_data_queue.pop();
    ++i;
  }
  if (seq_path_group->isStaClockGatePathGroup()) {
    LOG_INFO << "report the clock group "
             << dynamic_cast<StaClockGatePathGroup*>(seq_path_group)
                    ->get_clock_group()
             << " " << i << " timing paths.";
  } else {
    LOG_INFO << "report the clock group "
             << seq_path_group->get_capture_clock()->get_clock_name() << " "
             << i << " timing paths.";
  }

  return is_ok;
}

StaReportClockTNS::StaReportClockTNS(const char* rpt_file_name,
                                     AnalysisMode analysis_mode,
                                     unsigned n_worst)
    : StaReportPathSummary(rpt_file_name, analysis_mode, n_worst) {}

std::unique_ptr<StaReportTable> StaReportClockTNS::createReportTable(
    const char* tbl_name) {
  auto report_tbl = std::make_unique<StaReportTable>(tbl_name);

  (*report_tbl) << TABLE_HEAD;

  /* Fill each cell with operator[] */
  (*report_tbl)[0][0] = "Clock";
  (*report_tbl)[0][1] = "Delay Type";
  (*report_tbl)[0][2] = "TNS";
  (*report_tbl) << TABLE_ENDLINE;

  // LOG_INFO << "\n" << report_tbl->c_str();

  return report_tbl;
}

unsigned StaReportClockTNS::operator()(StaSeqPathData* seq_path_data) {
  unsigned is_ok = 1;
  if (seq_path_data->isStaClockGatePathData()) {
    return is_ok;
  }

  Sta* ista = Sta::getOrCreateSta();
  auto& report_tbl_TNS = ista->get_report_tbl_TNS();

  auto* capture_clock = seq_path_data->get_capture_clock();
  auto* capture_clock_name = capture_clock->get_clock_name();
  auto delay_type = seq_path_data->getDelayType();
  const char* delay_type_str =
      (delay_type == AnalysisMode::kMax) ? "max" : "min";

  std::string fix_str = "%." + std::to_string(get_significant_digits()) + "f";
  auto fix_point_str = [fix_str](double data) {
    return Str::printf(fix_str.c_str(), data);
  };

  double TNS = ista->getTNS(capture_clock_name, delay_type);

  (*report_tbl_TNS) << capture_clock->get_clock_name() << delay_type_str
                    << fix_point_str(TNS) << TABLE_ENDLINE;
  return is_ok;
}

StaReportPathDetail::StaReportPathDetail(const char* rpt_file_name,
                                         AnalysisMode analysis_mode,
                                         unsigned n_worst, bool is_derate)
    : StaReportPathSummary(rpt_file_name, analysis_mode, n_worst),
      _is_derate(is_derate) {}

std::unique_ptr<StaReportTable> StaReportPathDetail::createReportTable(
    const char* tbl_name, bool is_derate) {
  auto report_tbl = std::make_unique<StaReportTable>(tbl_name);

  (*report_tbl) << TABLE_HEAD;
  if (is_derate) {
    /* Fill each cell with operator[] */
    (*report_tbl)[0][0] = "Point";
    (*report_tbl)[0][1] = "Fanout";
    (*report_tbl)[0][2] = "Capacitance";
    (*report_tbl)[0][3] = "Resistance";
    (*report_tbl)[0][4] = "Transition";
    (*report_tbl)[0][5] = "Delta Delay";
    (*report_tbl)[0][6] = "Derate";
    (*report_tbl)[0][7] = "Incr";
    (*report_tbl)[0][8] = "Path";
  } else {
    /* Fill each cell with operator[] */
    (*report_tbl)[0][0] = "Point";
    (*report_tbl)[0][1] = "Fanout";
    (*report_tbl)[0][2] = "Capacitance";
    (*report_tbl)[0][3] = "Resistance";
    (*report_tbl)[0][4] = "Transition";
    (*report_tbl)[0][5] = "Delta Delay";
    (*report_tbl)[0][6] = "Incr";
    (*report_tbl)[0][7] = "Path";
  }
  (*report_tbl) << TABLE_ENDLINE;

  return report_tbl;
}

/**
 * @brief Report seq path detail information.
 *
 * @param seq_path_data
 * @return unsigned
 */
unsigned StaReportPathDetail::operator()(StaSeqPathData* seq_path_data) {
  unsigned is_ok = 1;
  bool is_derate = get_is_derate();
  auto report_tbl = createReportTable("path", is_derate);
  std::string fix_str = "%." + std::to_string(get_significant_digits()) + "f";
  auto fix_point_str = [fix_str](double data) {
    return Str::printf(fix_str.c_str(), data);
  };

  auto* ista = Sta::getOrCreateSta();

#if CUDA_PROPAGATION
  auto& at_to_index = ista->get_at_to_index();
#else
  std::map<ista::StaPathDelayData*, unsigned int> at_to_index;
#endif

  auto print_path_data = [&report_tbl, &fix_point_str, &is_derate,
                          &at_to_index](auto& path_stack,
                                        auto clock_path_arrive_time) {
    double last_arrive_time = 0;
    StaVertex* last_vertex = nullptr;
    while (!path_stack.empty()) {
      auto* path_delay_data = path_stack.top();
      std::string path_delay_index_str;
#if CUDA_PROPAGATION
      unsigned gpu_at_index =
          at_to_index[dynamic_cast<StaPathDelayData*>(path_delay_data)];
      path_delay_index_str = Str::printf("(GPU AT %d)", gpu_at_index);
#endif

      auto* own_vertex = path_delay_data->get_own_vertex();
      auto trans_type = path_delay_data->get_trans_type();
      auto delay_type = path_delay_data->get_delay_type();

      // print net, check own_vertex is net load.
      if (auto* obj = own_vertex->get_design_obj();
          last_vertex &&
          ((obj->isPin() && obj->isInput() && !own_vertex->is_assistant()) ||
           (obj->isPort() && obj->isOutput() && own_vertex->is_assistant()))) {
        auto snk_arcs = last_vertex->getSnkArc(own_vertex);
        LOG_FATAL_IF(snk_arcs.size() != 1)
            << last_vertex->getName() << " " << own_vertex->getName()
            << " net arc found " << snk_arcs.size() << " arc.";
        if (snk_arcs.size() == 1) {
          auto* net_arc = dynamic_cast<StaNetArc*>(snk_arcs.front());
          auto* net = net_arc->get_net();
          auto crosstalk_delay =
              net_arc->getCrossTalkDelayNs(delay_type, trans_type);
          std::string crosstalk_delay_str =
              crosstalk_delay ? fix_point_str(crosstalk_delay.value()) : "NA";

          std::string net_attribute =
              net->isClockNet() ? "(clock net)" : "(net)";
          if (is_derate) {
            (*report_tbl) << Str::printf("%s %s", net->get_name(),
                                         net_attribute.c_str())
                          << net->getLoads().size() << TABLE_SKIP << TABLE_SKIP
                          << TABLE_SKIP << crosstalk_delay_str << TABLE_SKIP
                          << TABLE_SKIP << TABLE_SKIP << TABLE_ENDLINE;
          } else {
            (*report_tbl) << Str::printf("%s %s", net->get_name(),
                                         net_attribute.c_str())
                          << net->getLoads().size() << TABLE_SKIP << TABLE_SKIP
                          << TABLE_SKIP << crosstalk_delay_str << TABLE_SKIP
                          << TABLE_SKIP << TABLE_ENDLINE;
          }
        }
      }

      auto arrive_time = FS_TO_NS(path_delay_data->get_arrive_time());

      // if vertex is clock, use trigger type to report.
      if (own_vertex->is_clock()) {
        trans_type = own_vertex->isRisingTriggered() ? TransType::kRise
                                                     : TransType::kFall;
      }

      const char* trans_type_str = (trans_type == TransType::kRise) ? "r" : "f";
      auto incr_time = arrive_time - last_arrive_time;
      last_arrive_time = arrive_time;

      auto vertex_load =
          own_vertex->getLoad(path_delay_data->get_delay_type(), trans_type);
      auto vertex_resistance = own_vertex->getResistance(
          path_delay_data->get_delay_type(), trans_type);
      auto vertex_slew =
          own_vertex->getSlewNs(path_delay_data->get_delay_type(), trans_type);

      float vertex_derate =
          path_delay_data->get_derate() ? *(path_delay_data->get_derate()) : 1;
      if (is_derate) {
        (*report_tbl) << own_vertex->getNameWithCellName() << TABLE_SKIP
                      << fix_point_str(vertex_load)
                      << fix_point_str(vertex_resistance)
                      << fix_point_str(vertex_slew ? *vertex_slew : 0.0)
                      << TABLE_SKIP << fix_point_str(vertex_derate)
                      << fix_point_str(incr_time)
                      << std::string(fix_point_str(arrive_time +
                                                   clock_path_arrive_time)) +
                             trans_type_str + path_delay_index_str

                      << TABLE_ENDLINE;
      } else {
        (*report_tbl) << own_vertex->getNameWithCellName() << TABLE_SKIP
                      << fix_point_str(vertex_load)
                      << fix_point_str(vertex_resistance)
                      << fix_point_str(vertex_slew ? *vertex_slew : 0.0)
                      << TABLE_SKIP << fix_point_str(incr_time)
                      << std::string(fix_point_str(arrive_time +
                                                   clock_path_arrive_time)) +
                             trans_type_str + path_delay_index_str

                      << TABLE_ENDLINE;
      }

      last_vertex = own_vertex;
      path_stack.pop();
    }
  };

  auto print_path_data_info = [&report_tbl, seq_path_data, &print_path_data,
                               &fix_point_str, &is_derate]() {
    std::stack<StaPathDelayData*> path_stack =
        seq_path_data->getPathDelayData();
    /*The arrive time*/
    auto* path_delay_data = path_stack.top();
    auto* launch_clock_data = path_delay_data->get_launch_clock_data();
    auto launch_clock_path_data_stack = launch_clock_data->getPathData();
    print_path_data(launch_clock_path_data_stack, 0.0);

    auto* launch_clock = launch_clock_data->get_prop_clock();
    char* launch_clock_info =
        Str::printf("clock %s (%s)", launch_clock->get_clock_name(),
                    launch_clock_data->get_clock_wave_type() == TransType::kRise
                        ? "rise edge"
                        : "fall edge");

    double launch_edge = FS_TO_NS(seq_path_data->getLaunchEdge());

    if (is_derate) {
      (*report_tbl) << launch_clock_info << TABLE_SKIP << TABLE_SKIP
                    << TABLE_SKIP << TABLE_SKIP << TABLE_SKIP << TABLE_SKIP
                    << launch_edge << launch_edge << TABLE_ENDLINE;
    } else {
      (*report_tbl) << launch_clock_info << TABLE_SKIP << TABLE_SKIP
                    << TABLE_SKIP << TABLE_SKIP << TABLE_SKIP << launch_edge
                    << launch_edge << TABLE_ENDLINE;
    }

    auto launch_network_time = FS_TO_NS(launch_clock_data->get_arrive_time());
    double clock_path_arrive_time = launch_edge + launch_network_time;

    std::string clock_network_attribute =
        launch_clock->isIdealClockNetwork()
            ? "clock network delay (ideal)"
            : "clock network delay (propagated)";

    if (is_derate) {
      (*report_tbl) << clock_network_attribute << TABLE_SKIP << TABLE_SKIP
                    << TABLE_SKIP << TABLE_SKIP << TABLE_SKIP << TABLE_SKIP
                    << fix_point_str(launch_network_time)
                    << fix_point_str(clock_path_arrive_time) << TABLE_ENDLINE;
    } else {
      (*report_tbl) << clock_network_attribute << TABLE_SKIP << TABLE_SKIP
                    << TABLE_SKIP << TABLE_SKIP << TABLE_SKIP
                    << fix_point_str(launch_network_time)
                    << fix_point_str(clock_path_arrive_time) << TABLE_ENDLINE;
    }

    print_path_data(path_stack, clock_path_arrive_time);

    (*report_tbl) << TABLE_ENDLINE;

    /*The require time*/
    auto* capture_clock = seq_path_data->get_capture_clock();
    auto* capture_clock_data = seq_path_data->get_capture_clock_data();

    auto capture_clock_path_data_stack = capture_clock_data->getPathData();
    print_path_data(capture_clock_path_data_stack, 0.0);

    char* capture_clock_info = Str::printf(
        "clock %s (%s)", capture_clock->get_clock_name(),
        capture_clock_data->get_clock_wave_type() == TransType::kRise
            ? "rise edge"
            : "fall edge");

    double capture_edge = FS_TO_NS(seq_path_data->getCaptureEdge());
    if (is_derate) {
      (*report_tbl) << capture_clock_info << TABLE_SKIP << TABLE_SKIP
                    << TABLE_SKIP << TABLE_SKIP << TABLE_SKIP << TABLE_SKIP
                    << capture_edge << capture_edge << TABLE_ENDLINE;
    } else {
      (*report_tbl) << capture_clock_info << TABLE_SKIP << TABLE_SKIP
                    << TABLE_SKIP << TABLE_SKIP << TABLE_SKIP << capture_edge
                    << capture_edge << TABLE_ENDLINE;
    }

    auto capture_network_time = FS_TO_NS(capture_clock_data->get_arrive_time());
    clock_path_arrive_time = capture_edge + capture_network_time;

    clock_network_attribute = capture_clock->isIdealClockNetwork()
                                  ? "clock network delay (ideal)"
                                  : "clock network delay (propagated)";
    if (is_derate) {
      (*report_tbl) << clock_network_attribute << TABLE_SKIP << TABLE_SKIP
                    << TABLE_SKIP << TABLE_SKIP << TABLE_SKIP << TABLE_SKIP
                    << fix_point_str(capture_network_time)
                    << fix_point_str(clock_path_arrive_time) << TABLE_ENDLINE;
    } else {
      (*report_tbl) << clock_network_attribute << TABLE_SKIP << TABLE_SKIP
                    << TABLE_SKIP << TABLE_SKIP << TABLE_SKIP
                    << fix_point_str(capture_network_time)
                    << fix_point_str(clock_path_arrive_time) << TABLE_ENDLINE;
    }

    auto* end_vertex = capture_clock_data->get_own_vertex();
    auto trans_type = capture_clock_data->get_trans_type();
    const char* trans_type_str = trans_type == TransType::kRise ? "r" : "f";

    if (is_derate) {
      (*report_tbl) << end_vertex->getNameWithCellName() << TABLE_SKIP
                    << TABLE_SKIP << TABLE_SKIP << TABLE_SKIP << TABLE_SKIP
                    << TABLE_SKIP << TABLE_SKIP
                    << std::string(fix_point_str(clock_path_arrive_time)) +
                           trans_type_str
                    << TABLE_ENDLINE;
    } else {
      (*report_tbl) << end_vertex->getNameWithCellName() << TABLE_SKIP
                    << TABLE_SKIP << TABLE_SKIP << TABLE_SKIP << TABLE_SKIP
                    << TABLE_SKIP
                    << std::string(fix_point_str(clock_path_arrive_time)) +
                           trans_type_str
                    << TABLE_ENDLINE;
    }

    auto constrain_value = FS_TO_NS(seq_path_data->get_constrain_value());
    auto delay_type = seq_path_data->getDelayType();

    const char* constraint_arc_type = nullptr;
    auto* check_arc = seq_path_data->get_check_arc();
    if (check_arc) {
      if (AnalysisMode::kMax == delay_type) {
        if (check_arc->isSetupArc()) {
          constraint_arc_type = "library setup";
        } else {
          constraint_arc_type = "library recovery";
        }
      } else {
        if (check_arc->isHoldArc()) {
          constraint_arc_type = "library hold";
        } else {
          constraint_arc_type = "library removal";
        }
      }
    } else {
      if (AnalysisMode::kMax == delay_type) {
        constraint_arc_type = "output external max delay";
      } else {
        constraint_arc_type = "output external min delay";
      }
    }

    char* constrain_str = Str::printf("%s time", constraint_arc_type);

    constrain_value =
        (AnalysisMode::kMax == delay_type) ? -constrain_value : constrain_value;

    if (is_derate) {
      (*report_tbl)
          << constrain_str << TABLE_SKIP << TABLE_SKIP << TABLE_SKIP
          << TABLE_SKIP << TABLE_SKIP << TABLE_SKIP
          << fix_point_str(constrain_value)
          << (AnalysisMode::kMax == delay_type
                  ? fix_point_str(clock_path_arrive_time + constrain_value)
                  : fix_point_str(clock_path_arrive_time + constrain_value))
          << TABLE_ENDLINE;
    } else {
      (*report_tbl)
          << constrain_str << TABLE_SKIP << TABLE_SKIP << TABLE_SKIP
          << TABLE_SKIP << TABLE_SKIP << fix_point_str(constrain_value)
          << (AnalysisMode::kMax == delay_type
                  ? fix_point_str(clock_path_arrive_time + constrain_value)
                  : fix_point_str(clock_path_arrive_time + constrain_value))
          << TABLE_ENDLINE;
    }

    auto uncertainty = seq_path_data->get_uncertainty();
    if (uncertainty) {
      double uncertainty_value = FS_TO_NS(uncertainty.value());
      uncertainty_value = (AnalysisMode::kMax == delay_type)
                              ? -uncertainty_value
                              : uncertainty_value;
      if (is_derate) {
        (*report_tbl)
            << "clock uncertainty" << TABLE_SKIP << TABLE_SKIP << TABLE_SKIP
            << TABLE_SKIP << TABLE_SKIP << TABLE_SKIP
            << fix_point_str(uncertainty_value)
            << (AnalysisMode::kMax == delay_type
                    ? fix_point_str(clock_path_arrive_time + constrain_value +
                                    uncertainty_value)
                    : fix_point_str(clock_path_arrive_time + constrain_value +
                                    uncertainty_value))
            << TABLE_ENDLINE;
      } else {
        (*report_tbl)
            << "clock uncertainty" << TABLE_SKIP << TABLE_SKIP << TABLE_SKIP
            << TABLE_SKIP << TABLE_SKIP << fix_point_str(uncertainty_value)
            << (AnalysisMode::kMax == delay_type
                    ? fix_point_str(clock_path_arrive_time + constrain_value +
                                    uncertainty_value)
                    : fix_point_str(clock_path_arrive_time + constrain_value +
                                    uncertainty_value))
            << TABLE_ENDLINE;
      }
    }

    auto cppr = seq_path_data->get_cppr();
    if (cppr) {
      double cppr_value = FS_TO_NS(cppr.value());
      cppr_value =
          (AnalysisMode::kMax == delay_type) ? cppr_value : -cppr_value;
      if (is_derate) {
        (*report_tbl)
            << "clock reconvergence pessimism" << TABLE_SKIP << TABLE_SKIP
            << TABLE_SKIP << TABLE_SKIP << TABLE_SKIP << TABLE_SKIP
            << fix_point_str(cppr_value)
            << (AnalysisMode::kMax == delay_type
                    ? fix_point_str(clock_path_arrive_time + constrain_value -
                                    FS_TO_NS(uncertainty.value_or(0)) +
                                    cppr_value)
                    : fix_point_str(clock_path_arrive_time + constrain_value +
                                    FS_TO_NS(uncertainty.value_or(0)) +
                                    cppr_value))
            << TABLE_ENDLINE;
      } else {
        (*report_tbl)
            << "clock reconvergence pessimism" << TABLE_SKIP << TABLE_SKIP
            << TABLE_SKIP << TABLE_SKIP << TABLE_SKIP
            << fix_point_str(cppr_value)
            << (AnalysisMode::kMax == delay_type
                    ? fix_point_str(clock_path_arrive_time + constrain_value -
                                    FS_TO_NS(uncertainty.value_or(0)) +
                                    cppr_value)
                    : fix_point_str(clock_path_arrive_time + constrain_value +
                                    FS_TO_NS(uncertainty.value_or(0)) +
                                    cppr_value))
            << TABLE_ENDLINE;
      }
    }

    (*report_tbl) << TABLE_ENDLINE;

    auto [cell_delay, net_delay] =
        seq_path_data->getCellAndNetDelayOfArriveTime();
    int64_t path_arrive_time = seq_path_data->getArriveTime();
    auto calc_percent = [path_arrive_time, &fix_point_str](auto delay) {
      std::string delay_percent = Str::join(
          {fix_point_str(FS_TO_NS(delay)), "(",
           fix_point_str(static_cast<double>(delay) * 100 / path_arrive_time),
           "%)"},
          "");
      return delay_percent;
    };

    if (is_derate) {
      (*report_tbl) << "path cell delay" << TABLE_SKIP << TABLE_SKIP
                    << TABLE_SKIP << TABLE_SKIP << TABLE_SKIP << TABLE_SKIP
                    << TABLE_SKIP << calc_percent(cell_delay) << TABLE_ENDLINE;
      (*report_tbl) << "path net delay" << TABLE_SKIP << TABLE_SKIP
                    << TABLE_SKIP << TABLE_SKIP << TABLE_SKIP << TABLE_SKIP
                    << TABLE_SKIP << calc_percent(net_delay) << TABLE_ENDLINE;

    } else {
      (*report_tbl) << "path cell delay" << TABLE_SKIP << TABLE_SKIP
                    << TABLE_SKIP << TABLE_SKIP << TABLE_SKIP << TABLE_SKIP
                    << calc_percent(cell_delay) << TABLE_ENDLINE;

      (*report_tbl) << "path net delay" << TABLE_SKIP << TABLE_SKIP
                    << TABLE_SKIP << TABLE_SKIP << TABLE_SKIP << TABLE_SKIP
                    << calc_percent(net_delay) << TABLE_ENDLINE;
    }

    (*report_tbl) << TABLE_ENDLINE;

    /*The slack summary*/
    if (is_derate) {
      (*report_tbl) << "data require time" << TABLE_SKIP << TABLE_SKIP
                    << TABLE_SKIP << TABLE_SKIP << TABLE_SKIP << TABLE_SKIP
                    << TABLE_SKIP
                    << fix_point_str(FS_TO_NS(seq_path_data->getRequireTime()))
                    << TABLE_ENDLINE;
    } else {
      (*report_tbl) << "data require time" << TABLE_SKIP << TABLE_SKIP
                    << TABLE_SKIP << TABLE_SKIP << TABLE_SKIP << TABLE_SKIP
                    << fix_point_str(FS_TO_NS(seq_path_data->getRequireTime()))
                    << TABLE_ENDLINE;
    }

    if (is_derate) {
      (*report_tbl) << "data arrival time" << TABLE_SKIP << TABLE_SKIP
                    << TABLE_SKIP << TABLE_SKIP << TABLE_SKIP << TABLE_SKIP
                    << TABLE_SKIP << fix_point_str(FS_TO_NS(path_arrive_time))
                    << TABLE_ENDLINE;

    } else {
      (*report_tbl) << "data arrival time" << TABLE_SKIP << TABLE_SKIP
                    << TABLE_SKIP << TABLE_SKIP << TABLE_SKIP << TABLE_SKIP
                    << fix_point_str(FS_TO_NS(path_arrive_time))
                    << TABLE_ENDLINE;
    }

    auto slack = seq_path_data->getSlack();
    char* slack_str = Str::printf("slack (%s)", slack < 0 ? "VIOLATED" : "MET");
    if (is_derate) {
      (*report_tbl) << slack_str << TABLE_SKIP << TABLE_SKIP << TABLE_SKIP
                    << TABLE_SKIP << TABLE_SKIP << TABLE_SKIP << TABLE_SKIP
                    << fix_point_str(FS_TO_NS(slack)) << TABLE_ENDLINE;
    } else {
      (*report_tbl) << slack_str << TABLE_SKIP << TABLE_SKIP << TABLE_SKIP
                    << TABLE_SKIP << TABLE_SKIP << TABLE_SKIP
                    << fix_point_str(FS_TO_NS(slack)) << TABLE_ENDLINE;
    }
  };

  print_path_data_info();

  // LOG_INFO << "\n" << report_tbl->c_str();

  auto& report_tbl_details = ista->get_report_tbl_details();
  report_tbl_details.emplace_back(std::move(report_tbl));

  return is_ok;
}

StaReportPathDump::StaReportPathDump(const char* rpt_file_name,
                                     AnalysisMode analysis_mode,
                                     unsigned n_worst)
    : StaReportPathSummary(rpt_file_name, analysis_mode, n_worst) {}

/**
 * @brief Dump the seq path inner data.
 *
 * @param seq_path_data
 * @return unsigned
 */
unsigned StaReportPathDump::operator()(StaSeqPathData* seq_path_data) {
  StaDumpYaml dump_yaml;
  std::stack<StaPathDelayData*> path_stack = seq_path_data->getPathDelayData();

  StaVertex* last_vertex = nullptr;
  while (!path_stack.empty()) {
    auto* path_delay_data = path_stack.top();
    auto* own_vertex = path_delay_data->get_own_vertex();
    own_vertex->exec(dump_yaml);

    if (last_vertex) {
      auto snk_arcs = last_vertex->getSnkArc(own_vertex);
      auto* snk_arc = snk_arcs.empty() ? nullptr : snk_arcs.front();
      snk_arc->exec(dump_yaml);
    }

    last_vertex = own_vertex;

    path_stack.pop();
  }

  std::string design_work_space = dump_yaml.getSta()->get_design_work_space();
  std::string path_dir = design_work_space + "/path";
  std::filesystem::create_directories(path_dir);

  static unsigned file_id = 1;
  std::string now_time = Time::getNowWallTime();
  std::string tmp = Str::replace(now_time, ":", "_");
  const char* text_file_name = Str::printf(
      "%s/path_%s_%d.yaml", path_dir.c_str(), tmp.c_str(), file_id++);

  dump_yaml.printText(text_file_name);

  return 1;
}

StaReportPathYaml::StaReportPathYaml(const char* rpt_file_name,
                                     AnalysisMode analysis_mode,
                                     unsigned n_worst)
    : StaReportPathDump(rpt_file_name, analysis_mode, n_worst) {}

/**
 * @brief print report path in yaml not report table.
 *
 * @param seq_path_data
 * @return unsigned
 */
unsigned StaReportPathYaml::operator()(StaSeqPathData* seq_path_data) {
  StaDumpDelayYaml dump_delay_yaml;
  std::stack<StaPathDelayData*> path_stack = seq_path_data->getPathDelayData();

  StaVertex* last_vertex = nullptr;
  while (!path_stack.empty()) {
    auto* path_delay_data = path_stack.top();
    auto* own_vertex = path_delay_data->get_own_vertex();
    dump_delay_yaml.set_analysis_mode(path_delay_data->get_delay_type());
    dump_delay_yaml.set_trans_type(path_delay_data->get_trans_type());

    if (last_vertex) {
      auto snk_arcs = last_vertex->getSnkArc(own_vertex);
      auto* snk_arc = snk_arcs.empty() ? nullptr : snk_arcs.front();
      snk_arc->exec(dump_delay_yaml);
    }

    own_vertex->exec(dump_delay_yaml);

    last_vertex = own_vertex;

    path_stack.pop();
  }

  std::string design_work_space =
      dump_delay_yaml.getSta()->get_design_work_space();
  std::string path_dir = design_work_space + "/path";
  std::filesystem::create_directories(path_dir);

  static unsigned file_id = 1;
  std::string now_time = Time::getNowWallTime();
  std::string tmp = Str::replace(now_time, ":", "_");
  const char* text_file_name = Str::printf(
      "%s/path_delay_%s_%d.yaml", path_dir.c_str(), tmp.c_str(), file_id++);

  dump_delay_yaml.printText(text_file_name);

  return 1;
}

StaReportWirePathYaml::StaReportWirePathYaml(const char* rpt_file_name,
                                             AnalysisMode analysis_mode,
                                             unsigned n_worst)
    : StaReportPathDump(rpt_file_name, analysis_mode, n_worst) {}

/**
 * @brief print timing path in yaml in wire level.
 *
 * @param seq_path_data
 * @return unsigned
 */
unsigned StaReportWirePathYaml::operator()(StaSeqPathData* seq_path_data) {
  // CPU_PROF_START(0);
  std::string design_work_space =
      ista::Sta::getOrCreateSta()->get_design_work_space();
  std::string path_dir = design_work_space + "/wire_paths";

  std::filesystem::create_directories(path_dir);

  static unsigned file_id = 1;
  std::string now_time = Time::getNowWallTime();
  std::string tmp = Str::replace(now_time, ":", "_");
  const char* text_file_name = Str::printf(
      "%s/wire_path_%s_%d.yml", path_dir.c_str(), tmp.c_str(), file_id++);

  std::ofstream file(text_file_name, std::ios::trunc);
  StaDumpWireYaml dump_wire_yaml(file);
  std::stack<StaPathDelayData*> path_stack = seq_path_data->getPathDelayData();

  StaVertex* last_vertex = nullptr;
  while (!path_stack.empty()) {
    auto* path_delay_data = path_stack.top();
    auto* own_vertex = path_delay_data->get_own_vertex();
    dump_wire_yaml.set_analysis_mode(path_delay_data->get_delay_type());
    dump_wire_yaml.set_trans_type(path_delay_data->get_trans_type());

    if (last_vertex) {
      auto snk_arcs = last_vertex->getSnkArc(own_vertex);
      auto* snk_arc = snk_arcs.empty() ? nullptr : snk_arcs.front();
      snk_arc->exec(dump_wire_yaml);
    }

    own_vertex->exec(dump_wire_yaml);

    last_vertex = own_vertex;

    path_stack.pop();
  }

  file.close();

  LOG_INFO << "output yaml file path: " << text_file_name;

  // CPU_PROF_END(0, "dump one timing path wire yaml");

  return 1;
}

StaReportPathTimingData::StaReportPathTimingData(const char* rpt_file_name,
                                                 AnalysisMode analysis_mode,
                                                 unsigned n_worst)
    : StaReportPathSummary(rpt_file_name, analysis_mode, n_worst) {}

/**
 * @brief report path timing data in memory for python call.
 *
 * @param seq_path_data
 * @return unsigned
 */
unsigned StaReportPathTimingData::operator()(StaSeqPathData* seq_path_data) {
  StaDumpTimingData dump_timing_data;

  std::stack<StaPathDelayData*> path_stack = seq_path_data->getPathDelayData();

  StaVertex* last_vertex = nullptr;
  while (!path_stack.empty()) {
    auto* path_delay_data = path_stack.top();
    auto* own_vertex = path_delay_data->get_own_vertex();
    dump_timing_data.set_analysis_mode(path_delay_data->get_delay_type());
    dump_timing_data.set_trans_type(path_delay_data->get_trans_type());

    if (last_vertex) {
      auto snk_arcs = last_vertex->getSnkArc(own_vertex);
      auto* snk_arc = snk_arcs.empty() ? nullptr : snk_arcs.front();
      snk_arc->exec(dump_timing_data);
    }

    last_vertex = own_vertex;

    path_stack.pop();
  }

  auto wire_timing_datas = dump_timing_data.get_wire_timing_datas();
  set_path_timing_data(std::move(wire_timing_datas));

  return 1;
}

StaReportTrans::StaReportTrans(const char* rpt_file_name,
                               AnalysisMode analysis_mode, unsigned n_worst)
    : _rpt_file_name(Str::copy(rpt_file_name)),
      _analysis_mode(analysis_mode),
      _n_worst(n_worst) {}

StaReportTrans::~StaReportTrans() { Str::free(_rpt_file_name); }

std::unique_ptr<StaReportTable> StaReportTrans::createReportTable(
    const char* tbl_name) {
  auto report_tbl = std::make_unique<StaReportTable>(tbl_name);

  (*report_tbl) << TABLE_HEAD;
  /* Fill each cell with operator[] */
  (*report_tbl)[0][0] = "Net / Pin";
  (*report_tbl)[0][1] = "MaxSlewTime";
  (*report_tbl)[0][2] = "SlewTime";
  (*report_tbl)[0][3] = "SlewSlack";
  (*report_tbl)[0][4] = "LibCellPort";
  (*report_tbl)[0][5] = "Note";

  (*report_tbl) << TABLE_ENDLINE;

  return report_tbl;
}

/**
 * @brief report the trans violation.
 *
 * @param the_graph
 * @return unsigned
 */
unsigned StaReportTrans::operator()(Sta* ista) {
  unsigned is_ok = 1;

  auto report_tbl = createReportTable("trans");

  auto cmp = [ista, this](StaVertex* left, StaVertex* right) -> bool {
    auto left_slack =
        ista->getVertexSlewSlack(left, _analysis_mode, TransType::kRise);
    auto right_slack =
        ista->getVertexSlewSlack(right, _analysis_mode, TransType::kRise);
    if (left_slack && right_slack) {
      return *left_slack > *right_slack;
    } else if (left_slack) {
      return false;
    } else if (right_slack) {
      return true;
    }
    return true;
  };

  std::priority_queue<StaVertex*, std::vector<StaVertex*>, decltype(cmp)>
      trans_data_queue(cmp);

  auto& the_graph = ista->get_graph();

  StaVertex* vertex;
  FOREACH_VERTEX(&the_graph, vertex) {
    if (vertex->is_const()) {
      continue;
    }
    trans_data_queue.push(vertex);
  }

  auto print_trans_data = [&report_tbl, ista, this](StaVertex* the_vertex) {
    auto* obj = the_vertex->get_design_obj();
    auto* net = obj->get_net();
    const auto* net_name = net->get_name();

    (*report_tbl) << net_name << TABLE_SKIP << TABLE_SKIP << TABLE_SKIP
                  << TABLE_SKIP << TABLE_SKIP << TABLE_ENDLINE;

    auto vertex_name = the_vertex->getName();
    auto rise_limit =
        ista->getVertexSlewLimit(the_vertex, _analysis_mode, TransType::kRise);
    auto fall_limit =
        ista->getVertexSlewLimit(the_vertex, _analysis_mode, TransType::kFall);

    auto str_or_na = [](std::optional<double>& data, bool is_rise) {
      std::string str;
      if (data) {
        str = Str::printf("%.3f%s", *data, is_rise ? "r" : "f");
      } else {
        str = "NA";
      }
      return str;
    };
    std::string limit_str;
    limit_str += str_or_na(rise_limit, true);
    limit_str += "/";
    limit_str += str_or_na(fall_limit, false);

    auto rise_slew = the_vertex->getSlewNs(_analysis_mode, TransType::kRise);
    auto fall_slew = the_vertex->getSlewNs(_analysis_mode, TransType::kFall);

    std::string slew_str;
    slew_str += Str::printf("%.3fr", *rise_slew);
    slew_str += "/";
    slew_str += Str::printf("%.3ff", *fall_slew);

    std::optional<double> rise_slack;
    std::optional<double> fall_slack;

    if (rise_limit && rise_slew) {
      rise_slack = *rise_limit - *rise_slew;
    }

    if (fall_limit && fall_slew) {
      fall_slack = *fall_limit - *fall_slew;
    }

    std::string slack_str;
    slack_str += str_or_na(rise_slack, true);
    slack_str += "/";
    slack_str += str_or_na(fall_slack, false);

    std::string cell_port_name;
    if (obj->isPin()) {
      auto* pin = dynamic_cast<Pin*>(obj);
      std::string cell_name =
          pin->get_cell_port()->get_ower_cell()->get_cell_name();
      std::string port_name = pin->get_cell_port()->get_port_name();

      cell_port_name = cell_name;
      cell_port_name += "/";
      cell_port_name += port_name;
    }

    std::string note = (rise_slack && (*rise_slack < 0.0)) ? "R" : "";

    (*report_tbl) << vertex_name << limit_str << slew_str << slack_str
                  << cell_port_name << note << TABLE_ENDLINE;
  };

  unsigned i = 0;
  while (!trans_data_queue.empty() && i < _n_worst) {
    auto* vertex = trans_data_queue.top();
    auto* obj = vertex->get_design_obj();
    if (!obj->get_net()) {
      trans_data_queue.pop();
      continue;
    }

    // print vertex trans data.
    print_trans_data(vertex);

    trans_data_queue.pop();
    i++;
  }

  // LOG_INFO << "\n" << report_tbl->c_str();

  auto close_file = [](std::FILE* fp) { std::fclose(fp); };

  std::unique_ptr<std::FILE, decltype(close_file)> f(
      std::fopen(_rpt_file_name, "w"), close_file);

  std::fprintf(f.get(), "Generate the report at %s, GitVersion: %s.\n",
               Time::getNowWallTime(), GIT_VERSION);
  std::fprintf(f.get(), "%s", report_tbl->c_str());

  return is_ok;
}

StaReportCap::StaReportCap(const char* rpt_file_name,
                           AnalysisMode analysis_mode, unsigned n_worst,
                           bool is_clock_cap)
    : _rpt_file_name(Str::copy(rpt_file_name)),
      _analysis_mode(analysis_mode),
      _n_worst(n_worst),
      _is_clock_cap(is_clock_cap) {}

StaReportCap::~StaReportCap() { Str::free(_rpt_file_name); }

std::unique_ptr<StaReportTable> StaReportCap::createReportTable(
    const char* tbl_name) {
  auto report_tbl = std::make_unique<StaReportTable>(tbl_name);

  (*report_tbl) << TABLE_HEAD;
  /* Fill each cell with operator[] */
  (*report_tbl)[0][0] = "Net / Pin";
  (*report_tbl)[0][1] = "MaxCapacitance";
  (*report_tbl)[0][2] = "Capacitance";
  (*report_tbl)[0][3] = "CapacitanceSlack";
  (*report_tbl)[0][4] = "LibCellPort";
  (*report_tbl)[0][5] = "Note";

  (*report_tbl) << TABLE_ENDLINE;

  return report_tbl;
}

/**
 * @brief report the cap violation.
 *
 * @param the_graph
 * @return unsigned
 */
unsigned StaReportCap::operator()(Sta* ista) {
  unsigned is_ok = 1;

  auto report_tbl = createReportTable("cap");

  auto cmp = [ista, this](StaVertex* left, StaVertex* right) -> bool {
    auto left_slack =
        ista->getVertexCapacitanceSlack(left, _analysis_mode, TransType::kRise);
    auto right_slack = ista->getVertexCapacitanceSlack(right, _analysis_mode,
                                                       TransType::kRise);
    if (left_slack && right_slack) {
      return *left_slack > *right_slack;
    } else if (left_slack) {
      return false;
    } else if (right_slack) {
      return true;
    }
    return true;
  };

  std::priority_queue<StaVertex*, std::vector<StaVertex*>, decltype(cmp)>
      cap_data_queue(cmp);

  auto& the_graph = ista->get_graph();

  StaVertex* vertex;
  bool is_clock_cap = get_is_clock_cap();
  if (is_clock_cap) {
    FOREACH_VERTEX(&the_graph, vertex) {
      if (vertex->isHavePropClock()) {
        cap_data_queue.push(vertex);
      }
    }
  } else {
    FOREACH_VERTEX(&the_graph, vertex) {
      if (vertex->is_const()) {
        continue;
      }
      cap_data_queue.push(vertex);
    }
  }

  auto print_cap_data = [&report_tbl, ista, this](StaVertex* the_vertex) {
    auto* obj = the_vertex->get_design_obj();
    auto* net = obj->get_net();
    const auto* net_name = net->get_name();

    (*report_tbl) << net_name << TABLE_SKIP << TABLE_SKIP << TABLE_SKIP
                  << TABLE_SKIP << TABLE_SKIP << TABLE_ENDLINE;

    auto vertex_name = the_vertex->getName();
    auto rise_limit = ista->getVertexCapacitanceLimit(
        the_vertex, _analysis_mode, TransType::kRise);
    auto fall_limit = ista->getVertexCapacitanceLimit(
        the_vertex, _analysis_mode, TransType::kFall);

    auto str_or_na = [](std::optional<double>& data, bool is_rise) {
      std::string str;
      if (data) {
        str = Str::printf("%.3f%s", *data, is_rise ? "r" : "f");
      } else {
        str = "NA";
      }
      return str;
    };
    std::string limit_str;
    limit_str += str_or_na(rise_limit, true);
    limit_str += "/";
    limit_str += str_or_na(fall_limit, false);

    double rise_cap = ista->getVertexCapacitance(the_vertex, _analysis_mode,
                                                 TransType::kRise);
    double fall_cap = ista->getVertexCapacitance(the_vertex, _analysis_mode,
                                                 TransType::kFall);

    std::string cap_str;
    cap_str += Str::printf("%.3fr", rise_cap);
    cap_str += "/";
    cap_str += Str::printf("%.3ff", fall_cap);

    std::optional<double> rise_slack;
    std::optional<double> fall_slack;

    if (rise_limit) {
      rise_slack = *rise_limit - rise_cap;
    }

    if (rise_limit) {
      fall_slack = *fall_limit - fall_cap;
    }

    std::string slack_str;
    slack_str += str_or_na(rise_slack, true);
    slack_str += "/";
    slack_str += str_or_na(fall_slack, false);

    std::string cell_port_name;
    if (obj->isPin()) {
      auto* pin = dynamic_cast<Pin*>(obj);
      std::string cell_name =
          pin->get_cell_port()->get_ower_cell()->get_cell_name();
      std::string port_name = pin->get_cell_port()->get_port_name();

      cell_port_name = cell_name;
      cell_port_name += "/";
      cell_port_name += port_name;
    }

    std::string note = (rise_slack && (*rise_slack < 0.0)) ? "R" : "";

    (*report_tbl) << vertex_name << limit_str << cap_str << slack_str
                  << cell_port_name << note << TABLE_ENDLINE;
  };

  unsigned i = 0;
  while (!cap_data_queue.empty() && i < _n_worst) {
    auto* vertex = cap_data_queue.top();
    auto* obj = vertex->get_design_obj();
    if (!obj->get_net()) {
      cap_data_queue.pop();
      continue;
    }

    // print vertex trans data.
    print_cap_data(vertex);

    cap_data_queue.pop();
    i++;
  }

  // LOG_INFO << "\n" << report_tbl->c_str();

  auto close_file = [](std::FILE* fp) { std::fclose(fp); };

  std::unique_ptr<std::FILE, decltype(close_file)> f(
      std::fopen(_rpt_file_name, "w"), close_file);

  std::fprintf(f.get(), "Generate the report at %s, GitVersion: %s.\n",
               Time::getNowWallTime(), GIT_VERSION);
  std::fprintf(f.get(), "%s", report_tbl->c_str());

  return is_ok;
}

StaReportFanout::StaReportFanout(const char* rpt_file_name,
                                 AnalysisMode analysis_mode, unsigned n_worst)
    : _rpt_file_name(Str::copy(rpt_file_name)),
      _analysis_mode(analysis_mode),
      _n_worst(n_worst) {}

StaReportFanout::~StaReportFanout() { Str::free(_rpt_file_name); }

std::unique_ptr<StaReportTable> StaReportFanout::createReportTable(
    const char* tbl_name) {
  auto report_tbl = std::make_unique<StaReportTable>(tbl_name);

  (*report_tbl) << TABLE_HEAD;
  /* Fill each cell with operator[] */
  (*report_tbl)[0][0] = "Net / Pin";
  (*report_tbl)[0][1] = "MaxFanout";
  (*report_tbl)[0][2] = "Fanout";
  (*report_tbl)[0][3] = "FanoutSlack";
  (*report_tbl)[0][4] = "LibCellPort";
  (*report_tbl)[0][5] = "Note";

  (*report_tbl) << TABLE_ENDLINE;

  return report_tbl;
}

/**
 * @brief report the trans violation.
 *
 * @param the_graph
 * @return unsigned
 */
unsigned StaReportFanout::operator()(Sta* ista) {
  unsigned is_ok = 1;

  auto report_tbl = createReportTable("fanout");

  auto cmp = [ista, this](StaVertex* left, StaVertex* right) -> bool {
    auto left_slack = ista->getDriverVertexFanoutSlack(left, _analysis_mode);
    auto right_slack = ista->getDriverVertexFanoutSlack(right, _analysis_mode);
    if (left_slack && right_slack) {
      return *left_slack > *right_slack;
    } else if (left_slack) {
      return false;
    } else if (right_slack) {
      return true;
    }
    return true;
  };

  std::priority_queue<StaVertex*, std::vector<StaVertex*>, decltype(cmp)>
      fanout_data_queue(cmp);

  auto& the_graph = ista->get_graph();

  StaVertex* vertex;
  FOREACH_VERTEX(&the_graph, vertex) {
    if (vertex->is_const()) {
      continue;
    }
    fanout_data_queue.push(vertex);
  }

  auto print_fanout_data = [&report_tbl, ista, this](StaVertex* the_vertex) {
    auto* obj = the_vertex->get_design_obj();
    auto* net = obj->get_net();
    const auto* net_name = net->get_name();

    (*report_tbl) << net_name << TABLE_SKIP << TABLE_SKIP << TABLE_SKIP
                  << TABLE_SKIP << TABLE_SKIP << TABLE_ENDLINE;

    auto vertex_name = the_vertex->getName();
    auto limit = ista->getDriverVertexFanoutLimit(the_vertex, _analysis_mode);

    auto str_or_na = [](std::optional<double>& data) {
      std::string str;
      if (data) {
        str = Str::printf("%d", (int)(*data));
      } else {
        str = "NA";
      }
      return str;
    };
    std::string limit_str;
    limit_str += str_or_na(limit);

    std::string fanout_str;
    std::size_t fanout = 1;
    if ((obj->isPin() && obj->isOutput()) ||
        (obj->isPort() && obj->isInput())) {
      fanout = the_vertex->get_src_arcs().size();
      fanout_str = Str::printf("%d", fanout);
    }

    std::optional<double> fanout_slack;
    if (limit) {
      fanout_slack = *limit - fanout;
    }

    std::string slack_str;
    slack_str = str_or_na(fanout_slack);

    std::string cell_port_name;
    if (obj->isPin()) {
      auto* pin = dynamic_cast<Pin*>(obj);
      std::string cell_name =
          pin->get_cell_port()->get_ower_cell()->get_cell_name();
      std::string port_name = pin->get_cell_port()->get_port_name();

      cell_port_name = cell_name;
      cell_port_name += "/";
      cell_port_name += port_name;
    }

    std::string note = (fanout_slack && (*fanout_slack < 0.0)) ? "R" : "";

    (*report_tbl) << vertex_name << limit_str << fanout_str << slack_str
                  << cell_port_name << note << TABLE_ENDLINE;
  };

  unsigned i = 0;
  while (!fanout_data_queue.empty() && i < _n_worst) {
    auto* vertex = fanout_data_queue.top();
    auto* obj = vertex->get_design_obj();
    if (!obj->get_net()) {
      fanout_data_queue.pop();
      continue;
    }

    if ((obj->isPin() && obj->isOutput()) ||
        (obj->isPort() && obj->isInput())) {
      // print vertex fanout data.
      print_fanout_data(vertex);
      i++;
    }
    fanout_data_queue.pop();
  }

  // LOG_INFO << "\n" << report_tbl->c_str();

  auto close_file = [](std::FILE* fp) { std::fclose(fp); };

  std::unique_ptr<std::FILE, decltype(close_file)> f(
      std::fopen(_rpt_file_name, "w"), close_file);

  std::fprintf(f.get(), "Generate the report at %s, GitVersion: %s.\n",
               Time::getNowWallTime(), GIT_VERSION);
  std::fprintf(f.get(), "%s", report_tbl->c_str());

  return is_ok;
}

StaReportSkewSummary::StaReportSkewSummary(const char* rpt_file_name,
                                           AnalysisMode analysis_mode,
                                           unsigned n_worst)
    : _rpt_file_name(Str::copy(rpt_file_name)),
      _analysis_mode(analysis_mode),
      _n_worst(n_worst) {}

StaReportSkewSummary::~StaReportSkewSummary() { Str::free(_rpt_file_name); }

std::unique_ptr<StaReportTable> StaReportSkewSummary::createReportTable(
    const char* tbl_name) {
  auto report_tbl = std::make_unique<StaReportTable>(tbl_name);

  (*report_tbl) << TABLE_HEAD;
  /* Fill each cell with operator[] */
  (*report_tbl)[0][0] = "Clock Pin";
  (*report_tbl)[0][1] = "Latency";
  //(*report_tbl)[0][2] = "CRP";
  (*report_tbl)[0][2] = "Skew";
  (*report_tbl)[0][3] = "    ";

  (*report_tbl) << TABLE_ENDLINE;

  return report_tbl;
}

/**
 * @brief report the seq path data in summary.
 *
 * @param seq_path_data
 * @return unsigned
 */
unsigned StaReportSkewSummary::operator()(StaSeqPathData* seq_path_data) {
  unsigned is_ok = 1;

  if (!_report_tbl) {
    _report_tbl =
        createReportTable(seq_path_data->get_capture_clock()->get_clock_name());
  }

  auto* launch_clock_data = seq_path_data->get_launch_clock_data();
  auto* capture_clock_data = seq_path_data->get_capture_clock_data();

  auto* launch_clock_vertex = launch_clock_data->get_own_vertex();
  auto* capture_clock_vertex = capture_clock_data->get_own_vertex();

  auto fix_point_str = [](double data) { return Str::printf("%.3f", data); };

  auto get_trans_info = [](auto* vertex) {
    auto trans_type =
        vertex->isRisingTriggered() ? TransType::kRise : TransType::kFall;
    const char* trans_type_str =
        (trans_type == TransType::kRise) ? "rp-+" : "fp-+";
    return trans_type_str;
  };

  (*_report_tbl) << launch_clock_vertex->getNameWithCellName()
                 << fix_point_str(
                        FS_TO_NS(launch_clock_data->get_arrive_time()))
                 << TABLE_SKIP << get_trans_info(launch_clock_vertex)
                 << TABLE_ENDLINE;

  (*_report_tbl) << capture_clock_vertex->getNameWithCellName()
                 << fix_point_str(
                        FS_TO_NS(capture_clock_data->get_arrive_time()))
                 << fix_point_str(FS_TO_NS(seq_path_data->getSkew()))
                 << get_trans_info(capture_clock_vertex) << TABLE_ENDLINE;

  // LOG_INFO << "\n" << _report_tbl->c_str();

  return is_ok;
}

/**
 * @brief report the path group skew.
 *
 * @param seq_path_group
 * @return unsigned
 */
unsigned StaReportSkewSummary::operator()(StaSeqPathGroup* seq_path_group) {
  unsigned is_ok = 1;

  auto cmp = [this](StaPathData* left, StaPathData* right) -> bool {
    int left_skew = left->getSkew();
    int right_skew = right->getSkew();
    return _analysis_mode == AnalysisMode::kMax ? (left_skew > right_skew)
                                                : (left_skew < right_skew);
  };

  std::priority_queue<StaPathData*, std::vector<StaPathData*>, decltype(cmp)>
      seq_data_queue(cmp);

  std::set<std::pair<StaClockData*, StaClockData*>> lanuch_capture_pairs;

  StaPathEnd* path_end;
  StaPathData* path_data;
  AnalysisMode analysis_mode = _analysis_mode;
  FOREACH_PATH_GROUP_END(seq_path_group, path_end)
  FOREACH_PATH_END_DATA(path_end, analysis_mode, path_data) {
    auto launch_capture_pair =
        std::make_pair(path_data->get_launch_clock_data(),
                       path_data->get_capture_clock_data());
    if (path_data->get_launch_clock_data()->get_own_vertex()->is_port() == 1) {
      continue;
    }
    if (!lanuch_capture_pairs.contains(launch_capture_pair)) {
      lanuch_capture_pairs.insert(launch_capture_pair);
      seq_data_queue.push(path_data);
    }
  }

  unsigned i = 0;
  while (!seq_data_queue.empty() && i < get_n_worst()) {
    auto* seq_path_data = dynamic_cast<StaSeqPathData*>(seq_data_queue.top());
    is_ok = (*this)(seq_path_data);
    if (!is_ok) {
      break;
    }

    seq_data_queue.pop();
    i++;
  }

  return is_ok;
}

/**
 * @brief report the skew violation.
 *
 * @param ista
 * @return unsigned
 */
unsigned StaReportSkewSummary::operator()(Sta* ista) {
  auto report_path = [this](Sta* ista) -> unsigned {
    unsigned is_ok = 1;

    for (auto&& [capture_clock, seq_path_group] : ista->get_clock_groups()) {
      is_ok = (*this)(seq_path_group.get());
      if (!is_ok) {
        break;
      }

      if (_report_tbl) {
        _report_path_skews.emplace_back(std::move(_report_tbl));
      }
    }
    return is_ok;
  };

  auto report_path_of_mode = [&report_path, this,
                              ista](AnalysisMode mode) -> unsigned {
    unsigned is_ok = 1;
    if ((ista->get_analysis_mode() == mode) ||
        (ista->get_analysis_mode() == AnalysisMode::kMaxMin)) {
      is_ok = report_path(ista);
    }

    return is_ok;
  };

  unsigned is_ok = report_path_of_mode(_analysis_mode);
  return is_ok;
}

StaReportSkewDetail::StaReportSkewDetail(const char* rpt_file_name,
                                         AnalysisMode analysis_mode,
                                         unsigned n_worst)
    : StaReportSkewSummary(rpt_file_name, analysis_mode, n_worst) {}

std::unique_ptr<StaReportTable> StaReportSkewDetail::createReportTable(
    const char* tbl_name) {
  auto report_tbl = std::make_unique<StaReportTable>(tbl_name);

  (*report_tbl) << TABLE_HEAD;
  /* Fill each cell with operator[] */
  (*report_tbl)[0][0] = "Point";
  (*report_tbl)[0][1] = "Fanout";
  (*report_tbl)[0][2] = "Capacitance";
  (*report_tbl)[0][3] = "Resistance";
  (*report_tbl)[0][4] = "Transition";
  (*report_tbl)[0][5] = "Incr";
  (*report_tbl)[0][6] = "Path";

  (*report_tbl) << TABLE_ENDLINE;

  return report_tbl;
}

/**
 * @brief report the seq path skew.
 *
 * @param seq_path_data
 * @return unsigned
 */
unsigned StaReportSkewDetail::operator()(StaSeqPathData* seq_path_data) {
  unsigned is_ok = 1;

  auto report_tbl = createReportTable("path");

  auto fix_point_str = [](double data) { return Str::printf("%.3f", data); };

  auto print_path_data = [&report_tbl, &fix_point_str](
                             auto& path_stack, auto clock_path_arrive_time) {
    double last_arrive_time = 0;
    StaVertex* last_vertex = nullptr;
    while (!path_stack.empty()) {
      auto* path_delay_data = path_stack.top();
      auto* own_vertex = path_delay_data->get_own_vertex();

      // print net
      if (auto* obj = own_vertex->get_design_obj();
          last_vertex &&
          ((obj->isPin() && obj->isInput() && !own_vertex->is_assistant()) ||
           (obj->isPort() && obj->isOutput() && own_vertex->is_assistant()))) {
        auto snk_arcs = last_vertex->getSnkArc(own_vertex);
        LOG_FATAL_IF(snk_arcs.size() != 1)
            << last_vertex->getName() << " " << own_vertex->getName()
            << " net arc found " << snk_arcs.size() << " arc.";
        if (snk_arcs.size() == 1) {
          auto* net_arc = dynamic_cast<StaNetArc*>(snk_arcs.front());
          auto* net = net_arc->get_net();

          (*report_tbl) << Str::printf("%s (net)", net->get_name())
                        << net->getLoads().size() << TABLE_SKIP << TABLE_SKIP
                        << TABLE_SKIP << TABLE_SKIP << TABLE_ENDLINE;
        }
      }

      auto arrive_time = FS_TO_NS(path_delay_data->get_arrive_time());

      auto trans_type = path_delay_data->get_trans_type();
      if (own_vertex->is_clock()) {
        trans_type = own_vertex->isRisingTriggered() ? TransType::kRise
                                                     : TransType::kFall;
      }

      const char* trans_type_str = (trans_type == TransType::kRise) ? "r" : "f";
      auto incr_time = arrive_time - last_arrive_time;
      last_arrive_time = arrive_time;

      auto vertex_load =
          own_vertex->getLoad(path_delay_data->get_delay_type(), trans_type);
      auto vertex_resistance = own_vertex->getResistance(
          path_delay_data->get_delay_type(), trans_type);
      auto vertex_slew =
          own_vertex->getSlewNs(path_delay_data->get_delay_type(), trans_type);

      (*report_tbl) << own_vertex->getNameWithCellName() << TABLE_SKIP
                    << fix_point_str(vertex_load)
                    << fix_point_str(vertex_resistance)
                    << fix_point_str(vertex_slew ? *vertex_slew : 0.0)
                    << fix_point_str(incr_time)
                    << Str::printf("%.3f%s",
                                   arrive_time + clock_path_arrive_time,
                                   trans_type_str)
                    << TABLE_ENDLINE;

      last_vertex = own_vertex;
      path_stack.pop();
    }
  };

  auto print_clock_data_info = [&report_tbl, seq_path_data, &print_path_data,
                                &fix_point_str, this]() {
    std::stack<StaPathDelayData*> path_stack =
        seq_path_data->getPathDelayData();
    /*The arrive time*/
    auto* path_delay_data = path_stack.top();
    auto* launch_clock_data = path_delay_data->get_launch_clock_data();

    auto* launch_clock = launch_clock_data->get_prop_clock();
    char* launch_clock_info =
        Str::printf("clock %s (%s)", launch_clock->get_clock_name(),
                    launch_clock_data->get_clock_wave_type() == TransType::kRise
                        ? "rise edge"
                        : "fall edge");
    auto launch_edge = FS_TO_NS(seq_path_data->getLaunchEdge());
    (*report_tbl) << launch_clock_info << TABLE_SKIP << TABLE_SKIP << TABLE_SKIP
                  << launch_edge << launch_edge << TABLE_ENDLINE;

    auto launch_clock_path_data_stack = launch_clock_data->getPathData();
    print_path_data(launch_clock_path_data_stack, 0.0);

    (*report_tbl) << TABLE_ENDLINE;

    /*The require time*/
    auto* capture_clock = seq_path_data->get_capture_clock();
    auto* capture_clock_data = seq_path_data->get_capture_clock_data();

    char* capture_clock_info = Str::printf(
        "clock %s (%s)", capture_clock->get_clock_name(),
        capture_clock_data->get_clock_wave_type() == TransType::kRise
            ? "rise edge"
            : "fall edge");
    auto capture_edge = FS_TO_NS(seq_path_data->getCaptureEdge());
    (*report_tbl) << capture_clock_info << TABLE_SKIP << TABLE_SKIP
                  << TABLE_SKIP << capture_edge << capture_edge
                  << TABLE_ENDLINE;

    auto capture_clock_path_data_stack = capture_clock_data->getPathData();
    print_path_data(capture_clock_path_data_stack, 0.0);

    (*report_tbl) << TABLE_ENDLINE;

    /*The slack summary*/
    (*report_tbl) << "startpoint clock latency" << TABLE_SKIP << TABLE_SKIP
                  << TABLE_SKIP << TABLE_SKIP
                  << fix_point_str(
                         FS_TO_NS(launch_clock_data->get_arrive_time()))
                  << TABLE_ENDLINE;

    (*report_tbl) << "endpoint clock latency" << TABLE_SKIP << TABLE_SKIP
                  << TABLE_SKIP << TABLE_SKIP
                  << fix_point_str(
                         FS_TO_NS(capture_clock_data->get_arrive_time()))
                  << TABLE_ENDLINE;

    auto cppr = seq_path_data->get_cppr();
    auto skew = seq_path_data->getSkew();
    int cppr1 = 0;
    AnalysisMode analysis_mode = get_analysis_mode();
    if (cppr) {
      analysis_mode == AnalysisMode::kMax ? cppr1 = (*cppr) : cppr1 = -(*cppr);
    }

    (*report_tbl) << "cppr" << TABLE_SKIP << TABLE_SKIP << TABLE_SKIP
                  << TABLE_SKIP << fix_point_str(FS_TO_NS(cppr1))
                  << TABLE_ENDLINE;
    (*report_tbl) << "skew" << TABLE_SKIP << TABLE_SKIP << TABLE_SKIP
                  << TABLE_SKIP << fix_point_str(FS_TO_NS(skew))
                  << TABLE_ENDLINE;
  };

  print_clock_data_info();

  // LOG_INFO << "\n" << report_tbl->c_str();

  _report_path_skews.emplace_back(std::move(report_tbl));

  return is_ok;
}

StaReportSpecifyPath::StaReportSpecifyPath(const char* rpt_file_name,
                                           AnalysisMode analysis_mode,
                                           const char* from,
                                           const char* through, const char* to)
    : _rpt_file_name(rpt_file_name),
      _analysis_mode(analysis_mode),
      _from(from),
      _through(through),
      _to(to) {}

/**
 * @brief report the specified path.
 *
 * @param ista
 * @return unsigned
 */
unsigned StaReportSpecifyPath::operator()(Sta* ista) {
  unsigned is_ok = true;
  for (auto&& [capture_clock, seq_path_group] : ista->get_clock_groups()) {
    is_ok = (*this)(seq_path_group.get());
    if (!is_ok) {
      break;
    }
  }

  return is_ok;
}

/**
 * @brief search the specified path from the group.
 *
 * @param seq_path_group
 * @return unsigned
 */
unsigned StaReportSpecifyPath::operator()(StaSeqPathGroup* seq_path_group) {
  unsigned is_ok = 1;

  StaPathEnd* path_end;
  StaPathData* path_data;
  FOREACH_PATH_GROUP_END(seq_path_group, path_end)
  FOREACH_PATH_END_DATA(path_end, _analysis_mode, path_data) {
    auto path_delay_data_vec = path_data->getPathDelayData();
    auto* path_delay_data = path_delay_data_vec.top();
    if (_from) {
      if (path_delay_data->get_own_vertex()->getName() != _from) {
        continue;
      }
    }

    bool is_match = _through ? false : true;
    path_delay_data_vec.pop();
    while (!path_delay_data_vec.empty()) {
      path_delay_data = path_delay_data_vec.top();
      if (_through) {
        if (path_delay_data->get_own_vertex()->getName() == _through) {
          is_match = true;
        }
      }

      path_delay_data_vec.pop();
    }

    if (_to) {
      if (path_delay_data->get_own_vertex()->getName() != _to) {
        continue;
      }
    }

    if (is_match) {
      StaReportPathDetail report_path_detail(_rpt_file_name, _analysis_mode, 1,
                                             false);
      is_ok = report_path_detail(dynamic_cast<StaSeqPathData*>(path_data));
      if (!is_ok) {
        break;
      }
    }
  }

  return is_ok;
}

}  // namespace ista
