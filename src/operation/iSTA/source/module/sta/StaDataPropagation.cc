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
 * @file StaDataPropagation.cc
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The implemention of data propagation.
 * @version 0.1
 * @date 2021-03-10
 */
// #include <gperftools/profiler.h>

#include "StaDataPropagation.hh"
#include "StaDataPropagationBFS.hh"

#include "StaData.hh"
#include "StaVertex.hh"
#include "ThreadPool/ThreadPool.h"
#include "log/Log.hh"

namespace ista {

/**
 * @brief Create the data path end require time.
 *
 * @param the_vertex
 * @return unsigned
 */
unsigned StaBwdPropagation::createEndData(StaVertex* the_vertex) {
  LOG_FATAL_IF(!the_vertex->is_end());
  unsigned is_ok = 1;

  auto* ista = getSta();

  StaData* delay_data;
  FOREACH_DELAY_DATA(the_vertex, delay_data) {
    // auto analysis_mode = delay_data->get_delay_type();
    // auto trans_type = delay_data->get_trans_type();

    auto seq_data_vec = ista->getSeqData(the_vertex, delay_data);
    if (!seq_data_vec.empty()) {
      std::sort(seq_data_vec.begin(), seq_data_vec.end(),
                [](auto* lhs, auto* rhs) {
                  return lhs->getSlack() < rhs->getSlack();
                });
      auto* seq_data = seq_data_vec.front();
      auto slack = seq_data->getSlack();
      LOG_INFO_IF_EVERY_N(slack < 0, 10)
          << "the endpoint vertex " << the_vertex->getName()
          << " has negative slack";
      auto arrive_time = delay_data->get_arrive_time();
      auto req_time = (delay_data->get_delay_type() == AnalysisMode::kMax)
                          ? arrive_time + slack
                          : arrive_time - slack;
      delay_data->set_req_time(req_time);
    }
  }

  return is_ok;
}

/**
 * @brief backward for the arc.
 *
 * @param the_arc
 * @return unsigned
 */
unsigned StaBwdPropagation::operator()(StaArc* the_arc) {
  auto* src_vertex = the_arc->get_src();
  StaVertex* snk_vertex = the_arc->get_snk();

  auto compare_signature = [the_arc](const StaData* lhs,
                                     const StaData* rhs) -> bool {
    if (lhs->get_delay_type() != rhs->get_delay_type()) {
      return false;
    }

    if (the_arc->isPositiveArc()) {
      if (lhs->get_trans_type() != rhs->get_trans_type()) {
        return false;
      }
    } else if (the_arc->isNegativeArc()) {
      if (lhs->get_trans_type() == rhs->get_trans_type()) {
        return false;
      }
    }

    auto* lhs_launch_clock_data =
        dynamic_cast<const StaPathDelayData*>(lhs)->get_launch_clock_data();
    auto* rhs_launch_clock_data =
        dynamic_cast<const StaPathDelayData*>(rhs)->get_launch_clock_data();

    if (lhs_launch_clock_data->get_prop_clock() !=
        rhs_launch_clock_data->get_prop_clock()) {
      return false;
    }

    if (lhs_launch_clock_data->get_clock_wave_type() !=
        rhs_launch_clock_data->get_clock_wave_type()) {
      return false;
    }

    return true;
  };

  StaData* delay_data;
  FOREACH_DELAY_DATA(snk_vertex, delay_data) {
    StaData* src_data;
    FOREACH_DELAY_DATA(src_vertex, src_data) {
      if (compare_signature(delay_data, src_data)) {
        auto analysis_mode = src_data->get_delay_type();
        auto trans_type = src_data->get_trans_type();
        int arc_delay = the_arc->get_arc_delay(analysis_mode, trans_type);

        if (delay_data->get_req_time()) {
          int req_time = *(delay_data->get_req_time()) - arc_delay;
          auto old_req_time = src_data->get_req_time();
          if (!old_req_time) {
            src_data->set_req_time(req_time);
          } else {
            if (analysis_mode == AnalysisMode::kMax) {
              // for the setup, req time is more criticaller when req time is
              // smaller.
              if (req_time < *(old_req_time)) {
                src_data->set_req_time(req_time);
              }

            } else {
              // for the hold, req time is more criticaller when req time is
              // larger.
              if (req_time > *(old_req_time)) {
                src_data->set_req_time(req_time);
              }
            }
          }
        }
      }
    }
  }

  return 1;
}

/**
 * @brief Propagate forward to the end vertex.
 *
 * @param the_vertex
 * @return unsigned
 */
unsigned StaBwdPropagation::operator()(StaVertex* the_vertex) {
  std::lock_guard<std::mutex> lk(the_vertex->get_bwd_mutex());
  unsigned is_ok = 1;

  if (the_vertex->is_bwd() || the_vertex->is_const()) {
    return 1;
  }

  if (the_vertex->is_end()) {
    DLOG_INFO_FIRST_N(10) << "Thread " << std::this_thread::get_id()
                          << " data bwd propagate found end vertex."
                          << the_vertex->getName();
    the_vertex->set_is_bwd();
    return createEndData(the_vertex);
  }

  FOREACH_SRC_ARC(the_vertex, src_arc) {
    if (!src_arc->isDelayArc()) {
      continue;
    }
    if (src_arc->is_loop_disable()) {
      continue;
    }

    if (src_arc->is_disable_arc()) {
      continue;
    }

    auto* snk_vertex = src_arc->get_snk();

    // for power gate start loop.
    if (snk_vertex->is_start()) {
      continue;
    }

    if (!snk_vertex->get_prop_tag().is_prop()) {
      continue;
    }

    if (!snk_vertex->exec(*this)) {
      return 0;
    }

    if (!src_arc->exec(*this)) {
      LOG_FATAL << "data propgation error";
      is_ok = 0;
      break;
    }
  }

  the_vertex->set_is_bwd();

  return is_ok;
}

/**
 * @brief bwd propagation for the graph.
 *
 * @param the_graph
 * @return unsigned
 */
unsigned StaBwdPropagation::operator()(StaGraph* the_graph) {
  unsigned num_threads = getNumThreads();
  unsigned is_ok = 1;
  auto* ista = getSta();

  ThreadPool pool(num_threads);
  StaBwdPropagation bwd_propagation;
  StaVertex* start_vertex;

  FOREACH_START_VERTEX(the_graph, start_vertex) {
    // not constrained port not propagation.
    if (start_vertex->is_port() &&
        ista->getIODelayConstrain(start_vertex).empty()) {
      continue;
    }

#if 0
    // enqueue and store future
    pool.enqueue(
        [](StaFunc& func, StaVertex* start_vertex) {
          return start_vertex->exec(func);
        },
        bwd_propagation, start_vertex);
#else

    LOG_INFO_EVERY_N(200) << "propagate start vertex "
                          << start_vertex->getName();

    is_ok = start_vertex->exec(bwd_propagation);
    if (!is_ok) {
      break;
    }

#endif
  }

  return 1;
}

/**
 * @brief Create the data path clock vertex start data.
 *
 * @param the_vertex
 * @return unsigned
 */
unsigned StaFwdPropagation::createClockVertexStartData(StaVertex* the_vertex) {
  AnalysisMode analysis_mode = get_analysis_mode();

  auto create_delay_data = [](StaVertex* the_vertex, AnalysisMode analysis_mode,
                              TransType trans_type) -> void {
    std::vector<StaData*> clock_data_vec;
    // DLOG_INFO << "start vertex " << the_vertex->getName();
    if (the_vertex->isRisingTriggered()) {
      clock_data_vec =
          the_vertex->getClockData(analysis_mode, TransType::kRise);
    } else {
      clock_data_vec =
          the_vertex->getClockData(analysis_mode, TransType::kFall);
    }

    if (clock_data_vec.empty()) {
      LOG_INFO_FIRST_N(10) << "start vertex " << the_vertex->getName()
                           << " have no clock data.";
      return;
    }

    LOG_FATAL_IF(clock_data_vec.empty());

    for (auto* clock_data : clock_data_vec) {
      StaPathDelayData* delay_data = new StaPathDelayData(
          analysis_mode, trans_type, 0, dynamic_cast<StaClockData*>(clock_data),
          the_vertex);
      the_vertex->addData(delay_data);
    }
  };

  if (AnalysisMode::kMax == analysis_mode ||
      AnalysisMode::kMaxMin == analysis_mode) {
    create_delay_data(the_vertex, AnalysisMode::kMax, TransType::kRise);
    create_delay_data(the_vertex, AnalysisMode::kMax, TransType::kFall);
  }

  if (AnalysisMode::kMin == analysis_mode ||
      AnalysisMode::kMaxMin == analysis_mode) {
    create_delay_data(the_vertex, AnalysisMode::kMin, TransType::kRise);
    create_delay_data(the_vertex, AnalysisMode::kMin, TransType::kFall);
  }

  return 1;
}

/**
 * @brief Create the data path port vertex start data.
 *
 * @param the_vertex
 * @return unsigned
 */
unsigned StaFwdPropagation::createPortVertexStartData(StaVertex* the_vertex) {
  auto construct_io_delay_data = [](AnalysisMode delay_type,
                                    TransType trans_type, StaVertex* own_vertex,
                                    double delay,
                                    StaClockData* launch_clock_data) {
    StaPathDelayData* path_delay_data = new StaPathDelayData(
        delay_type, trans_type, NS_TO_FS(delay), launch_clock_data, own_vertex);
    own_vertex->addData(path_delay_data);
  };

  auto* ista = getSta();

  the_vertex->set_is_start();
  ista->get_graph().addStartVertex(the_vertex);

  auto set_io_delays = ista->getIODelayConstrain(the_vertex);
  if (set_io_delays.empty()) {
    return 1;
  }

  for (auto* set_io_delay : set_io_delays) {
    unsigned is_clock_fall = set_io_delay->isClockFall();
    double delay = set_io_delay->get_delay_value();

    auto construct_delay_data =
        [is_clock_fall, delay, the_vertex, set_io_delay,
         &construct_io_delay_data](AnalysisMode analysis_mode) {
          StaClockData* launch_clock_data = nullptr;
          auto clock_datas =
              !is_clock_fall
                  ? the_vertex->getClockData(analysis_mode, TransType::kRise)
                  : the_vertex->getClockData(analysis_mode, TransType::kFall);
          LOG_FATAL_IF(clock_datas.empty() || (clock_datas.size() != 1))
              << "found clock data is not correct.";
          // should found one.
          launch_clock_data = dynamic_cast<StaClockData*>(clock_datas.front());

          if (set_io_delay->isRise()) {
            construct_io_delay_data(analysis_mode, TransType::kRise, the_vertex,
                                    delay, launch_clock_data);
          }

          if (set_io_delay->isFall()) {
            construct_io_delay_data(analysis_mode, TransType::kFall, the_vertex,
                                    delay, launch_clock_data);
          }
        };

    if (ista->isMaxAnalysis() && set_io_delay->isMax()) {
      construct_delay_data(AnalysisMode::kMax);
    }

    if (ista->isMinAnalysis() && set_io_delay->isMin()) {
      construct_delay_data(AnalysisMode::kMin);
    }
  }

  return 1;
}

/**
 * @brief Create the data path start data.
 *
 * @param the_vertex
 * @return unsigned
 */
unsigned StaFwdPropagation::createStartData(StaVertex* the_vertex) {
  unsigned is_ok = 1;
  if (the_vertex->get_prop_tag().is_prop()) {
    if (the_vertex->is_clock()) {
      is_ok = createClockVertexStartData(the_vertex);
    } else if (the_vertex->is_port()) {
      is_ok = createPortVertexStartData(the_vertex);
    }
  }

  return is_ok;
}

/**
 * @brief Generate the next data accord the timing arc.
 *
 * @param src_vertex
 * @param the_arc
 * @return unsigned
 */
unsigned StaFwdPropagation::operator()(StaArc* the_arc) {
  // lambda function, get delay derate.
  auto get_delay_derate = [this](AnalysisMode delay_type,
                                 bool is_cell) -> std::optional<double> {
    auto* ista = getSta();
    auto& derate_table = ista->get_derate_table();
    if (delay_type == AnalysisMode::kMax) {
      if (is_cell) {
        return derate_table.getMaxDataCellDerate();
      } else {
        return derate_table.getMaxDataNetDerate();
      }
    } else {
      if (is_cell) {
        return derate_table.getMinDataCellDerate();
      } else {
        return derate_table.getMinDataNetDerate();
      }
    }
  };

  // lambda function, get delay aocv derate.
  auto get_aocv_delay_derate =
      [this](const char* object_name, AnalysisMode anlysis_mode,
             TransType trans_type, AocvObjectSpec::DelayType delay_type,
             int depth) -> std::optional<float> {
    auto* ista = getSta();
    auto object_spec_set = ista->findDataAocvObjectSpecSet(object_name);

    if (object_spec_set) {
      auto* objec_spec =
          (*object_spec_set)
              ->get_object_spec(trans_type, anlysis_mode, delay_type);
      auto depth2table = objec_spec->get_depth2table();
      float derate;
      depth2table ? derate = (*depth2table)[depth]
                  : derate = (float)(*(objec_spec->get_default_table()));
      return derate;
    } else {
      return std::nullopt;
    }
  };

  auto* src_vertex = the_arc->get_src();
  auto* snk_vertex = the_arc->get_snk();
  unsigned src_vertex_depth = src_vertex->get_level();

  DesignObject* design_obj = src_vertex->get_design_obj();
  std::string obj_name;
  if (src_vertex->is_port()) {
    obj_name = design_obj->get_name();
  } else {
    Instance* inst = dynamic_cast<Pin*>(design_obj)->get_own_instance();
    obj_name = inst->get_inst_cell()->get_cell_name();
  }

  auto apply_derate_to_delay = [&get_delay_derate, &get_aocv_delay_derate,
                                &obj_name, &src_vertex_depth, the_arc](
                                   int arc_delay, StaData* next_data) -> int {
    auto derate =
        get_delay_derate(next_data->get_delay_type(), the_arc->isInstArc());
    auto aocv_derate = get_aocv_delay_derate(
        obj_name.c_str(), next_data->get_delay_type(),
        next_data->get_trans_type(), AocvObjectSpec::DelayType::kCell,
        src_vertex_depth);

    if (aocv_derate) {
      arc_delay *= aocv_derate.value();
      next_data->set_derate(aocv_derate.value());
    } else if (derate) {
      arc_delay *= derate.value();
      next_data->set_derate(derate.value());
    }
    return arc_delay;
  };

  StaData* delay_data;
  StaData* next_data1;
  StaData* next_data2;
  FOREACH_DELAY_DATA(src_vertex, delay_data) {
    next_data1 = nullptr;
    next_data2 = nullptr;

    // generate next data.
    if (isIncremental()) {
      auto trans_type = delay_data->get_trans_type();
      if (the_arc->isNegativeArc()) {
        trans_type = FLIP_TRANS(trans_type);
      }
      next_data1 = snk_vertex->getPathDelayData(delay_data->get_delay_type(),
                                                trans_type, delay_data);
    }

    if (!next_data1) {
      next_data1 = delay_data->copy();

      delay_data->add_fwd(next_data1);
      next_data1->set_bwd(delay_data);

      if (the_arc->isNegativeArc()) {
        next_data1->flipTransType();
      }

      auto arc_delay1 = the_arc->get_arc_delay(next_data1->get_delay_type(),
                                               next_data1->get_trans_type());
      arc_delay1 = apply_derate_to_delay(arc_delay1, next_data1);

      next_data1->set_arrive_time(delay_data->get_arrive_time() + arc_delay1);

      snk_vertex->addData(dynamic_cast<StaPathDelayData*>(next_data1));
    } else {
      auto arc_delay1 = the_arc->get_arc_delay(next_data1->get_delay_type(),
                                               next_data1->get_trans_type());
      arc_delay1 = apply_derate_to_delay(arc_delay1, next_data1);
      next_data1->set_arrive_time(delay_data->get_arrive_time() + arc_delay1);
    }

    // This is the non-unate arc, but for clk to q not need consider.
    if (!the_arc->isUnateArc() && !src_vertex->is_clock()) {
      if (isIncremental()) {
        auto trans_type = delay_data->get_trans_type();
        trans_type = FLIP_TRANS(trans_type);
        next_data2 = snk_vertex->getPathDelayData(delay_data->get_delay_type(),
                                                  trans_type, delay_data);
      }

      if (!next_data2) {
        next_data2 = delay_data->copy();
        delay_data->add_fwd(next_data2);
        next_data2->set_bwd(delay_data);

        next_data2->flipTransType();
        auto arc_delay2 = the_arc->get_arc_delay(next_data2->get_delay_type(),
                                                 next_data2->get_trans_type());
        arc_delay2 = apply_derate_to_delay(arc_delay2, next_data2);

        next_data2->set_arrive_time(delay_data->get_arrive_time() + arc_delay2);

        snk_vertex->addData(dynamic_cast<StaPathDelayData*>(next_data2));
      } else {
        auto arc_delay2 = the_arc->get_arc_delay(next_data2->get_delay_type(),
                                                 next_data2->get_trans_type());
        arc_delay2 = apply_derate_to_delay(arc_delay2, next_data2);
        next_data2->set_arrive_time(delay_data->get_arrive_time() + arc_delay2);
      }
    }
  }
  return 1;
}

/**
 * @brief Propagate backward to the start vertex.
 *
 * @param the_vertex
 * @return unsigned
 */
unsigned StaFwdPropagation::operator()(StaVertex* the_vertex) {
  std::lock_guard<std::mutex> lk(the_vertex->get_fwd_mutex());

  if (the_vertex->is_fwd() || the_vertex->is_const()) {
    return 1;
  }

  // DLOG_INFO << "Thread " << std::this_thread::get_id() << " propagate vertex
  // "
  //           << the_vertex->getName();

  if (the_vertex->is_start() && !isIncremental()) {
    DLOG_INFO_FIRST_N(10) << "Thread " << std::this_thread::get_id()
                          << " date fwd propagate found start vertex."
                          << the_vertex->getName();

    the_vertex->set_is_fwd();

    if (isTracePath()) {
      addTracePathVertex(the_vertex);
    }

    return createStartData(the_vertex);
  }

  if (isTracePath()) {
    addTracePathVertex(the_vertex);
  }

  FOREACH_SNK_ARC(the_vertex, snk_arc) {
    if (!snk_arc->isDelayArc()) {
      continue;
    }

    if (snk_arc->is_loop_disable()) {
      continue;
    }

    auto* src_vertex = snk_arc->get_src();

    if (!src_vertex->get_prop_tag().is_prop()) {
      continue;
    }

    if (!src_vertex->exec(*this)) {
      return 0;
    }

    if (!snk_arc->exec(*this)) {
      LOG_FATAL << "data propagation error";
      break;
    }
  }

  the_vertex->set_is_fwd();

  return 1;
}

/**
 * @brief fwd propagation of the graph.
 *
 * @param the_graph
 * @return unsigned
 */
unsigned StaFwdPropagation::operator()(StaGraph* the_graph) {
  unsigned is_ok = 1;
  unsigned num_threads = getNumThreads();
  // create thread pool
  ThreadPool pool(num_threads);
  StaVertex* end_vertex;
  FOREACH_END_VERTEX(the_graph, end_vertex) {
    if (!end_vertex->get_prop_tag().is_prop()) {
      continue;
    }
#if 1
    // enqueue and store future
    pool.enqueue(
        [this](StaVertex* end_vertex) { return end_vertex->exec(*this); },
        end_vertex);
#else

    VLOG_EVERY_N(1, 200) << "propagate end vertex " << end_vertex->getName();

    is_ok = end_vertex->exec(*this);
    if (!is_ok) {
      break;
    }

    if (isTracePath()) {
      PrintTraceRecord();
      reset_is_trace_path();
    }

#endif
  }

  return is_ok;
}

/**
 * @brief The arrive time and require time data propagation.
 *
 * @param the_graph
 * @return unsigned return 1 if success, else return 0.
 */
unsigned StaDataPropagation::operator()(StaGraph* the_graph) {
  unsigned is_ok = 1;
  auto* ista = getSta();
  auto prop_mode = ista->get_propagation_method();

  if ((_prop_type == PropType::kFwdProp) ||
      (_prop_type == PropType::kIncrFwdProp)) {
    LOG_INFO << "data fwd propagation start";
    // ProfilerStart("fwd_prop.prof");
    {
      if (prop_mode == PropagationMethod::kDFS) {
        StaFwdPropagation fwd_propagation;
        if (_prop_type == PropType::kIncrFwdProp) {
          fwd_propagation.set_is_incremental();
        }
        fwd_propagation(the_graph);
      } else {
        StaFwdPropagationBFS fwd_propagation_bfs;
        if (_prop_type == PropType::kIncrFwdProp) {
          fwd_propagation_bfs.set_is_incremental();
        }
        fwd_propagation_bfs(the_graph);
      }
    }

    // ProfilerStop();
    LOG_INFO << "data fwd propagation end";

  } else {
    LOG_INFO << "data bwd propagation start";
    // ProfilerStart("bwd_prop.prof");
    StaBwdPropagation bwd_propagation;
    bwd_propagation(the_graph);
    // ProfilerStop();
    LOG_INFO << "data bwd propagation end";
  }

  return is_ok;
}

}  // namespace ista
