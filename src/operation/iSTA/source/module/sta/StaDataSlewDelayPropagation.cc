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
 * @file StaDataSlewDelayPropagation.hh
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The Data slew delay propagation using BFS method.
 * @version 0.1
 * @date 2024-12-26
 */
#include "StaDataSlewDelayPropagation.hh"

#include "StaDelayPropagation.hh"
#include "StaSlewPropagation.hh"
#include "ThreadPool/ThreadPool.h"
#include "delay/ElmoreDelayCalc.hh"
#include "delay/ReduceDelayCal.hh"

namespace ista {

/**
 * @brief propagate the arc to calc slew and delay of the snk vertex.
 *
 * @param the_arc
 * @return unsigned
 */
unsigned StaDataSlewDelayPropagation::operator()(StaArc* the_arc) {
#if 0
  std::lock_guard<std::mutex> lk(the_arc->get_snk()->get_fwd_mutex());
  StaSlewPropagation slew_propagation;
  StaDelayPropagation delay_propagation;

  slew_propagation(the_arc);
  delay_propagation(the_arc);
#else
  auto flip_trans_type = [](auto trans_type) {
    return trans_type == TransType::kRise ? TransType::kFall : TransType::kRise;
  };

  auto construct_slew_delay_data =
      [this](AnalysisMode delay_type, TransType trans_type,
             StaVertex* own_vertex, StaArc* own_arc, int slew, int delay,
             std::unique_ptr<LibCurrentData> output_current_data,
             StaData* src_slew_data) -> void {
    StaSlewData* slew_data = nullptr;
    StaArcDelayData* arc_delay = nullptr;
    if (isIncremental()) {
      // find the exist data.
      slew_data =
          own_vertex->getSlewData(delay_type, trans_type, src_slew_data);
      arc_delay = own_arc->getArcDelayData(delay_type, trans_type);
    }

    if (!slew_data) {
      slew_data = new StaSlewData(delay_type, trans_type, own_vertex, slew);

      slew_data->set_bwd(src_slew_data);
      src_slew_data->add_fwd(slew_data);

      slew_data->set_output_current_data(std::move(output_current_data));
      own_vertex->addData(slew_data);
    }

    if (!arc_delay) {
      arc_delay = new StaArcDelayData(delay_type, trans_type, own_arc, delay);
      own_arc->addData(arc_delay);
    }

    if (isIncremental()) {
      slew_data->set_slew(slew);
      slew_data->set_output_current_data(std::move(output_current_data));
      arc_delay->set_arc_delay(delay);
    }
  };

  unsigned is_ok = 1;

  auto* src_vertex = the_arc->get_src();
  auto* snk_vertex = the_arc->get_snk();

  auto* obj = snk_vertex->get_design_obj();

  auto* the_net = obj->get_net();
  if (!the_net) {
    LOG_ERROR << "Not connect net in the design object: " << obj->get_name();
    the_arc->set_is_disable_arc(true);
    return is_ok;
  }

  StaData* slew_data;
  FOREACH_SLEW_DATA(src_vertex, slew_data) {
    auto trans_type = slew_data->get_trans_type();

    // for the clock vertex, only need the trigger transition type data.
    if ((snk_vertex->isRisingTriggered() && IS_FALL(trans_type)) ||
        (snk_vertex->isFallingTriggered() && IS_RISE(trans_type))) {
      continue;
    }

    auto analysis_mode = slew_data->get_delay_type();
    if (analysis_mode == get_analysis_mode() ||
        AnalysisMode::kMaxMin == get_analysis_mode()) {
      auto* from_slew_data = dynamic_cast<StaSlewData*>(slew_data);
      auto in_slew_fs = from_slew_data->get_slew();
      /*convert fs to ns*/
      double in_slew = FS_TO_NS(in_slew_fs);

      if (the_arc->isInstArc()) {
        auto* inst_arc = dynamic_cast<StaInstArc*>(the_arc);
        auto* lib_arc = inst_arc->get_lib_arc();
        auto* lib_arc_set =
            dynamic_cast<StaInstArc*>(the_arc)->get_lib_arc_set();

        if (the_arc->isCheckArc()) {
          // Since slew is fitter accord trigger type, May be do not need below
          // code
          if ((src_vertex->isRisingTriggered() && IS_FALL(trans_type)) ||
              (src_vertex->isFallingTriggered() && IS_RISE(trans_type))) {
            continue;
          }

          StaData* snk_slew_data;
          FOREACH_SLEW_DATA(snk_vertex, snk_slew_data) {
            if (snk_slew_data->get_delay_type() != analysis_mode) {
              continue;
            }

            auto snk_trans_type = snk_slew_data->get_trans_type();
            auto snk_slew_fs =
                dynamic_cast<StaSlewData*>(snk_slew_data)->get_slew();
            auto snk_slew = FS_TO_NS(snk_slew_fs);
            auto delay_values = lib_arc_set->getDelayOrConstrainCheckNs(
                trans_type, snk_trans_type, in_slew, snk_slew);
            double delay_ns = analysis_mode == AnalysisMode::kMax
                                  ? delay_values.front()
                                  : delay_values.back();
            auto delay = NS_TO_FS(delay_ns);

            StaArcDelayData* arc_delay = nullptr;
            if (isIncremental()) {
              arc_delay =
                  the_arc->getArcDelayData(analysis_mode, snk_trans_type);
            }

            if (!arc_delay) {
              arc_delay = new StaArcDelayData(analysis_mode, snk_trans_type,
                                              the_arc, delay);
            }
            the_arc->addData(arc_delay);
          }

        } else if (the_arc->isDelayArc()) {
          auto out_trans_type = the_arc->isNegativeArc()
                                    ? flip_trans_type(trans_type)
                                    : trans_type;

          auto* rc_net = getSta()->getRcNet(the_net);
          auto trans_to_index = [](TransType trans_type) -> int {
            return static_cast<int>(trans_type) - 1;
          };

          std::array<double, 2> load_array;  // rise, fall load.

          for (auto load_trans_type : {TransType::kRise, TransType::kFall}) {
            auto load_pf =
                rc_net ? rc_net->load(analysis_mode, out_trans_type)
                       : the_net->getLoad(analysis_mode, out_trans_type);
            auto* the_lib = lib_arc->get_owner_cell()->get_owner_lib();

            double load{0};
            if (the_lib->get_cap_unit() == CapacitiveUnit::kFF) {
              load = PF_TO_FF(load_pf);
            } else if (the_lib->get_cap_unit() == CapacitiveUnit::kPF) {
              load = load_pf;
            }

            load_array[trans_to_index(load_trans_type)] = load;
          }

          if (auto* arnoldi_net = dynamic_cast<ArnoldiNet*>(rc_net);
              arnoldi_net) {
            arnoldi_net->set_lib_arc(lib_arc);
          }

          // fix the timing type not match the trans type, which would lead to
          // crash.
          if (!lib_arc_set->isMatchTimingType(out_trans_type)) {
            continue;
          }

          auto slew_values = lib_arc_set->getSlewNs(
              trans_type, out_trans_type, in_slew,
              load_array[trans_to_index(out_trans_type)]);
          double out_slew_ns = analysis_mode == AnalysisMode::kMax
                                   ? slew_values.front()
                                   : slew_values.back();

          auto output_current = lib_arc->getOutputCurrent(
              out_trans_type, in_slew,
              load_array[trans_to_index(out_trans_type)]);

          auto delay_values = lib_arc_set->getDelayOrConstrainCheckNs(
              trans_type, out_trans_type, in_slew,
              load_array[trans_to_index(out_trans_type)]);
          double delay_ns = analysis_mode == AnalysisMode::kMax
                                ? delay_values.front()
                                : delay_values.back();
          auto delay = NS_TO_FS(delay_ns);

          construct_slew_delay_data(analysis_mode, out_trans_type, snk_vertex,
                                    the_arc, NS_TO_FS(out_slew_ns), delay,
                                    std::move(output_current), slew_data);

          /*The non-unate arc or tco should split two.*/
          if (!the_arc->isUnateArc() || the_arc->isTwoTypeSenseArc() || src_vertex->is_clock()) {
            auto out_trans_type1 = flip_trans_type(trans_type);

            // fix the timing type not match the trans type, which would lead to
            // crash.
            if (!lib_arc_set->isMatchTimingType(out_trans_type1)) {
              continue;
            }

            auto slew_values = lib_arc_set->getSlewNs(
                trans_type, out_trans_type1, in_slew,
                load_array[trans_to_index(out_trans_type1)]);
            double out_slew1_ns = analysis_mode == AnalysisMode::kMax
                                      ? slew_values.front()
                                      : slew_values.back();

            auto output_current1 = lib_arc->getOutputCurrent(
                out_trans_type1, in_slew,
                load_array[trans_to_index(out_trans_type1)]);

            auto delay_values = lib_arc_set->getDelayOrConstrainCheckNs(
                trans_type, out_trans_type1, in_slew,
                load_array[trans_to_index(out_trans_type1)]);
            double delay1_ns = analysis_mode == AnalysisMode::kMax
                                   ? delay_values.front()
                                   : delay_values.back();
            auto delay1 = NS_TO_FS(delay1_ns);

            construct_slew_delay_data(analysis_mode, out_trans_type1,
                                      snk_vertex, the_arc,
                                      NS_TO_FS(out_slew1_ns), delay1,
                                      std::move(output_current1), slew_data);
          }

        } else if (the_arc->isMpwArc()) {
          // TODO(to taosimin) fix mpw arc
          return is_ok;
        }

      } else {  // net arc
        auto* rc_net = getSta()->getRcNet(the_net);
        auto net_out_slew =
            rc_net ? rc_net->slew(*obj, NS_TO_PS(in_slew),
                                  from_slew_data->get_output_current_data(),
                                  analysis_mode, trans_type)
                   : std::nullopt;

        double out_slew_ps = net_out_slew ? *net_out_slew : NS_TO_PS(in_slew);
        auto out_slew = PS_TO_FS(out_slew_ps);

        auto output_current = from_slew_data->get_output_current_data();
        auto net_delay = rc_net ? rc_net->delay(*obj, in_slew, output_current,
                                                analysis_mode, trans_type)
                                : std::nullopt;
        auto delay_ps = net_delay ? net_delay->first : 0.0;
        auto delay = PS_TO_FS(delay_ps);
        construct_slew_delay_data(analysis_mode, trans_type, snk_vertex,
                                  the_arc, out_slew, delay, nullptr, slew_data);

        if (rc_net) {
          auto* arnoldi_rc_net = dynamic_cast<ArnoldiNet*>(rc_net);
          if (arnoldi_rc_net && net_delay) {
            auto* net_arc = dynamic_cast<StaNetArc*>(the_arc);
            auto node_waveform = net_delay->second;
            if (output_current) {
              auto [total_time, num_points] =
                  (*output_current)->getSimulationTotalTimeAndNumPoints();
              double step_time_ns = total_time / (num_points - 1);

              std::vector<Waveform> waveforms;
              using RowIdx = decltype(node_waveform.rows());
              for (RowIdx i = 0; i < node_waveform.rows(); ++i) {
                waveforms.emplace_back(step_time_ns, node_waveform.row(i));
              }

              auto* arc_waveform_data =
                  new StaArcWaveformData(analysis_mode, trans_type,
                                         from_slew_data, std::move(waveforms));
              net_arc->addWaveformData(arc_waveform_data);
            }
          }
        }
      }
    }
  }
  return is_ok;

#endif
}

/**
 * @brief propagate the vertex, and get the next bfs vertexes.
 *
 * @param the_vertex
 * @return unsigned
 */
unsigned StaDataSlewDelayPropagation::operator()(StaVertex* the_vertex) {
  if (the_vertex->is_const()) {
    return 1;
  }

  // data propagation end at the clock vertex.
  if (the_vertex->is_end()) {
    // calc check arc
    FOREACH_SNK_ARC(the_vertex, snk_arc) { snk_arc->exec(*this); }
  }

  unsigned is_ok = 1;
  FOREACH_SRC_ARC(the_vertex, src_arc) {
    if (!src_arc->isDelayArc()) {
      continue;
    }

    if (src_arc->is_loop_disable()) {
      continue;
    }

    is_ok = src_arc->exec(*this);
    if (!is_ok) {
      LOG_FATAL << "slew propgation error";
      break;
    }

    // get the next level bfs vertex and add it to the queue.
    auto* snk_vertex = src_arc->get_snk();
    if (snk_vertex->get_level() == (the_vertex->get_level() + 1)) {
      addNextBFSQueue(snk_vertex);
    }
  }

  the_vertex->set_is_slew_prop();
  the_vertex->set_is_delay_prop();

  return 1;
}

/**
 * @brief propagate from the clock source vertex.
 *
 * @return unsigned
 */
unsigned StaDataSlewDelayPropagation::operator()(StaGraph* the_graph) {
  ieda::Stats stats;
  LOG_INFO << "data slew delay propagation start";
  unsigned is_ok = 1;

  StaVertex* the_vertex;
  FOREACH_VERTEX(the_graph, the_vertex) {
    // start from the vertex which is level one and has slew prop.
    if (the_vertex->get_level() == 1) {
      // only propagate the vertex has slew.
      if (the_vertex->is_slew_prop()) {
        LOG_FATAL_IF(!the_vertex->is_delay_prop())
            << "the vertex should be delay propagated.";
        _bfs_queue.emplace_back(the_vertex);
      }
    }
  }

  // lambda for propagate the current queue.
  auto propagate_current_queue = [this](auto& current_queue) {
    LOG_INFO << "propagating current data queue vertexes number is "
             << current_queue.size();

#if 1
    {
      // create thread pool
      unsigned num_threads = getNumThreads();
      // unsigned num_threads = 1;
      ThreadPool pool(num_threads);

      for (auto* the_vertex : current_queue) {
        pool.enqueue(
            [this](StaVertex* the_vertex) { return the_vertex->exec(*this); },
            the_vertex);
      }
    }

#else
    for (auto* the_vertex : current_queue) {
      the_vertex->exec(*this);
    }

#endif
  };

  // do the bfs traverse for calc the clock slew/delay.
  do {
    propagate_current_queue(_bfs_queue);
    _bfs_queue.clear();

    // swap to the next bfs queue.
    std::swap(_bfs_queue, _next_bfs_queue);

  } while (!_bfs_queue.empty());

  LOG_INFO << "data slew delay propagation end";

  double memory_delta = stats.memoryDelta();
  LOG_INFO << "data slew delay propagation memory usage " << memory_delta
           << "MB";
  double time_delta = stats.elapsedRunTime();
  LOG_INFO << "data slew delay propagation time elapsed " << time_delta << "s";

  return is_ok;
}

}  // namespace ista