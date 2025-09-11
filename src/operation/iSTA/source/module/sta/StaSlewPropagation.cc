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
 * @file StaSlewPropagation.cc
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The slew propagation implemention from input port.
 * @version 0.1
 * @date 2021-04-08
 */
#include "StaSlewPropagation.hh"

#include <optional>

#include "ThreadPool/ThreadPool.h"
#include "delay/ReduceDelayCal.hh"
#include "netlist/Pin.hh"
#include "netlist/Port.hh"

namespace ista {

/**
 * @brief The slew propagation from the arc.
 *
 * @param the_arc
 * @return unsigned  1 if success, 0 else fail.
 */
unsigned StaSlewPropagation::operator()(StaArc* the_arc) {
  auto flip_trans_type = [](auto trans_type) {
    return trans_type == TransType::kRise ? TransType::kFall : TransType::kRise;
  };

  auto construct_slew_data =
      [this](AnalysisMode delay_type, TransType trans_type,
             StaVertex* own_vertex, int slew,
             std::unique_ptr<LibCurrentData> output_current_data,
             StaData* src_slew_data) -> void {
    StaSlewData* slew_data = nullptr;
    if (isIncremental()) {
      // find the exist data.
      slew_data =
          own_vertex->getSlewData(delay_type, trans_type, src_slew_data);
    }

    if (!slew_data) {
      slew_data = new StaSlewData(delay_type, trans_type, own_vertex, slew);

      slew_data->set_bwd(src_slew_data);
      src_slew_data->add_fwd(slew_data);

      slew_data->set_output_current_data(std::move(output_current_data));
      own_vertex->addData(slew_data);
    }

    if (isIncremental()) {
      slew_data->set_slew(slew);
      slew_data->set_output_current_data(std::move(output_current_data));
    }
  };

  unsigned is_ok = 1;

  auto* src_vertex = the_arc->get_src();
  auto* snk_vertex = the_arc->get_snk();

  auto* obj = snk_vertex->get_design_obj();

  /*The check arc and output port do not generate output slew.*/
  if (!the_arc->isDelayArc() || obj->isPort()) {
    return is_ok;
  }

  auto* the_pin = dynamic_cast<Pin*>(obj);
  LOG_FATAL_IF(!the_pin) << "obj " << obj->getFullName() << " is not a pin";
  auto* the_net = the_pin->get_net();
  LOG_FATAL_IF(!the_net) << "The pin " << the_pin->getFullName()
                         << " has not connect net.";

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

        auto out_trans_type =
            the_arc->isNegativeArc() ? flip_trans_type(trans_type) : trans_type;

        auto* rc_net = getSta()->getRcNet(the_net);
        auto trans_to_index = [](TransType trans_type) -> int {
          return static_cast<int>(trans_type) - 1;
        };

        std::array<double, 2> load_array;  // rise, fall load.

        for (auto load_trans_type : {TransType::kRise, TransType::kFall}) {
          auto load_pf = rc_net
                             ? rc_net->load(analysis_mode, out_trans_type)
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

        auto slew_values =
            lib_arc_set->getSlewNs(trans_type, out_trans_type, in_slew,
                                   load_array[trans_to_index(out_trans_type)]);
        double out_slew_ns = analysis_mode == AnalysisMode::kMax
                                 ? slew_values.front()
                                 : slew_values.back();

        auto output_current = lib_arc->getOutputCurrent(
            out_trans_type, in_slew,
            load_array[trans_to_index(out_trans_type)]);

        construct_slew_data(analysis_mode, out_trans_type, snk_vertex,
                            NS_TO_FS(out_slew_ns), std::move(output_current),
                            slew_data);

        /*The non-unate arc or tco should split two.*/
        if (!the_arc->isUnateArc() || the_arc->isTwoTypeSenseArc() || src_vertex->is_clock()) {
          auto out_trans_type1 = flip_trans_type(trans_type);

          // fix the timing type not match the trans type, which would lead to
          // crash.
          if (!lib_arc_set->isMatchTimingType(out_trans_type1)) {
            continue;
          }

          auto slew_values =
              lib_arc_set->getSlewNs(trans_type, out_trans_type1, in_slew,
                                 load_array[trans_to_index(out_trans_type1)]);
          double out_slew1_ns = analysis_mode == AnalysisMode::kMax
                                 ? slew_values.front()
                                 : slew_values.back();

          auto output_current1 = lib_arc->getOutputCurrent(
              out_trans_type1, in_slew,
              load_array[trans_to_index(out_trans_type1)]);

          construct_slew_data(analysis_mode, out_trans_type1, snk_vertex,
                              NS_TO_FS(out_slew1_ns),
                              std::move(output_current1), slew_data);
        }
      } else {  // net arc
        auto* rc_net = getSta()->getRcNet(the_net);
        auto net_out_slew =
            rc_net ? rc_net->slew(*the_pin, NS_TO_PS(in_slew),
                                  from_slew_data->get_output_current_data(),
                                  analysis_mode, trans_type)
                   : std::nullopt;

        double out_slew_ps = net_out_slew ? *net_out_slew : NS_TO_PS(in_slew);
        auto out_slew = PS_TO_FS(out_slew_ps);
        construct_slew_data(analysis_mode, trans_type, snk_vertex, out_slew,
                            nullptr, slew_data);
      }
    }
  }
  return is_ok;
}

/**
 * @brief The slew propagation from the vertex.
 *
 * @param the_vertex
 * @return unsigned 1 if success, 0 else fail.
 */
unsigned StaSlewPropagation::operator()(StaVertex* the_vertex) {
  std::lock_guard<std::mutex> lk(the_vertex->get_fwd_mutex());

  if (the_vertex->is_slew_prop() || the_vertex->is_const()) {
    if (isTracePath()) {
      addTracePathVertex(the_vertex);
    }

    return 1;
  }

  unsigned is_ok = 1;

  if ((the_vertex->is_clock() && the_vertex->is_ideal_clock_latency()) ||
      (the_vertex->is_port() && the_vertex->is_start()) ||
      the_vertex->is_sdc_clock_pin() || the_vertex->get_snk_arcs().empty()) {
    auto* obj = the_vertex->get_design_obj();

    LOG_FATAL_IF(
        !(obj->isPort() && obj->isInput()) &&
        !(obj->isPin() && obj->isOutput()) &&
        !(the_vertex->is_clock() && the_vertex->is_ideal_clock_latency()) &&
        !the_vertex->is_sdc_clock_pin() && !the_vertex->get_snk_arcs().empty())
        << "slew propgation start point " << obj->getFullName()
        << " is not input port or output pin.";

    VLOG(1) << "slew propgation start point " << obj->getFullName();

    the_vertex->set_is_slew_prop();
    the_vertex->initSlewData();

    if (isTracePath()) {
      addTracePathVertex(the_vertex);
    }

    return is_ok;
  }

  FOREACH_SNK_ARC(the_vertex, snk_arc) {
    if (!snk_arc->isDelayArc()) {
      continue;
    }

    if (snk_arc->is_loop_disable()) {
      continue;
    }

    auto* src_vertex = snk_arc->get_src();
    if (!src_vertex->exec(*this)) {
      return 0;
    }

    is_ok = snk_arc->exec(*this);
    if (!is_ok) {
      LOG_FATAL << "slew propgation error";
      break;
    }
  }

  if (isTracePath()) {
    addTracePathVertex(the_vertex);
  }

  the_vertex->set_is_slew_prop();

  return is_ok;
}

/**
 * @brief The slew propagation from the graph port vertex.
 *
 * @param the_graph
 * @return unsigned 1 if success, 0 else fail.
 */
unsigned StaSlewPropagation::operator()(StaGraph* the_graph) {
  LOG_INFO << "slew propagation start";
  unsigned is_ok = 1;
  {
#if 1
    // create thread pool
    unsigned num_threads = getNumThreads();
    ThreadPool pool(num_threads);
    StaVertex* end_vertex;

    FOREACH_END_VERTEX(the_graph, end_vertex) {
      if (end_vertex->get_snk_arcs().empty()) {
        continue;
      }
      // enqueue and store future
      pool.enqueue([](StaFunc& func,
                      StaVertex* end_vertex) { return end_vertex->exec(func); },
                   *this, end_vertex);
    }

#else

    StaVertex* end_vertex;
    FOREACH_END_VERTEX(the_graph, end_vertex) {
      if (end_vertex->get_snk_arcs().empty()) {
        continue;
      }
      is_ok = end_vertex->exec(*this);
      if (!is_ok) {
        break;
      }

      if (isTracePath()) {
        PrintTraceRecord();
        reset_is_trace_path();
      }
    }
#endif
  }

  LOG_INFO << "slew propagation end";

  return is_ok;
}

}  // namespace ista
