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
 * @file StaClockPropagation.cc
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The implemention of clock propagation.
 * @version 0.1
 * @date 2021-02-19
 */

#include "StaClockPropagation.hh"

#include "Sta.hh"
#include "StaApplySdc.hh"
#include "StaArc.hh"
#include "StaBuildClockTree.hh"
#include "log/Log.hh"

namespace ista {

/**
 * @brief propagate clock along the arc.
 *
 * @param the_arc
 * @param rise_data
 * @param fall_data
 * @return unsigned
 */
unsigned StaClockPropagation::propagateClock(StaArc* the_arc,
                                             StaClockData* rise_data,
                                             StaClockData* fall_data) {
  auto* snk_vertex = the_arc->get_snk();
  auto* src_vertex = the_arc->get_src();

  DesignObject* design_obj = src_vertex->get_design_obj();
  std::string obj_name;
  if (src_vertex->is_port()) {
    obj_name = design_obj->get_name();
  } else {
    Instance* inst = dynamic_cast<Pin*>(design_obj)->get_own_instance();
    obj_name = inst->get_inst_cell()->get_cell_name();
  }

  std::priority_queue<int, std::vector<int>, std::greater<int>> depth_min_queue;
  src_vertex->getPathDepth(depth_min_queue);
  int src_vertex_depth = depth_min_queue.top();

  // lambda function, get delay derate.
  auto get_delay_derate = [this](AnalysisMode delay_type,
                                 bool is_cell) -> std::optional<double> {
    auto* ista = getSta();
    auto& derate_table = ista->get_derate_table();
    if (delay_type == AnalysisMode::kMax) {
      if (is_cell) {
        return derate_table.getMaxClockCellDerate();
      } else {
        return derate_table.getMaxClockNetDerate();
      }
    } else {
      if (is_cell) {
        return derate_table.getMinClockCellDerate();
      } else {
        return derate_table.getMinClockNetDerate();
      }
    }
  };

  // lambda function, get delay aocv derate.
  auto get_aocv_delay_derate =
      [this](const char* object_name, AnalysisMode analysis_mode,
             TransType trans_type, AocvObjectSpec::DelayType delay_type,
             int depth) -> std::optional<float> {
    auto* ista = getSta();
    auto object_spec_set = ista->findClockAocvObjectSpecSet(object_name);

    if (object_spec_set) {
      auto* objec_spec =
          (*object_spec_set)
              ->get_object_spec(trans_type, analysis_mode, delay_type);
      auto depth2table = objec_spec->get_depth2table();
      float derate;
      depth2table ? derate = (*depth2table)[depth]
                  : derate = (float)(*(objec_spec->get_default_table()));
      return derate;
    } else {
      return std::nullopt;
    }
  };

  // DLOG_INFO << "clock propagate at vertex " << snk_vertex->getName();

  auto incr_arrive_time = [this, snk_vertex, &obj_name, &src_vertex_depth,
                           &get_delay_derate, &get_aocv_delay_derate](
                              StaArc* the_arc, StaClockData* orig_data,
                              StaClockData* new_data) -> void {
    int arc_delay = 0;
    if (!isIdealClock()) {
      arc_delay = the_arc->get_arc_delay(new_data->get_delay_type(),
                                         new_data->get_trans_type());
      auto derate =
          get_delay_derate(new_data->get_delay_type(), the_arc->isInstArc());

      // obj_name,vetex_depth(src,snk?),trans_type(new_data)
      auto aocv_derate = get_aocv_delay_derate(
          obj_name.c_str(), new_data->get_delay_type(),
          new_data->get_trans_type(), AocvObjectSpec::DelayType::kCell,
          src_vertex_depth);

      if (aocv_derate) {
        arc_delay *= aocv_derate.value();
        new_data->set_derate(aocv_derate.value());
      } else if (derate) {
        arc_delay *= derate.value();
        new_data->set_derate(derate.value());
      }

    } else {
      if (snk_vertex->is_clock()) {
        arc_delay = _propagate_clock->get_ideal_network_latency().value_or(0);
        snk_vertex->set_is_ideal_clock_latency();
        VLOG(1) << "The clock vertex " << snk_vertex->getName()
                << " set ideal network latency " << PS_TO_NS(arc_delay) << "ns";
      }
    }

    new_data->incrArriveTime(arc_delay);
    orig_data->add_fwd(new_data);
    new_data->set_bwd(orig_data);
  };

  auto construct_new_data = [this, the_arc, snk_vertex, &incr_arrive_time](
                                StaClockData* data) -> StaClockData* {
    if ((_prop_type == PropType::kIdealClockProp) &&
        !_propagate_clock->isIdealClockNetwork()) {
      return nullptr;
    }

    if (!data) {
      return nullptr;
    }

    auto* new_data = data->copy();
    // inv buffer
    if (the_arc->isNegativeArc()) {
      new_data->flipTransType();
    }

    incr_arrive_time(the_arc, data, new_data);

    snk_vertex->addData(new_data);
    return new_data;
  };

  if (the_arc->isNetArc()) {
    auto* net_arc = dynamic_cast<StaNetArc*>(the_arc);
    net_arc->get_net()->set_is_clock_net();
  }

  StaClockData* new_rise_data = nullptr;
  StaClockData* new_fall_data = nullptr;

  StaClockData* data1 = nullptr;
  StaClockData* data2 = nullptr;
  if (snk_vertex->is_clock()) {
    if (snk_vertex->isRisingTriggered()) {
      data1 = construct_new_data(rise_data);
    } else {
      data2 = construct_new_data(fall_data);
    }
  } else {
    data1 = construct_new_data(rise_data);
    data2 = construct_new_data(fall_data);
  }

  if (!the_arc->isNegativeArc()) {
    new_rise_data = data1;
    new_fall_data = data2;
  } else {
    new_rise_data = data2;
    new_fall_data = data1;
  }

  // The clock propagated endpoint should be the clock pin.
  if (snk_vertex->is_clock() && (!snk_vertex->is_clock_gate_clock())) {
    Sta* ista = getSta();
    StaGraph& the_graph = ista->get_graph();
    // The clock propagated endpoint is the start point of timing path.

    LOG_INFO_EVERY_N(10000)
        << "clock propagate end to vertex " << snk_vertex->getName();

    the_graph.addStartVertex(snk_vertex);
    return 1;
  }

  return propagateClock(snk_vertex, new_rise_data, new_fall_data);
}

/**
 * @brief Propagate clock data from the vertex, include the rise data and fall
 * data.
 *
 * @param the_vertex
 * @param rise_data
 * @param fall_data
 * @return unsigned
 */
unsigned StaClockPropagation::propagateClock(StaVertex* the_vertex,
                                             StaClockData* rise_data,
                                             StaClockData* fall_data) {
  auto get_vertex_own_instance = [](StaVertex* the_vertex) -> Instance* {
    Instance* inst;
    if (the_vertex->is_port()) {
      inst = nullptr;
    } else {
      DesignObject* design_obj = the_vertex->get_design_obj();
      inst = dynamic_cast<Pin*>(design_obj)->get_own_instance();
    }
    return inst;
  };

  unsigned is_ok = 1;
  FOREACH_SRC_ARC(the_vertex, src_arc) {
    if (!src_arc->isDelayArc()) {
      continue;
    }

    if (src_arc->is_disable_arc()) {
      continue;
    }

    auto* snk_vertex = src_arc->get_snk();
    Instance* the_inst = get_vertex_own_instance(the_vertex);
    if (the_inst &&
        the_inst->get_inst_cell()->get_is_clock_gating_integrated_cell() &&
        src_arc->isNetArc() && (!snk_vertex->is_clock())) {
      continue;
    }

    is_ok = propagateClock(src_arc, rise_data, fall_data);
    if (!is_ok) {
      break;
    }
  }
  return is_ok;
}

/**
 * @brief update the sdc generated clock after the KIdealClockProp.
 *
 */
void StaClockPropagation::updateSdcGeneratedClock() {
  Sta* ista = getSta();
  SdcConstrain* _the_constrain = ista->getConstrain();
  auto& the_clocks = _the_constrain->get_sdc_clocks();
  for (auto& [clock_name, the_clock] : the_clocks) {
    if (the_clock->isGenerateClock() &&
        (dynamic_cast<SdcGenerateCLock*>(the_clock.get())
             ->isNeedUpdateSourceClock())) {
      auto* the_generated_clock =
          dynamic_cast<SdcGenerateCLock*>(the_clock.get());
      auto source_pins = the_generated_clock->get_source_pins();
      int divide_by_value = the_generated_clock->get_divide_by();
      LOG_FATAL_IF(source_pins.size() != 1);
      StaClock* source_clock = nullptr;
      for (auto* source_pin : source_pins) {
        auto the_vertex = ista->findVertex(source_pin);
        LOG_FATAL_IF(!the_vertex) << "The vertex is not exist.";
        source_clock = the_vertex->getPropClock();
        LOG_FATAL_IF(!source_clock)
            << "The generated clock " << the_generated_clock->get_clock_name()
            << " source clock not found";
        the_generated_clock->set_source_name(source_clock->get_clock_name());
      }

      int generate_clock_period = source_clock->get_period() *
                                      divide_by_value;
      the_generated_clock->set_period(PS_TO_NS(generate_clock_period));
      auto& wave_form = source_clock->get_wave_form();
      ieda::Vector<double> edges;
      for (auto& wave_edge : wave_form.get_wave_edges()) {
        int edge = divide_by_value * wave_edge;
        edges.push_back(PS_TO_NS(edge));
      }

      if (the_generated_clock->isWaveformInv()) {
        size_t i = 0;
        double time_after_rising = edges[1] - edges[0];
        // invert by shift backward
        for (; i < edges.size() - 1; ++i) {
          edges[i] = edges[1 + i];
        }
        // add first time (time_after_rising) to last
        edges[i] += time_after_rising;
      }

      the_clock->set_edges(std::move(edges));
    }
  }
}

/**
 * @brief The functor of propagate clock.
 *
 * @param the_graph
 * @return unsigned
 */
unsigned StaClockPropagation::operator()(StaGraph* /* the_graph */) {
  if (_prop_type == PropType::kIdealClockProp) {
    LOG_INFO << "ideal clock propagation start";
  } else if (_prop_type == PropType::kNormalClockProp) {
    LOG_INFO << "propagated(kNormal) clock propagation start";
  } else {
    LOG_INFO << "propagated(kGenerated) clock propagation start";
  }

  Sta* ista = getSta();
  auto& clocks = ista->get_clocks();
  auto& the_graph = ista->get_graph();

  auto create_clock_data = [this](StaVertex* vertex, AnalysisMode analysis_mode,
                                  TransType trans_type, int edge,
                                  StaClock* sta_clock) -> StaClockData* {
    if ((_prop_type == PropType::kIdealClockProp) &&
        !_propagate_clock->isIdealClockNetwork()) {
      return nullptr;
    }

    StaClockData* clock_data =
        new StaClockData(analysis_mode, trans_type, edge, vertex, sta_clock);
    clock_data->set_clock_wave_type(trans_type);

    vertex->addData(clock_data);

    return clock_data;
  };

  unsigned is_ok = 1;

  LOG_INFO << "clock propagation clock num: " << clocks.size();

  for (auto& clock : clocks) {
    if (((_prop_type == PropType::kIdealClockProp) &&
         clock->isIdealClockNetwork()) ||
        ((_prop_type == PropType::kNormalClockProp) &&
         !clock->isIdealClockNetwork()) ||
        ((_prop_type == PropType::kUpdateGeneratedClockProp) &&
         !clock->isIdealClockNetwork() &&
         clock->isNeedUpdatePeriodWaveform())) {
      set_propagate_clock(clock.get());
      auto& vertexes = clock->get_clock_vertexes();
      for (auto* vertex : vertexes) {
        VLOG(1) << "clock " << clock->get_clock_name()
                << " propagate start from vertex " << vertex->getName();
        the_graph.removeStartVertex(vertex);

        auto* max_rise_clock_data = create_clock_data(
            vertex, AnalysisMode::kMax, TransType::kRise, 0, clock.get());

        auto* max_fall_clock_data = create_clock_data(
            vertex, AnalysisMode::kMax, TransType::kFall, 0, clock.get());

        is_ok =
            propagateClock(vertex, max_rise_clock_data, max_fall_clock_data);

        if (!is_ok) {
          break;
        }

        auto* min_rise_clock_data = create_clock_data(
            vertex, AnalysisMode::kMin, TransType::kRise, 0, clock.get());

        auto* min_fall_clock_data = create_clock_data(
            vertex, AnalysisMode::kMin, TransType::kFall, 0, clock.get());

        is_ok =
            propagateClock(vertex, min_rise_clock_data, min_fall_clock_data);

        if (!is_ok) {
          break;
        }
      }

      // reset propagate type.
      if (_prop_type == PropType::kUpdateGeneratedClockProp) {
        clock->set_is_need_update_period_waveform(false);
      }
    }
  }

  // for debug
  // for (auto& clock : clocks) {
  //   LOG_INFO << "build clock tree : " << clock->get_clock_name();
  //   StaBuildClockTree build_clock_tree;
  //   build_clock_tree(clock.get());

  //   auto& clock_trees = build_clock_tree.takeClockTrees();
  //   std::string path = std::string(clock->get_clock_name()) + "_tree.dot";
  //   clock_trees.front()->printInstGraphViz(path.c_str());
  //   clock_trees.clear();
  // }

  if (_prop_type == PropType::kNormalClockProp) {
    updateSdcGeneratedClock();
  }

  if (_prop_type == PropType::kIdealClockProp) {
    LOG_INFO << "ideal clock propagation end";
  } else if (_prop_type == PropType::kNormalClockProp) {
    LOG_INFO << "propagated(kNormal) clock propagation end";
  } else {
    LOG_INFO << "propagated(kGenerated) clock propagation end";
  }

  return is_ok;
}

}  // namespace ista
