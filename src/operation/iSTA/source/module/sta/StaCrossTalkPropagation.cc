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
 * @file StaCrossTalkPropagation.cc
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The implemention of crosstalk delay propagation.
 * @version 0.1
 * @date 2022-10-31
 */
#include "StaCrossTalkPropagation.hh"

#include "ThreadPool/ThreadPool.h"
#include "delay/CrossTalkDelayCalc.hh"
#include "delay/ReduceDelayCal.hh"
#include "solver/Interpolation.hh"
#include "string/Str.hh"

namespace ista {

/**
 * @brief calc the specify point voltage of waveform.
 *
 * @param waveform
 * @param step_time
 * @param start_arrive_time
 * @param point_time
 * @return std::optional<double>
 */
std::optional<double> calcWaveformPointVoltage(const VectorXd& waveform,
                                               double step_time,
                                               double start_arrive_time,
                                               double point_time) {
  double time1;
  double time2;
  double v1;
  double v2;
  bool is_found = false;
  auto num_point = waveform.rows();
  for (int i = 1; i < num_point; ++i) {
    if (double tmp = start_arrive_time + i * step_time; tmp >= point_time) {
      time1 = tmp;
      v1 = waveform(i);
      time2 = tmp - step_time;
      v2 = waveform(i - 1);
      is_found = true;
      break;
    }
  }

  std::optional<double> time;
  if (is_found) {
    time = LinearInterpolate(time2, time1, v2, v1, point_time);
  }
  return time;
}

/**
 * @brief The timing arc propagation, we need calc net arc crosstalk.
 *
 * @param the_arc
 * @return unsigned
 */
unsigned StaCrossTalkPropagation::operator()(StaArc* the_arc) {
  unsigned is_ok = 1;
  auto* ista = getSta();
  auto* nl = ista->get_netlist();

  if (the_arc->isNetArc()) {
    auto* snk_vertex = the_arc->get_snk();
    auto* design_obj = snk_vertex->get_design_obj();
    if (design_obj->isPort()) {
      return 1;
    }

    auto* load_pin = dynamic_cast<Pin*>(design_obj);
    auto* local_net_arc = dynamic_cast<StaNetArc*>(the_arc);

    auto* the_net = local_net_arc->get_net();
    auto* local_rc_net = ista->getRcNet(the_net);
    if (!local_rc_net) {
      return 1;
    }
    auto* local_rc_tree = local_rc_net->rct();
    if (!local_rc_tree || !local_rc_tree->isHaveCoupledNodes()) {
      return 1;
    }

    auto* local_arnoldi_rc_net = dynamic_cast<ArnoldiNet*>(local_rc_net);
    auto* pin_node = local_rc_tree->node(load_pin->getFullName());
    LOG_FATAL_IF(!pin_node)
        << "not found pin rc node" << load_pin->getFullName();

    std::vector<CrossTalkDelayCalc> crosstalk_models;
    // for arc waveform data, how we choose which data to propagated, GBA should
    // be choosed.
    StaData* local_arc_waveform_data;
    FOREACH_ARC_WAVEFORM_DATA(local_net_arc, local_arc_waveform_data) {
      auto local_arc_delay_type = local_arc_waveform_data->get_delay_type();
      if (local_arc_delay_type != get_analysis_mode() &&
          (AnalysisMode::kMaxMin != get_analysis_mode())) {
        continue;
      }

      TransType local_trans_type = local_arc_waveform_data->get_trans_type();
      // TODO(to taosimin) fix fall trans type.
      if (local_trans_type != TransType::kRise) {
        continue;
      }

      auto& load_waveform =
          ((StaArcWaveformData*)local_arc_waveform_data)
              ->getWaveform(local_arnoldi_rc_net->getNodeID(pin_node));

      auto step_time_ns = load_waveform.get_step_time_ns();
      auto& load_pin_waveform_vec = load_waveform.get_waveform_vector();

      auto& coupled_nodes = local_rc_tree->get_coupled_nodes();

      // The coupled node should be on the net, and act as victim or aggressor.
      // As victim, should calc the two-pi model and the noise waveform.
      // As aggressor, should calc the input slew.
      for (auto& couple_node : coupled_nodes) {
        auto& local_node_name = couple_node.get_local_node();
        auto* local_rc_node = local_rc_tree->node(local_node_name);
        if (!local_rc_node) {
          LOG_INFO_EVERY_N(100) << local_node_name << " node is not exist.";
          continue;
        }
        auto& remote_node_name = couple_node.get_remote_node();

        LOG_INFO_EVERY_N(100)
            << "calc couple node noise local : " << local_node_name
            << " remote : " << remote_node_name;

        // Firstly, from remote node, we get the remote rc net according the net
        // name.
        auto [remote_net_or_instance_name, remote_node_point] =
            Str::splitTwoPart(remote_node_name.c_str(), ":");
        Net* remote_design_net;
        DesignObject* remote_pin{nullptr};

        if (std::isdigit(remote_node_point[0])) {
          remote_design_net = nl->findNet(remote_net_or_instance_name.c_str());
        } else {
          auto found_pins = nl->findPin(remote_node_name.c_str(), false, false);
          LOG_FATAL_IF(found_pins.empty())
              << "pin " << remote_node_name << " is empty";
          remote_pin = found_pins.front();
          remote_design_net = remote_pin->get_net();
        }
        LOG_FATAL_IF(!remote_design_net)
            << "remote design net " << remote_node_name << " is null.";

        auto* remote_rc_net = ista->getRcNet(remote_design_net);
        // Then, we get the input slew waveform.
        auto* remote_rc_tree = remote_rc_net->rct();
        auto* remote_driver_node = remote_rc_tree->get_root();
        auto* remote_rc_node = remote_rc_tree->node(remote_node_name);
        if (!remote_rc_node) {
          LOG_INFO << remote_node_name << " node is not exist.";
          continue;
        }

        auto* remote_driver_pin = remote_design_net->getDriver();
        auto* remote_driver_vertex = ista->findVertex(remote_driver_pin);
        LOG_FATAL_IF(!remote_driver_vertex) << "remote driver vertex not found"
                                            << remote_driver_pin->get_name();
        auto* remote_net_arc = remote_driver_vertex->get_src_arcs().front();

        auto remote_arrive_time = remote_driver_vertex->getArriveTimeNs(
            local_arc_delay_type, local_trans_type);

        auto* remote_arnoldi_net = dynamic_cast<ArnoldiNet*>(remote_rc_net);
        LOG_FATAL_IF(!remote_arnoldi_net)
            << remote_node_name << " arnolid net is not created.";

        StaData* remote_arc_waveform_data;
        FOREACH_ARC_WAVEFORM_DATA(remote_net_arc, remote_arc_waveform_data) {
          auto remote_arc_delay_type =
              local_arc_waveform_data->get_delay_type();
          TransType remote_trans_type =
              local_arc_waveform_data->get_trans_type();
          if ((local_arc_delay_type != remote_arc_delay_type) ||
              (local_trans_type != remote_trans_type)) {
            continue;
          }

          unsigned remote_rc_node_id =
              remote_arnoldi_net->getNodeID(remote_rc_node);
          auto& remote_rc_node_waveform =
              ((StaArcWaveformData*)remote_arc_waveform_data)
                  ->getWaveform(remote_rc_node_id);
          unsigned remote_driver_node_id =
              remote_arnoldi_net->getNodeID(remote_driver_node);
          auto& remote_driver_waveform =
              ((StaArcWaveformData*)remote_arc_waveform_data)
                  ->getWaveform(remote_driver_node_id);

          auto remote_rc_node_arrive_time = remote_arnoldi_net->delay(
              remote_driver_waveform, remote_rc_node_waveform, remote_pin);

          auto slew =
              remote_arnoldi_net->slew(remote_rc_node_waveform, remote_pin);
          if (!slew) {
            LOG_INFO_EVERY_N(100) << remote_node_name << " slew is not exist.";
            continue;
          }

          // Thirdly, we reduced the local net to 2-pi model. Finally, we calc
          // the output noise waveform.

          CrossTalkDelayCalc crosstalk_delay_calc(remote_design_net, the_net);
          crosstalk_delay_calc.reduceRCTreeToTwoPiModel(local_rc_net,
                                                        local_rc_node);
          if (!remote_arrive_time || !remote_rc_node_arrive_time) {
            LOG_INFO << remote_node_name << " arrive time is not exist.";
            continue;
          }
          double noise_arrive_time =
              remote_arrive_time.value() + remote_rc_node_arrive_time.value();
          crosstalk_delay_calc.calcNoiseAmplitude(noise_arrive_time, *slew);

          crosstalk_models.emplace_back(std::move(crosstalk_delay_calc));
        }
      }

      // The crosstalk waveform and ccs waveform overlap.
      std::vector<std::pair<double, VectorXd>> crosstalk_waveforms;
      auto waveform_num_point = load_pin_waveform_vec.size();
      for (auto& crosstalk_model : crosstalk_models) {
        VectorXd the_model_crosstalk_waveform(waveform_num_point);
        for (int i = 0; i < waveform_num_point; ++i) {
          double point_voltage =
              crosstalk_model.calcNoiseVoltage(step_time_ns * i);
          the_model_crosstalk_waveform(i) = point_voltage;
        }

        double arrive_time = crosstalk_model.get_input_arrive_time();
        crosstalk_waveforms.emplace_back(
            arrive_time, std::move(the_model_crosstalk_waveform));
      }

      auto* src_vertex = the_arc->get_src();
      auto local_arrive_time =
          src_vertex->getArriveTimeNs(local_arc_delay_type, local_trans_type);
      if (!local_arrive_time) {
        LOG_INFO << "The drive vertex " << src_vertex->getName()
                 << " has no time.";
        continue;
      }

      // The overlap waveform need consider the arrive time of the first point.
      for (auto& [crosstalk_start_time, crosstalk_waveform] :
           crosstalk_waveforms) {
        // LOG_INFO << "crosstalk waveform \n" << crosstalk_waveform;
        for (int i = 0; i < waveform_num_point; ++i) {
          // need to consider the arrive time
          double local_waveform_time =
              local_arrive_time.value() + i * step_time_ns;

          auto crosstalk_voltage = calcWaveformPointVoltage(
              crosstalk_waveform, step_time_ns, crosstalk_start_time,
              local_waveform_time);
          load_pin_waveform_vec(i) += crosstalk_voltage.value_or(0.0);
        }
      }

      // Update the crosstalk delay to load pin vertex.
      Waveform local_driver_waveform =
          ((StaArcWaveformData*)local_arc_waveform_data)
              ->getWaveform(0);  // driver node id is zero.
      auto& driver_waveform_vec = local_driver_waveform.get_waveform_vector();
      // LOG_INFO << "load pin waveform \n" << load_pin_waveform_vec;

      local_arnoldi_rc_net->set_is_debug(true);
      auto new_delay_with_crosstalk_ps = local_arnoldi_rc_net->calcDelay(
          driver_waveform_vec, load_pin_waveform_vec, step_time_ns, load_pin);

      if (new_delay_with_crosstalk_ps &&
          new_delay_with_crosstalk_ps.value() != 0) {
        auto origin_net_delay_fs =
            the_arc->get_arc_delay(local_arc_delay_type, local_trans_type);
        double crosstalk_delay_fs =
            PS_TO_FS(new_delay_with_crosstalk_ps.value()) - origin_net_delay_fs;
        LOG_INFO << "the net " << the_net->get_name() << " crosstalk delay "
                 << FS_TO_NS(crosstalk_delay_fs) / 1e6 << " ns";

        // Set the crosstalk delay to net.
        local_net_arc->updateCrosstalkDelay(
            local_arc_delay_type, local_trans_type, crosstalk_delay_fs);

        // TODO(to taosimin) fix fall trans type.
        local_net_arc->updateCrosstalkDelay(
            local_arc_delay_type, TransType::kFall, crosstalk_delay_fs);
      }
    }
  }

  return is_ok;
}

/**
 * @brief Crosstalk backward to timing path start node.
 *
 * @param the_vertex
 * @return unsigned
 */
unsigned StaCrossTalkPropagation::operator()(StaVertex* the_vertex) {
  std::lock_guard<std::mutex> lk(the_vertex->get_fwd_mutex());
  if (the_vertex->is_crosstalk_prop()) {
    return 1;
  }

  unsigned is_ok = 1;

  if (the_vertex->is_start()) {
    return 1;
  }

  FOREACH_SNK_ARC(the_vertex, snk_arc) {
    if (!snk_arc->isDelayArc()) {
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
      LOG_FATAL << "crosstalk propagation error";
      break;
    }
  }
  the_vertex->set_is_crosstalk_prop();

  return is_ok;
}

/**
 * @brief The crosstalk delay calc propagation.
 *
 * @param the_graph
 * @return unsigned
 */
unsigned StaCrossTalkPropagation::operator()(StaGraph* the_graph) {
  LOG_INFO << "crosstalk delay propagation start";
  unsigned is_ok = 1;

  // create thread pool
  unsigned num_threads = getNumThreads();
  ThreadPool pool(num_threads);
  StaVertex* end_vertex;

  FOREACH_END_VERTEX(the_graph, end_vertex) {
    if (end_vertex->get_snk_arcs().empty()) {
      continue;
    }
#if 1
    // enqueue and store future
    pool.enqueue([](StaFunc& func,
                    StaVertex* end_vertex) { return end_vertex->exec(func); },
                 *this, end_vertex);
#else
    is_ok = end_vertex->exec(*this);

#endif
  }

  LOG_INFO << "crosstalk delay propagation end";
  return is_ok;
}

}  // namespace ista
