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
 * @file CrossTalkDelayCalc.cc
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The crosstalk delay calc method use two pi model.
 * @version 0.1
 * @date 2022-09-16
 */
#include "CrossTalkDelayCalc.hh"
#include "netlist/Net.hh"

#include "Reduce.hh"

namespace ista {

TwoPiModel::TwoPiModel(double C1, double Rs, double C2, double Re, double CL,
                       double Cx)
    : _C1(C1), _Rs(Rs), _C2(C2), _Re(Re), _CL(CL), _Cx(Cx) {}

/**
 * @brief reduce rc tree to two pi model in order to calc the noise.
 *
 * @param rc_tree
 */
void CrossTalkDelayCalc::reduceRCTreeToTwoPiModel(RcNet* rc_net,
                                                  RctNode* coupled_point) {
  std::function<bool(RctNode*, RctNode*, RctNode*, RcTree&)>
      traverse_downstream_tree = [&traverse_downstream_tree](
                                     RctNode* coupled_point, RctNode* last_node,
                                     RctNode* downstream_node,
                                     RcTree& rc_tree) -> bool {
    if (downstream_node == coupled_point) {
      return true;
    }
    std::string downstream_node_name = downstream_node->get_name();
    FOREACH_RCNODE_FANIN_EDGE(downstream_node, fanin_edge) {
      auto& from_node = fanin_edge->get_from();
      if (&from_node == last_node) {
        continue;
      }

      std::string from_node_name = from_node.get_name();
      rc_tree.insertNode(from_node_name, from_node.get_cap());
      rc_tree.insertSegment(downstream_node_name, from_node_name,
                            fanin_edge->get_res());

      if (traverse_downstream_tree(coupled_point, downstream_node, &from_node,
                                   rc_tree)) {
        return true;
      }
    }

    return false;
  };

  std::function<bool(RctNode*, RctNode*, RctNode*, RcTree&)>
      traverse_upstream_tree =
          [&traverse_upstream_tree](RctNode* coupled_point, RctNode* last_node,
                                    RctNode* upstream_node,
                                    RcTree& rc_tree) -> bool {
    if (upstream_node == coupled_point) {
      return true;
    }

    std::string upstream_node_name = upstream_node->get_name();
    FOREACH_RCNODE_FANOUT_EDGE(upstream_node, fanout_edge) {
      auto& to_node = fanout_edge->get_to();
      if (&to_node == last_node) {
        continue;
      }

      std::string to_node_name = to_node.get_name();
      rc_tree.insertNode(to_node_name, to_node.get_cap());
      rc_tree.insertSegment(upstream_node_name, to_node_name,
                            fanout_edge->get_res());

      if (traverse_upstream_tree(coupled_point, upstream_node, &to_node,
                                 rc_tree)) {
        return true;
      }
    }

    return false;
  };

  auto* design_net = rc_net->get_net();
  auto loads = design_net->getLoads();
  if (loads.size() == 1) {
    // one pin net.
    auto* load_pin = loads.front();
    auto* rc_tree = rc_net->rct();
    auto* root_node = rc_tree->get_root();
    auto* load_node = rc_tree->node(load_pin->getFullName());
    // reduce to one pi model for coupled point downstream.
    // why need load node, need check.
    double load_nodes_pin_cap_sum = 0.0;
    std::for_each(loads.begin(), loads.end(),
                  [&load_nodes_pin_cap_sum](auto* pin) {
                    load_nodes_pin_cap_sum += pin->cap();
                  });

    RcTree downstream_tree;
    std::string root_point_name = coupled_point->get_name();
    auto* downstream_root_point =
        downstream_tree.insertNode(root_point_name, coupled_point->get_cap());
    downstream_tree.set_root(downstream_root_point);
    downstream_tree.insertNode(load_node->get_name(), load_node->get_cap());
    traverse_downstream_tree(coupled_point, nullptr, load_node,
                             downstream_tree);
    WaveformApproximation downstream_waveform;
    PiModel downstream_pi_model = downstream_waveform.reduceRCTreeToPIModel(
        downstream_tree.get_root(), load_nodes_pin_cap_sum);

    // reduce to another pi model for coupled point upstream.
    RcTree upstream_tree;
    root_point_name = root_node->get_name();
    auto* upstream_root_point =
        upstream_tree.insertNode(root_point_name, coupled_point->get_cap());
    upstream_tree.set_root(upstream_root_point);
    traverse_upstream_tree(coupled_point, nullptr, root_node, upstream_tree);
    WaveformApproximation upstream_waveform;
    PiModel upstream_pi_model = upstream_waveform.reduceRCTreeToPIModel(
        upstream_tree.get_root(), coupled_point->get_cap());

    TwoPiModel two_pi_model(
        upstream_pi_model.C_near, upstream_pi_model.R,
        upstream_pi_model.C_far + downstream_pi_model.C_near,
        downstream_pi_model.R, downstream_pi_model.C_far,
        coupled_point->get_cap());

    _two_pi_model = std::move(two_pi_model);

  } else {
    // TODO(to taosimin) multi pin net, lumped pin cap.
  }
}

/**
 * @brief calc noise waveform, please reference the paper <<Improved Crosstalk
 * Modeling for Noise Constrained Interconnect Optimization>>
 *
 * @param input_transition
 */
void CrossTalkDelayCalc::calcNoiseAmplitude(double input_arrive_time,
                                            double input_transition) {
  if (_two_pi_model) {
    _input_arrive_time = input_arrive_time;
    _tr_coeff = input_transition;

    _tx_coeff = (_two_pi_model->get_Rd() + _two_pi_model->get_Rs()) *
                _two_pi_model->get_Cx();

    _tv_coeff = (_two_pi_model->get_Rd() + _two_pi_model->get_Rs()) *
                    (_two_pi_model->get_Cx() + _two_pi_model->get_C2() +
                     _two_pi_model->get_CL()) +
                (_two_pi_model->get_Re() * _two_pi_model->get_CL() +
                 _two_pi_model->get_Rd() * _two_pi_model->get_C1());
  }
}

/**
 * @brief calc noise voltage of the time point.
 *
 * @param time
 * @return double
 */
double CrossTalkDelayCalc::calcNoiseVoltage(double time) const {
  if (IsDoubleEqual(_tv_coeff, 0.0)) {
    return 0.0;
  }

  if (!_two_pi_model) {
    return 0.0;
  }

  double voltage;
  if (time < _tr_coeff) {
    voltage = (_tx_coeff / _tr_coeff) * (1 - exp(-time / _tv_coeff));
  } else {
    voltage = (_tx_coeff / _tr_coeff) *
              (exp(-(time - _tr_coeff) / _tv_coeff) - exp(-time / _tv_coeff));
  }

  return voltage;
}

}  // namespace ista
