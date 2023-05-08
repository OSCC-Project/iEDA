/*
 * @Author: S.J Chen
 * @Date: 2022-04-19 14:15:53
 * @LastEditTime: 2022-08-12 10:10:13
 * @LastEditors: sjchanson 13560469332@163.com
 * @Description:
 * @FilePath: /iEDA/src/iPL/src/evaluator/timing/TimingEvaluation.cc
 * Contact : https://github.com/sjchanson
 */

#include "TimingEvaluation.hh"

#include <map>

#include "util/utility.hh"

namespace ipl {

void TimingEvaluation::updateEvalTiming()
{
  _steiner_wirelength->updateAllNetWorkPointPair();

  std::vector<eval::TimingNet*> timing_net_list;
  timing_net_list.reserve(_topo_manager->get_network_list().size());
  for (auto* network : _topo_manager->get_network_list()) {
    eval::TimingNet* timing_net = new eval::TimingNet();
    timing_net->set_name(network->get_name());

    std::map<Point<int32_t>, Node*, PointCMP> point_to_node;
    std::map<Point<int32_t>, eval::TimingPin*, PointCMP> point_to_timing_pin;

    for (auto* node : network->get_node_list()) {
      const auto& node_loc = node->get_location();

      auto iter = point_to_node.find(node_loc);
      if (iter != point_to_node.end()) {
        auto* timing_pin_1 = wrapTimingTruePin(node);
        auto* timing_pin_2 = wrapTimingTruePin(iter->second);
        timing_net->add_pin_pair(timing_pin_1, timing_pin_2);
      } else {
        point_to_node.emplace(node_loc, node);
      }
    }

    const auto& point_pair_list = _steiner_wirelength->obtainPointPairList(network);
    int fake_pin_id = 0;
    for (auto point_pair : point_pair_list) {
      if (point_pair.first == point_pair.second) {
        continue;
      }

      eval::TimingPin* timing_pin_1 = nullptr;
      eval::TimingPin* timing_pin_2 = nullptr;

      auto iter_1 = point_to_node.find(point_pair.first);
      if (iter_1 != point_to_node.end()) {
        auto iter_1_1 = point_to_timing_pin.find(point_pair.first);
        if (iter_1_1 != point_to_timing_pin.end()) {
          timing_pin_1 = iter_1_1->second;
        } else {
          timing_pin_1 = wrapTimingTruePin(iter_1->second);
          point_to_timing_pin.emplace(point_pair.first, timing_pin_1);
        }
      } else {
        auto iter_1_2 = point_to_timing_pin.find(point_pair.first);
        if (iter_1_2 != point_to_timing_pin.end()) {
          timing_pin_1 = iter_1_2->second;
        } else {
          timing_pin_1 = wrapTimingFakePin(fake_pin_id++, point_pair.first);
          point_to_timing_pin.emplace(point_pair.first, timing_pin_1);
        }
      }

      auto iter_2 = point_to_node.find(point_pair.second);
      if (iter_2 != point_to_node.end()) {
        auto iter_2_1 = point_to_timing_pin.find(point_pair.second);
        if (iter_2_1 != point_to_timing_pin.end()) {
          timing_pin_2 = iter_2_1->second;
        } else {
          timing_pin_2 = wrapTimingTruePin(iter_2->second);
          point_to_timing_pin.emplace(point_pair.second, timing_pin_2);
        }
      } else {
        auto iter_2_2 = point_to_timing_pin.find(point_pair.second);
        if (iter_2_2 != point_to_timing_pin.end()) {
          timing_pin_2 = iter_2_2->second;
        } else {
          timing_pin_2 = wrapTimingFakePin(fake_pin_id++, point_pair.second);
          point_to_timing_pin.emplace(point_pair.second, timing_pin_2);
        }
      }

      timing_net->add_pin_pair(timing_pin_1, timing_pin_2);
    }
    timing_net_list.push_back(timing_net);
  }

  _timing_evaluator->updateEstimateDelay(timing_net_list);
}

eval::TimingPin* TimingEvaluation::wrapTimingTruePin(Node* node)
{
  eval::TimingPin* timing_pin = new eval::TimingPin();
  timing_pin->set_name(node->get_name());
  timing_pin->set_coord(eval::Point<int64_t>(node->get_location().get_x(), node->get_location().get_y()));
  timing_pin->set_is_real_pin(true);

  return timing_pin;
}

eval::TimingPin* TimingEvaluation::wrapTimingFakePin(int id, Point<int32_t> coordi)
{
  eval::TimingPin* timing_pin = new eval::TimingPin();
  timing_pin->set_name("fake_" + std::to_string(id));
  timing_pin->set_id(id);
  timing_pin->set_coord(eval::Point<int64_t>(coordi.get_x(), coordi.get_y()));
  timing_pin->set_is_real_pin(false);

  return timing_pin;
}

}  // namespace ipl
