// ***************************************************************************************
// Copyright (c) 2023-2025 Peng Cheng Laboratory
// Copyright (c) 2023-2025 Institute of Computing Technology, Chinese Academy of Sciences
// Copyright (c) 2023-2025 Beijing Institute of Open Source Chip
//
// iEDA is licensed under Mulan PSL v2.
// You can use this software according to the terms and conditions of the Mulan PSL v2.
// You may obtain a copy of Mulan PSL v2 at:
// http://license.coscl.org.cn/MulanPSL2
//
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
// EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
// MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
//
// See the Mulan PSL v2 for more details.
// ***************************************************************************************

#include "TimingAnnotation.hh"

#include <algorithm>

#include "utility/Utility.hh"
#include "PLAPI.hh"

namespace ipl {
  //

  void TimingAnnotation::init()
  {
    // init steiner wirelength
    _steiner_wirelength = new SteinerWirelength(_topology_manager);

    // init all topology order
    std::deque<std::tuple<int32_t, Node*>> node_order_list;
    std::deque<std::tuple<int32_t, NetWork*>> network_order_list;
    std::deque<std::tuple<int32_t, Group*>> group_order_list;

    for (auto* node : _topology_manager->get_node_list()) {
      node_order_list.push_back(std::make_tuple(node->get_topo_id(), node));
    }
    for (auto* network : _topology_manager->get_network_list()) {
      network_order_list.push_back(std::make_pair(network->obtainTopoIndex(), network));
    }
    for (auto* group : _topology_manager->get_group_list()) {
      group_order_list.push_back(std::make_tuple(group->obtainTopoIndex(), group));
    }

    std::sort(node_order_list.begin(), node_order_list.end());
    std::sort(network_order_list.begin(), network_order_list.end());
    std::sort(group_order_list.begin(), group_order_list.end());

    for (size_t i = 0; i < node_order_list.size(); i++) {
      _topo_order_node_list.push_back(std::get<1>(node_order_list[i]));
    }
    for (size_t i = 0; i < network_order_list.size(); i++) {
      _topo_order_net_list.push_back(std::get<1>(network_order_list[i]));
    }
    for (size_t i = 0; i < group_order_list.size(); i++) {
      _topo_order_group_list.push_back(std::get<1>(group_order_list[i]));
    }

    // tmp for specify the clock name
    _clock_name = iPLAPIInst.obtainClockNameList().at(0);

    // update sta timing
    this->updateSTATimingFull();
  }

  float TimingAnnotation::get_early_wns() {
    return iPLAPIInst.obtainEarlyWNS(_clock_name.c_str()) * _unit;
  }

  float TimingAnnotation::get_late_wns() {
    return iPLAPIInst.obtainLateWNS(_clock_name.c_str()) * _unit;
  }

  float TimingAnnotation::get_early_tns() {
    return iPLAPIInst.obtainEarlyTNS(_clock_name.c_str()) * _unit;
  }

  float TimingAnnotation::get_late_tns() {
    return iPLAPIInst.obtainLateTNS(_clock_name.c_str()) * _unit;
  }

  float TimingAnnotation::get_node_early_slack(int32_t node_id) {
    auto* node = _topology_manager->findNodeById(node_id);
    return iPLAPIInst.obtainPinEarlySlack(node->get_name()) * _unit;
  }

  float TimingAnnotation::get_node_late_slack(int32_t node_id) {
    auto* node = _topology_manager->findNodeById(node_id);
    return iPLAPIInst.obtainPinLateSlack(node->get_name()) * _unit;
  }

  float TimingAnnotation::get_node_early_arrival_time(int32_t node_id) {
    auto* node = _topology_manager->findNodeById(node_id);
    return iPLAPIInst.obtainPinEarlyArrivalTime(node->get_name()) * _unit;
  }

  float TimingAnnotation::get_node_late_arrival_time(int32_t node_id) {
    auto* node = _topology_manager->findNodeById(node_id);
    return iPLAPIInst.obtainPinLateArrivalTime(node->get_name()) * _unit;
  }

  float TimingAnnotation::get_node_early_required_time(int32_t node_id) {
    auto* node = _topology_manager->findNodeById(node_id);
    return iPLAPIInst.obtainPinEarlyRequiredTime(node->get_name()) * _unit;
  }

  float TimingAnnotation::get_node_late_required_time(int32_t node_id) {
    auto* node = _topology_manager->findNodeById(node_id);
    return iPLAPIInst.obtainPinLateRequiredTime(node->get_name()) * _unit;
  }

  Node* TimingAnnotation::get_clock_node(Group* group)
  {
    Node* clock_node = nullptr;
    for (auto* node : group->obtainInputNodes()) {
      auto* network = node->get_network();
      if (network) {
        if (network->get_network_type() == NETWORK_TYPE::kClock) {
          clock_node = node;
          break;
        }
      }
    }

    return clock_node;
  }

  Node* TimingAnnotation::get_data_node(Group* group)
  {
    Node* data_node = nullptr;
    for (auto* node : group->obtainInputNodes()) {
      auto* network = node->get_network();
      if (network) {
        if (network->get_network_type() != NETWORK_TYPE::kClock) {
          data_node = node;
          break;
        }
      }
    }

    return data_node;
  }

  void TimingAnnotation::updateSTATimingFull() {
    _steiner_wirelength->updateAllNetWorkPointPair();
    std::map<int32_t, std::vector<std::pair<Point<int32_t>, Point<int32_t>>>> net_id_to_points_map;

    for (auto* network : _topology_manager->get_network_list()) {
      net_id_to_points_map.emplace(network->get_network_id(), _steiner_wirelength->obtainPointPairList(network));
    }

    iPLAPIInst.updatePartOfTiming(_topology_manager, net_id_to_points_map);
  }

  void TimingAnnotation::updateSTATimingIncremental(NetWork* network)
  {
    _steiner_wirelength->updateNetWorkPointPair(network);
    std::map<int32_t, std::vector<std::pair<Point<int32_t>, Point<int32_t>>>> net_id_to_points_map;

    net_id_to_points_map.emplace(network->get_network_id(), _steiner_wirelength->obtainPointPairList(network));

    iPLAPIInst.updatePartOfTiming(_topology_manager, net_id_to_points_map);
  }

  void TimingAnnotation::updateSTATimingIncremental(std::vector<NetWork*>& network_list) {
    _steiner_wirelength->updatePartOfNetWorkPointPair(network_list);
    std::map<int32_t, std::vector<std::pair<Point<int32_t>, Point<int32_t>>>> net_id_to_points_map;

    for (auto* network : network_list) {
      net_id_to_points_map.emplace(network->get_network_id(), _steiner_wirelength->obtainPointPairList(network));
    }

    iPLAPIInst.updatePartOfTiming(_topology_manager, net_id_to_points_map);
  }

  float TimingAnnotation::get_node_criticality(Node* node)
  {
    float node_slack = get_node_late_slack(node->get_node_id());
    node_slack > 0 ? node_slack = 0 : node_slack;
    float wns = get_late_wns();

    if (wns > 0) {
      return 0.0f;
    }
    else {
      return std::max(0.0f, std::min((float)(1.0), node_slack / wns));
    }
  }

  float TimingAnnotation::get_group_criticality(Group* group)
  {
    float criticality = 0.0f;
    for (auto* node : group->obtainOutputNodes()) {
      criticality = std::max(criticality, get_node_criticality(node));
    }

    if (group->get_group_type() == GROUP_TYPE::kFlipflop) {
      // obtain the dpin of flipflop
      auto* d_node = get_data_node(group);
      criticality = std::max(criticality, get_node_criticality(d_node));
    }

    return criticality;
  }

  float TimingAnnotation::get_network_criticality(NetWork* network) {
    float criticality = 0.0f;
    auto* driver_node = network->get_transmitter();
    if (driver_node) {
      criticality = this->get_node_criticality(driver_node);
    }

    return criticality;
  }

  float TimingAnnotation::get_network_centrality(NetWork* network) {
    float centrality = 0.0f;
    auto* driver_node = network->get_transmitter();
    if (driver_node) {
      centrality = driver_node->get_centrality();
    }

    return centrality;
  }

  float TimingAnnotation::get_node_importance(Node* node) {
    float node_centrality = node->get_centrality();
    float node_criticality = node->get_criticality();
    return (2 * node_centrality + node_criticality / 3);
  }

  void TimingAnnotation::reportCurrentTiming() {
    LOG_INFO << "eWNS: " << this->get_early_wns() << " ; " << "eTNS: " << this->get_early_tns() << " ; "
      << "lWNS: " << this->get_late_wns() << " ; " << "lTNS: " << this->get_late_tns();
  }

  void TimingAnnotation::printALLTimingInfoForDebug() {
    for (auto* node : _topo_order_node_list) {
      LOG_INFO << "Node: " << node->get_name() << " --- " << "RT: " << this->get_node_late_required_time(node->get_node_id())
        << " --- " << "AT: " << this->get_node_late_arrival_time(node->get_node_id())
        << " --- " << "Slack: " << this->get_node_late_slack(node->get_node_id());
    }
    reportCurrentTiming();
  }

  void TimingAnnotation::printTargetGroupTimingInfoForDebug(Group* group){
    for(auto* node : group->get_node_list()){
      LOG_INFO << "Node: " << node->get_name() << " --- " << "RT: " << this->get_node_late_required_time(node->get_node_id())
        << " --- " << "AT: " << this->get_node_late_arrival_time(node->get_node_id())
        << " --- " << "Slack: " << this->get_node_late_slack(node->get_node_id());      
    }
    reportCurrentTiming();
  }

  void TimingAnnotation::printTargetGroupSideTimingEffect(Group* group){
    for(auto* node : group->get_node_list()){
      LOG_INFO << "Target Node: " << node->get_name() << " --- " << "RT: " << this->get_node_late_required_time(node->get_node_id())
        << " --- " << "AT: " << this->get_node_late_arrival_time(node->get_node_id())
        << " --- " << "Slack: " << this->get_node_late_slack(node->get_node_id());        
      auto* network = node->get_network();
      if(network){
        for(auto* network_sink_node : network->get_receiver_list()){
          if(network_sink_node->get_group() == group){
            continue;
          }
          LOG_INFO << "Side Node: " << network_sink_node->get_name() << " --- " << "RT: " << this->get_node_late_required_time(network_sink_node->get_node_id())
            << " --- " << "AT: " << this->get_node_late_arrival_time(network_sink_node->get_node_id())
            << " --- " << "Slack: " << this->get_node_late_slack(network_sink_node->get_node_id());
        }
      }
    }
    reportCurrentTiming();
  }

  void TimingAnnotation::updateCriticalityAndCentralityFull()
  {
    // reset max centrality.
    _max_centrality = 0.0f;

    // update all network in reverse order
    int32_t network_size = _topo_order_net_list.size();
    for (int32_t i = network_size - 1; i >= 0; i--) {
      updateCriticalityAndCentrality(_topo_order_net_list[i]);
    }
  }

  void TimingAnnotation::updateCriticalityAndCentralityIncremental(const std::vector<NetWork*>& network_list) {
    std::deque<std::tuple<int32_t, NetWork*>> ordered_network_list;
    for (auto* network : network_list) {
      ordered_network_list.push_back(std::make_tuple(network->obtainTopoIndex(), network));
    }

    std::sort(ordered_network_list.begin(), ordered_network_list.end());

    int32_t network_size = ordered_network_list.size();
    for (int32_t i = network_size - 1; i >= 0; i--) {
      updateCriticalityAndCentrality(std::get<1>(ordered_network_list[i]));
    }
  }

  void TimingAnnotation::updateCriticalityAndCentrality(NetWork* network)
  {
    bool dont_propagate_clock_net = true;

    float sum_sink_centrality = 0.0f;
    for (auto* sink : network->get_receiver_list()) {
      sink->set_criticality(get_node_criticality(sink));
      sink->set_centrality(0.0);

      // update the centrality of this sink.
      bool has_arcs = false;
      for (auto* arc : sink->get_output_arc_list()) {
        auto* to_node = arc->get_to_node();

        has_arcs = true;
        double centrality = sink->get_centrality() + arc->get_flow_value() * to_node->get_centrality();
        sink->set_centrality(centrality);
      }
      // if no arcs were found, keep set its centrality to its criticality.
      if (!has_arcs) {
        double centrality = get_node_criticality(sink);
        sink->set_centrality(centrality);
      }

      bool is_clock_pin = (sink->get_network()->get_network_type() == NETWORK_TYPE::kClock);
      if (!dont_propagate_clock_net || !is_clock_pin) {
        sum_sink_centrality += sink->get_centrality();
      }
    }

    // update maximum centrality
    _max_centrality = std::max(_max_centrality, sum_sink_centrality);

    // update driver centrality and driving arc flows.
    // assumption is that driver net is single.
    auto* node_driver = network->get_transmitter();
    if (node_driver) {
      node_driver->set_centrality(sum_sink_centrality);
    }

    // update driving arcs flow
    float sum = 0.0f;
    int counter_arcs = 0;
    if (node_driver) {
      for (auto* arc : node_driver->get_input_arc_list()) {
        auto* from_node = arc->get_from_node();
        sum += get_node_criticality(from_node);
        counter_arcs++;
      }
      if (Utility().isFloatApproximatelyZero(sum)) {
        sum = 1 / float(counter_arcs);
      }

      for (auto* arc : node_driver->get_input_arc_list()) {
        auto* from_node = arc->get_from_node();
        float flow_value = get_node_criticality(from_node) / sum;
        arc->set_flow_value(flow_value);
      }
    }

  }

  std::string TimingAnnotation::extractLastName(std::string input) {
    char split_flag = ':';
    auto target_char_pos = std::find(input.begin(), input.end(), split_flag);
    int target_index = std::distance(input.begin(), target_char_pos+1);
    std::string sub_str = input.substr(target_index);
    return sub_str;
  }

  float TimingAnnotation::getOutNodeRes(Node* out_node) {
    std::string group_name = "";
    if (out_node->get_group()) {
      group_name = out_node->get_group()->get_name();
    }
    std::string node_last_name = extractLastName(out_node->get_name());
    
    return iPLAPIInst.obtainInstOutPinRes(group_name, node_last_name);
  }

  float TimingAnnotation::getAvgWireResPerUnitLength() {
    return iPLAPIInst.obtainAvgWireResUnitLengthUm();
  }

  float TimingAnnotation::getAvgWireCapPerUnitLength() {
    return iPLAPIInst.obtainAvgWireCapUnitLengthUm();
  }

  float TimingAnnotation::getNodeInputCap(Node* input_node) {
    return iPLAPIInst.obtainPinCap(input_node->get_name()) * _unit;
  }

}  // namespace ipl