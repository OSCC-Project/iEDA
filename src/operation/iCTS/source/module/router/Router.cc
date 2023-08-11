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
#include "Router.h"

#include "CTSAPI.hpp"
#include "GOCA.hh"
#include "HCTS.h"
namespace icts {

void Router::update()
{
  auto* design = CTSAPIInst.get_design();
  for (auto& clk_topo : _clk_topos) {
    design->addClockTopo(clk_topo);
  }
}

void Router::build()
{
  auto* config = CTSAPIInst.get_config();
  auto router_type = config->get_router_type();
  if (router_type == "ZST" || router_type == "BST" || router_type == "UST") {
    DMEBuild();
    return;
  }
  if (router_type == "SlewAware") {
    slewAwareBuild();
    return;
  }
  if (router_type == "HCTS") {
    hctsBuild();
    return;
  }
  if (router_type == "GOCA") {
    gocaBuild();
    return;
  }
}

void Router::DMEBuild()
{
  for (auto* clock : _clocks) {
    auto& clock_nets = clock->get_clock_nets();
    for (auto* clock_net : clock_nets) {
      if (clock_net->get_pins().size() <= 3) {
        continue;
      }
      clock_net->setClockRouted();
      // routing(clock_net); // no constraint build
      comfortRouting(clock_net);
    }
  }
}

void Router::slewAwareBuild()
{
  auto* config = CTSAPIInst.get_config();
  auto* timer = new TimingCalculator();

  for (auto* clock : _clocks) {
    auto& clock_nets = clock->get_clock_nets();
    for (auto* clk_net : clock_nets) {
      CTSAPIInst.resetId();
      auto insts = get_clustering_insts(clk_net);
      if (insts.size() <= 1) {
        continue;
      }
      clk_net->setClockRouted();
      slewAwareRouting(clk_net);
      // find root node
      auto root_topo = _clk_topos.back();
      auto* root_node = CTSAPIInst.findTimingNode(root_topo.get_driver()->get_name());

      auto* clk_inst = clk_net->get_driver_inst();
      auto* clk_node = new TimingNode(clk_inst);
      CTSAPIInst.addTimingNode(clk_node);
      auto slew_constraint = config->get_max_buf_tran();
      clk_node->set_slew_constraint(slew_constraint);
      clk_node->set_left(root_node);
      root_node->set_parent(clk_node);
      root_node->set_left(nullptr);
      root_node->set_right(nullptr);
      if (timer->calcShortestLength(clk_node, root_node) > config->get_max_length()) {
        timer->breakLongWire(clk_node, root_node);
        timer->updateCap(clk_node);
        timer->timingPropagate(clk_node);
        auto net_name = clk_node->get_name() + "_break";
        auto* slew_aware = new SlewAware(net_name, {});
        // top down
        slew_aware->topDown(clk_node);
        // make topo
        slew_aware->buildClockTopo(clk_node->get_left(), net_name);
#ifdef TIMING_LOG
        slew_aware->timingLog();
#endif
        auto break_topos = slew_aware->get_clk_topos();
        for (auto topo : break_topos) {
          _clk_topos.emplace_back(topo);
        }
        root_node = clk_node->get_left();
        root_node->set_left(nullptr);
        root_node->set_right(nullptr);
      }
      timer->updateTiming(clk_node);
      ClockTopo root_clk_topo = create_clock_topo(clk_net);
      root_clk_topo.connect_load(root_node->get_inst());
      _clk_topos.emplace_back(root_clk_topo);
    }
  }
}

void Router::hctsBuild()
{
  for (auto* clock : _clocks) {
    auto& clock_nets = clock->get_clock_nets();
    for (auto* clk_net : clock_nets) {
      auto insts = get_clustering_insts(clk_net);
      if (insts.size() <= 1) {
        continue;
      }
      clk_net->setClockRouted();
      hctsRouting(clk_net);
      // find root node
      auto root_topo = _clk_topos.back();
      ClockTopo root_clk_topo = create_clock_topo(clk_net);
      root_clk_topo.connect_load(root_topo.get_driver());
      _clk_topos.emplace_back(root_clk_topo);
    }
  }
}

void Router::gocaBuild()
{
  for (auto* clock : _clocks) {
    auto& clock_nets = clock->get_clock_nets();
    for (auto* clk_net : clock_nets) {
      auto insts = get_clustering_insts(clk_net);
      if (insts.size() <= 1) {
        continue;
      }
      clk_net->setClockRouted();
      // debug
      if (insts.size() < 32) {
        continue;
      }
      gocaRouting(clk_net);
      // find root node
      auto root_topo = _clk_topos.back();
      ClockTopo root_clk_topo = create_clock_topo(clk_net);
      root_clk_topo.connect_load(root_topo.get_driver());
      _clk_topos.emplace_back(root_clk_topo);
    }
  }
}

void Router::routing(CtsNet* clk_net)
{
  std::string net_name = clk_net->get_net_name();
  std::vector<CtsInstance*> insts = get_clustering_insts(clk_net);
  if (insts.empty()) {
    return;
  }
  _steiner_index += insts.size() * 100;
  LOG_INFO << "Start router for net: " << net_name;
  // hierarchical clustering
  int insert_num = 0;
  for (int level = 0; insts.size() > 1; ++level) {
    std::vector<ClockTopo> clk_topos;
    std::vector<std::vector<CtsInstance*>> clusters;
    clustering(clusters, insts);
    for (size_t i = 0; i < clusters.size(); ++i) {
      Topology<Endpoint> topo;
      topoligize(topo, clusters[i]);
      std::string clk_topo_name = connect_string(net_name, level, i);
      init_node_name(topo, clk_topo_name);
      dme(topo);
      ClockTopo clk_topo = create_clock_topo(topo, clk_topo_name);
      auto driver = clk_topo.get_driver();
      CTSAPIInst.placeInstance(driver);
      driver->set_level(level + 1);
      clk_topos.emplace_back(clk_topo);
    }
    insert_num += clk_topos.size();
    insts.clear();
    std::for_each(clk_topos.begin(), clk_topos.end(), [&insts](const ClockTopo& clk_topo) { insts.push_back(clk_topo.get_driver()); });
    std::copy(clk_topos.begin(), clk_topos.end(), std::back_inserter(_clk_topos));
  }
  LOG_INFO << net_name << " insert buffer num: " << insert_num;
  ClockTopo root_clk_topo = create_clock_topo(clk_net);
  root_clk_topo.connect_load(insts.front());
  root_clk_topo.get_driver()->set_level(insts.front()->get_level() + 1);
  _clk_topos.emplace_back(root_clk_topo);
}

void Router::comfortRouting(CtsNet* clk_net)
{
  auto* config = CTSAPIInst.get_config();
  std::string net_name = clk_net->get_net_name();
  std::vector<CtsInstance*> insts = get_clustering_insts(clk_net);
  if (insts.empty()) {
    return;
  }
  LOG_INFO << "Start router for net: " << net_name;
  _steiner_index += insts.size() * 100;
  if (insts.size() == 1) {
    auto driver = insts[0];
    ClockTopo root_clk_topo = create_clock_topo(clk_net);
    root_clk_topo.connect_load(driver);
    root_clk_topo.get_driver()->set_level(driver->get_level() + 1);
    _clk_topos.emplace_back(root_clk_topo);
    return;
  }
  // total topology
  auto total_topo = biClusterTopo(insts);
  dme(total_topo);
  // cut sub topo
  std::vector<Topology<Endpoint>> sub_topos;
  if (static_cast<int>(insts.size()) <= config->get_cluster_size()) {
    sub_topos = {total_topo};
  } else {
    sub_topos = splitTopo(total_topo, net_name);
  }
  for (size_t i = 0; i < sub_topos.size(); ++i) {
    auto& topo = sub_topos[i];
    std::string clk_topo_name = net_name + "_" + std::to_string(i);
    init_node_name(topo, clk_topo_name);
    ClockTopo clk_topo = create_clock_topo(topo, clk_topo_name);
    auto driver = clk_topo.get_driver();
    for (auto load : clk_topo.get_loads()) {
      auto level = std::max(driver->get_level(), load->get_level() + 1);
      driver->set_level(level);
    }
    _clk_topos.emplace_back(clk_topo);
  }
  LOG_INFO << net_name << " insert buffer num: " << sub_topos.size();
  // connet source
  auto root_topo = _clk_topos.back();
  ClockTopo root_clk_topo = create_clock_topo(clk_net);
  root_clk_topo.connect_load(root_topo.get_driver());
  root_clk_topo.get_driver()->set_level(root_topo.get_driver()->get_level() + 1);
  _clk_topos.emplace_back(root_clk_topo);
}

void Router::slewAwareRouting(CtsNet* clk_net)
{
  std::string net_name = clk_net->get_net_name();
  std::vector<CtsInstance*> insts = get_clustering_insts(clk_net);
  if (insts.size() <= 1) {
    return;
  }
  // total topology
  auto* slew_aware = new SlewAware(net_name, insts);
  insts.size() > 3000 ? slew_aware->slewAwareByKmeans() : slew_aware->slewAware();
  // slew_aware->slewAwareByBiCluster();
#ifdef TIMING_LOG
  slew_aware->timingLog();
#endif
#ifdef SAVE_TRAINING_DATA
  slew_aware->saveTrainingData();
#endif
  auto clk_topos = slew_aware->get_clk_topos();
  for (auto& clk_topo : clk_topos) {
    _clk_topos.emplace_back(clk_topo);
  }
}

void Router::hctsRouting(CtsNet* clk_net)
{
  std::string net_name = clk_net->get_net_name();
  std::vector<CtsInstance*> insts = get_clustering_insts(clk_net);
  if (insts.size() <= 1) {
    return;
  }
  // total topology
  auto* hcts = new HCTS(net_name, insts);
  auto clk_topos = hcts->get_clk_topos();
  for (auto& clk_topo : clk_topos) {
    _clk_topos.emplace_back(clk_topo);
  }
}

void Router::gocaRouting(CtsNet* clk_net)
{
  std::string net_name = clk_net->get_net_name();
  std::vector<CtsInstance*> insts = get_clustering_insts(clk_net);
  if (insts.size() <= 1) {
    return;
  }
  // total topology
  auto goca = GOCA(net_name, insts);
  goca.run();
  auto clk_topos = goca.get_clk_topos();
  for (auto& clk_topo : clk_topos) {
    _clk_topos.emplace_back(clk_topo);
  }
}

std::vector<CtsInstance*> Router::get_clustering_insts(CtsNet* clk_net)
{
  std::vector<CtsInstance*> insts;
  for (auto* load_pin : clk_net->get_load_pins()) {
    auto* load_inst = load_pin->get_instance();
    auto inst_type = load_inst->get_type();
    if (inst_type != CtsInstanceType::kBuffer) {
      insts.push_back(load_inst);
    }
  }
  return insts;
}

std::string Router::connect_string(const std::string& net_name, int level, int index) const
{
  return net_name + "_" + std::to_string(level) + "_" + std::to_string(index);
}

ClockTopo Router::create_clock_topo(CtsNet* clk_net)
{
  auto* driver = clk_net->get_driver_inst();
  auto loads = clk_net->get_load_insts();

  ClockTopo clk_topo(clk_net->get_net_name());
  clk_topo.add_driver(driver);
  for (auto* load : loads) {
    if (load->get_type() == CtsInstanceType::kBuffer) {
      clk_topo.add_load(load);
    }
    auto dist = pgl::manhattan_distance(driver->get_location(), load->get_location()) + load->getSubWirelength();
    if (dist > driver->getSubWirelength()) {
      driver->setSubWirelength(dist);
    }
  }

  Endpoint epl{driver->get_name(), driver->get_location()};
  int steiner_id = 0;
  for (auto* load : loads) {
    if (load->get_type() == CtsInstanceType::kBuffer) {
      Endpoint epr{load->get_name(), load->get_location()};
      if (pgl::rectilinear(driver->get_location(), load->get_location())) {
        clk_topo.add_signal_wire(CtsSignalWire(epl, epr));
      } else {
        auto trunk_loc = Point(driver->get_location().x(), load->get_location().y());
        Endpoint ep_trunk{"steiner_" + std::to_string(steiner_id++), trunk_loc};
        clk_topo.add_signal_wire(CtsSignalWire(epl, ep_trunk));
        clk_topo.add_signal_wire(CtsSignalWire(ep_trunk, epr));
      }
    }
  }
  return clk_topo;
}

template <typename T>
void Router::init_node_name(Topology<T>& topo, const std::string& clk_topo_name)
{
  // set the id of topo internal nodes
  int internal_node_name = 0;
  auto vertex_itr = topo.preorder_vertexs();
  for (auto itr = vertex_itr.first; itr != vertex_itr.second; ++itr) {
    if (!itr.is_leaf()) {
      DataTraits<T>::setId(*itr, "steiner_" + std::to_string(internal_node_name++));
    }
  }

  // set the id of topo root node
  std::string root_name = clk_topo_name + "_buf";
  auto& root_val = topo.value(topo.root());
  DataTraits<T>::setId(root_val, root_name);
}

template <typename T>
ClockTopo Router::create_clock_topo(Topology<T>& topo, const std::string& clk_topo_name)
{
  auto* config = CTSAPIInst.get_config();
  std::string root_name = clk_topo_name + "_buf";
  auto& root_val = topo.value(topo.root());
  // create clock topology, add signal wires and load instance to clock
  // topology
  ClockTopo clk_topo(clk_topo_name);
  auto edge_itr = topo.edges();
  for (auto itr = edge_itr.first; itr != edge_itr.second; ++itr) {
    auto edge = *itr;
    auto name1 = DataTraits<T>::getId(edge.first);
    auto loc1 = DataTraits<T>::getPoint(edge.first);
    auto name2 = DataTraits<T>::getId(edge.second);
    auto loc2 = DataTraits<T>::getPoint(edge.second);
    if (DataTraits<T>::needDetour(edge.second)) {
      auto detour_points = DataTraits<T>::getDetourPoints(edge.second);
      std::vector<std::string> name_vec
          = {name1, "steiner_" + std::to_string(_steiner_index++), "steiner_" + std::to_string(_steiner_index++), name2};
      for (size_t i = 0; i < detour_points.size() - 1; ++i) {
        auto epl = Endpoint(name_vec[i], detour_points[i]);
        auto epr = Endpoint(name_vec[i + 1], detour_points[i + 1]);
        CtsSignalWire wire(epl, epr);
        clk_topo.add_signal_wire(wire);
      }
    } else {
      CtsSignalWire wire(Endpoint(name1, loc1), Endpoint(name2, loc2));
      clk_topo.add_signal_wire(wire);
    }
    if (_name_to_inst.count(name2)) {
      clk_topo.add_load(_name_to_inst.at(name2));
    }
  }
  // create driver instance, add it to clock topology
  auto* driver
      = new CtsInstance(root_name, config->get_buffer_types().front(), CtsInstanceType::kBuffer, DataTraits<T>::getPoint(root_val));
  driver->setSubWirelength(DataTraits<T>::getSubWirelength(root_val));
  clk_topo.add_driver(driver);
  _name_to_inst.insert(std::make_pair(root_name, driver));
  return clk_topo;
}

template <typename T>
void Router::dme(Topology<T>& topo) const
{
  auto* config = CTSAPIInst.get_config();
  std::string delay_type = config->get_delay_type();
  DelayModel delay_model;
  if (delay_type == "elmore") {
    delay_model = DelayModel::kElmore;
  } else {
    delay_model = DelayModel::kLinear;
  }

  std::string router_type = config->get_router_type();
  if (router_type == "BST") {
    double skew_bound = config->get_skew_bound();
    BstParams params(BstType::kIME, CTSAPIInst.getDbUnit(), skew_bound, DelayModel::kLinear);
    icts::dme(topo, params);
  } else if (router_type == "ZST") {
    ZstParams params(delay_model, CTSAPIInst.getDbUnit(), CTSAPIInst.getClockUnitRes(), CTSAPIInst.getClockUnitCap());
    icts::dme(topo, params);
  } else {
    UstParams params(delay_model, CTSAPIInst.getDbUnit(), config->get_skew_bound(), CTSAPIInst.getClockUnitRes(),
                     CTSAPIInst.getClockUnitCap());
    icts::dme(topo, params, _skew_scheduler);
  }
}

void Router::clustering(std::vector<std::vector<CtsInstance*>>& clusters, const std::vector<CtsInstance*>& insts) const
{
  auto* config = CTSAPIInst.get_config();
  Kmeans<CtsInstance*> kmeans;
  auto temp_clusters = kmeans(insts, config->get_cluster_size());
  for (auto itr = temp_clusters.begin(); itr != temp_clusters.end(); ++itr) {
    // count average dist
    auto cluster = *itr;
    auto sum_x = 0;
    auto sum_y = 0;
    for (auto inst : cluster) {
      sum_x += inst->get_location().x();
      sum_y += inst->get_location().y();
    }
    auto avg_x = sum_x / cluster.size();
    auto avg_y = sum_y / cluster.size();
    auto center_point = Point(avg_x, avg_y);
    auto total_dist = 0;
    for (auto inst : cluster) {
      total_dist += pgl::manhattan_distance(center_point, inst->get_location());
    }
    auto avg_dist = total_dist / cluster.size();
    auto avg_ratio = avg_dist / CTSAPIInst.getDbUnit() / 15;
    if (avg_ratio < 1) {
      clusters.emplace_back(cluster);
    } else {
      auto sub_size = config->get_cluster_size() / std::pow(2, std::ceil(std::log2(avg_ratio)));
      sub_size = sub_size > 2 ? sub_size : 2;
      auto sub_clusters = kmeans(cluster, sub_size);
      for (auto sub_cluster : sub_clusters) {
        clusters.emplace_back(sub_cluster);
      }
    }
  }
}
int Router::calFeasibleFanout(const double& avg_wirelength) const
{
  auto* config = CTSAPIInst.get_config();
  auto level_length = CTSAPIInst.getDbUnit() * 15;  // um
  auto avg_ratio = avg_wirelength / level_length;
  if (avg_ratio < 1) {
    return config->get_cluster_size();
  } else {
    int sub_size = std::floor(config->get_cluster_size() / std::pow(2, std::ceil(std::log2(avg_ratio))));
    sub_size = sub_size > 2 ? sub_size : 2;
    return sub_size;
  }
}

template <typename T>
double Router::calAvgWirelength(const int& root_id, Topology<T>& topo) const
{
  std::deque<int> id_que = {root_id};
  int leaf_num = 0;
  double total_wirelength = 0;
  while (!id_que.empty()) {
    auto id = id_que.front();
    id_que.pop_front();
    auto& node = topo.node(id);
    auto& value = topo.value(id);
    if (node.left() != -1 && node.right() != -1) {
      auto left_value = topo.value(node.left());
      auto left_wirelength = pgl::manhattan_distance(DataTraits<T>::getPoint(left_value), DataTraits<T>::getPoint(value));
      auto right_value = topo.value(node.right());
      auto right_wirelength = pgl::manhattan_distance(DataTraits<T>::getPoint(right_value), DataTraits<T>::getPoint(value));
      total_wirelength += left_wirelength + right_wirelength;
      id_que.emplace_back(node.left());
      id_que.emplace_back(node.right());
    } else {
      ++leaf_num;
    }
  }
  return total_wirelength / leaf_num;
}
template <typename T>
Topology<T> Router::cutTopo(const int& root_id, Topology<T>& topo) const
{
  auto& parent_value = topo.value(root_id);
  std::vector<TopoNode<T>> new_nodes = {TopoNode<T>{parent_value, -1, -1, -1, 0}};
  std::map<int, int> id_map = {{root_id, 0}};
  std::deque<int> id_que = {root_id};

  while (!id_que.empty()) {
    auto itr_id = id_que.front();
    id_que.pop_front();
    auto& node = topo.node(itr_id);
    if (node.left() != -1 && node.right() != -1) {
      auto parent_id = id_map.at(itr_id);
      auto& parent_node = new_nodes[parent_id];

      auto& left_value = topo.value(node.left());
      auto& right_value = topo.value(node.right());
      auto left_node = TopoNode<T>{left_value, parent_id, -1, -1, 0};
      auto right_node = TopoNode<T>{right_value, parent_id, -1, -1, 0};

      auto left_id = new_nodes.size();
      auto right_id = new_nodes.size() + 1;

      id_map.insert(std::make_pair(node.left(), left_id));
      id_map.insert(std::make_pair(node.right(), right_id));

      parent_node.left(left_id);
      parent_node.right(right_id);

      new_nodes.emplace_back(left_node);
      new_nodes.emplace_back(right_node);

      id_que.emplace_back(node.left());
      id_que.emplace_back(node.right());
    }
  }
  return Topology<T>(new_nodes, 0);
}

template <typename T>
std::vector<Topology<T>> Router::splitTopo(Topology<T>& topo, const std::string& net_name) const
{
  auto* config = CTSAPIInst.get_config();
  if (static_cast<int>(topo.nodes().size()) <= config->get_cluster_size()) {
    return std::vector<Topology<T>>{topo};
  }
  std::vector<Topology<T>> all_topos;
  auto vertex_itr = topo.postorder_vertexs();
  for (auto itr = vertex_itr.first; itr != vertex_itr.second; ++itr) {
    if (itr.is_leaf()) {
      DataTraits<T>::setFanout(*itr, 1);
    } else {
      auto avg_wirelength = calAvgWirelength(itr.get_vertex(), topo);
      auto feasible_fanout = calFeasibleFanout(avg_wirelength);
      auto left_fanout = DataTraits<T>::getFanout(*itr.left());
      auto right_fanout = DataTraits<T>::getFanout(*itr.right());
      auto fanout = left_fanout + right_fanout;
      bool have_top_cut = false;
      if (fanout == feasible_fanout) {
        auto buffer_name = net_name + "_" + std::to_string(all_topos.size()) + "_buf";
        DataTraits<T>::setId(*itr, buffer_name);
        auto make_topo = cutTopo(itr.get_vertex(), topo);
        itr.cut();
        all_topos.emplace_back(make_topo);
        fanout = 1;
        have_top_cut = true;
      } else if (fanout > feasible_fanout && left_fanout > 1 && right_fanout > 1) {
        auto left_itr = itr.left();
        auto left_buffer_name = net_name + "_" + std::to_string(all_topos.size()) + "_buf";
        DataTraits<T>::setId(*left_itr, left_buffer_name);
        auto make_left_topo = cutTopo(left_itr.get_vertex(), topo);
        left_itr.cut();
        all_topos.emplace_back(make_left_topo);
        DataTraits<T>::setFanout(*left_itr, 1);

        auto right_itr = itr.right();
        auto right_buffer_name = net_name + "_" + std::to_string(all_topos.size()) + "_buf";
        DataTraits<T>::setId(*right_itr, right_buffer_name);
        auto make_right_topo = cutTopo(right_itr.get_vertex(), topo);
        right_itr.cut();
        all_topos.emplace_back(make_right_topo);
        DataTraits<T>::setFanout(*right_itr, 1);

        fanout = 2;
      }
      if (itr.is_root() && !have_top_cut) {
        auto buffer_name = net_name + "_" + std::to_string(all_topos.size()) + "_buf";
        DataTraits<T>::setId(*itr, buffer_name);
        auto make_topo = cutTopo(itr.get_vertex(), topo);
        itr.cut();
        all_topos.emplace_back(make_topo);
        fanout = 1;
      }
      DataTraits<T>::setFanout(*itr, fanout);
    }
  }
  if (all_topos.empty()) {
    all_topos.emplace_back(topo);
  }
  return all_topos;
}

template <typename T>
Topology<T> Router::biClusterTopo(const std::vector<CtsInstance*>& insts) const
{
  std::vector<TopoNode<T>> all_nodes;
  if (insts.size() == 1) {
    auto inst = insts[0];
    T data;
    DataTraits<T>::setPoint(data, inst->get_location());
    DataTraits<T>::setId(data, inst->get_name());
    DataTraits<T>::setSubWirelength(data, inst->getSubWirelength());
    all_nodes.emplace_back(TopoNode<T>{data, -1, -1, -1, inst->getSubWirelength()});
    return Topology<T>(all_nodes, 0);
  }
  auto root = biCluster(insts, all_nodes);
  all_nodes.emplace_back(root);
  setParentId(root, all_nodes.size() - 1, all_nodes);
  return Topology<T>(all_nodes, all_nodes.size() - 1);
}

template <typename T>
TopoNode<T> Router::biCluster(const std::vector<CtsInstance*>& insts, std::vector<TopoNode<T>>& all_nodes) const
{
  if (insts.size() == 1) {
    auto inst = insts[0];
    T data;
    DataTraits<T>::setPoint(data, inst->get_location());
    DataTraits<T>::setId(data, inst->get_name());
    DataTraits<T>::setSubWirelength(data, inst->getSubWirelength());
    return TopoNode<T>{data, -1, -1, -1, inst->getSubWirelength()};
  }
  Kmeans<CtsInstance*> kmeans;
  auto clusters = kmeans(insts, insts.size() / 2, 2);
  auto left = biCluster(clusters[0], all_nodes);
  auto right = biCluster(clusters[1], all_nodes);
  T val;
  auto parent = TopoNode<T>{val, -1, static_cast<int>(all_nodes.size()), static_cast<int>(all_nodes.size() + 1), 0};
  all_nodes.emplace_back(left);
  all_nodes.emplace_back(right);
  return parent;
}

template <typename T>
void Router::setParentId(TopoNode<T>& node, const int& id, std::vector<TopoNode<T>>& all_nodes) const
{
  if (node.left() != -1) {
    auto& left_node = all_nodes[node.left()];
    setParentId(left_node, node.left(), all_nodes);
    left_node.parent(id);
  }
  if (node.right() != -1) {
    auto& right_node = all_nodes[node.right()];
    setParentId(right_node, node.right(), all_nodes);
    right_node.parent(id);
  }
  return;
}

void Router::init()
{
  CTSAPIInst.saveToLog("\n\nRouter Log");
  auto* config = CTSAPIInst.get_config();
  auto* design = CTSAPIInst.get_design();
  auto* db_wrapper = CTSAPIInst.get_db_wrapper();
  // report unit res & cap
  CTSAPIInst.saveToLog("\nRouter unit res: ", CTSAPIInst.getClockUnitRes());
  CTSAPIInst.saveToLog("Router unit cap: ", CTSAPIInst.getClockUnitCap());
  _end_index = 0;
  printLog();
  auto& clocks = design->get_clocks();
  for (auto& clock : clocks) {
    _clocks.push_back(clock);
    for (auto* clk_net : clock->get_clock_nets()) {
      auto& clk_pins = clk_net->get_pins();
      for (auto* clk_pin : clk_pins) {
        auto* clk_inst = clk_pin->get_instance();
        auto& inst_name = clk_inst->get_name();
        if (_name_to_inst.count(inst_name) == 0) {
          _name_to_inst[inst_name] = clk_inst;
        }
      }
    }
  }
  if (config->get_router_type() == "UST") {
    // std logic routing
    auto logic_nets = db_wrapper->get_logic_nets();
    for (auto logic_net : logic_nets) {
      if (logic_net->get_pins().size() >= 2) {
        auto eval_net = EvalNet(logic_net);
        CTSAPIInst.buildLogicRCTree(eval_net);
      }
    }
    auto constraints = CTSAPIInst.skewConstraints();
    _skew_scheduler = new SkewScheduler(constraints, config->get_skew_bound());
    // determine negative-weighted circle and build matrix D
    if (_skew_scheduler->haveNegativeWeightCycle()) {
      LOG_WARNING << "Negative-weighted circle exists in skew constraints";
      constraints = CTSAPIInst.fixSkewConstraints();
      _skew_scheduler = new SkewScheduler(constraints, config->get_skew_bound());
    }
    _skew_scheduler->buildSkewConstraint();
  }
}

void Router::printLog()
{
  LOG_INFO << "\033[1;31m";
  LOG_INFO << R"(                  _             )";
  LOG_INFO << R"(                 | |            )";
  LOG_INFO << R"(  _ __ ___  _   _| |_ ___ _ __  )";
  LOG_INFO << R"( | '__/ _ \| | | | __/ _ \ '__| )";
  LOG_INFO << R"( | | | (_) | |_| | ||  __/ |    )";
  LOG_INFO << R"( |_|  \___/ \__,_|\__\___|_|    )";
  LOG_INFO << "\033[0m";
  LOG_INFO << "Enter router!";
}

bool Router::haveSink(CtsNet* net) const
{
  for (auto* load_pin : net->get_load_pins()) {
    if (load_pin->get_pin_type() == CtsPinType::kClock) {
      return true;
    }
  }
  return false;
}

}  // namespace icts