/**
 * @file SlewAware.cc
 * @author Dawn Li (dawnli619215645@gmail.com)
 */
#include "SlewAware.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <queue>
#include <random>

#include "CTSAPI.hpp"
#include "CtsReport.h"
#include "Kmeans.h"
#include "Operator.h"
#include "Placer.h"
#include "log/Log.hh"
#include "time/Time.hh"

namespace icts {

TimingNode* SlewAware::slewAwareBase(std::vector<TimingNode*>& timing_nodes, TimingCalculator* timer)
{
  if (timing_nodes.empty()) {
    return nullptr;
  }
  // build cost calculator
  auto* cost_cal = new CostCalculator();
  cost_cal->initID(timing_nodes.size());

  auto exist_id = cost_cal->get_id_set();
  // cost init
  costBuild(timing_nodes, cost_cal, timer);
  // merge tree by min cost
  while (exist_id.size() > 1) {
    auto min_cost = cost_cal->minCost();
    // no more merge
    if (min_cost == std::numeric_limits<double>::max()) {
      cost_cal->costReset();
      std::vector<TimingNode*> unmerged_nodes;
      std::vector<TimingNode*> exist_nodes;
      for (auto itr = exist_id.begin(); itr != exist_id.end(); ++itr) {
        auto* node = timing_nodes[*itr];
        exist_nodes.emplace_back(node);
        if (node->is_merged()) {
          node->reset_merged();
          continue;
        }
        unmerged_nodes.emplace_back(node);
      }
      auto lower_delay_nodes = timer->screenNodes(unmerged_nodes);
      auto guide_center = timer->guideCenter(exist_nodes);
      auto guide_dist = timer->guideDist(exist_nodes);
      for (auto* node : lower_delay_nodes) {
        timer->insertBuffer(node, guide_center, guide_dist);
      }
      // cost rebuild
      costBuild(timing_nodes, cost_cal, timer);
      continue;
    }
    // init new node k from i and j (node k have been fix timing)
    auto min_cost_node = cost_cal->minCostNode();
    auto* node_i = timing_nodes[min_cost_node.get_pair().get_first()];
    auto* node_j = timing_nodes[min_cost_node.get_pair().get_second()];
    auto* node_k = timer->calcMergeNode(node_i, node_j);
    // update calculator (erase i and j and insert k)
    cost_cal->popMinCost();
    cost_cal->insertId(timing_nodes.size());
    // update exist node id (erase i and j and insert k)
    timing_nodes.emplace_back(node_k);
    exist_id = cost_cal->get_id_set();
    // update cost (exist node with k)
    for (auto itr = exist_id.begin(); itr != exist_id.end(); ++itr) {
      if (*itr == static_cast<int>(timing_nodes.size() - 1)) {
        continue;  // ignore self
      }
      auto* node = timing_nodes[*itr];
      double slew_wire = timer->calcMergeWireSlew(node, node_k);
      if (slew_wire < std::min(node->get_slew_constraint(), node_k->get_slew_constraint())) {
        double cost = timer->calcMergeCost(node, node_k);
        cost_cal->insertCost(*itr, timing_nodes.size() - 1, cost);
      }
    }
  }
  auto* root = timing_nodes[*exist_id.begin()];
  return root;
}

void SlewAware::slewAware()
{
  LOG_INFO << "#################### [Slew Aware] ####################";
  LOG_INFO << "[Slew Aware] build net: " << _net_name;
  LOG_INFO << "[Slew Aware] Flip-Flops num: " << _instances.size();
  CTSAPIInst.saveToLog("#################### [Slew Aware] ####################");
  CTSAPIInst.saveToLog("[Slew Aware] build net: ", _net_name);
  CTSAPIInst.saveToLog("[Slew Aware] Flip-Flops num: ", _instances.size());
  auto* config = CTSAPIInst.get_config();
  // init timing nodes
  auto max_sink_tran = config->get_max_sink_tran();
  std::vector<TimingNode*> timing_nodes;
  for (auto* inst : _instances) {
    auto* node = new TimingNode(inst);
    node->set_slew_constraint(max_sink_tran);
    node->set_cap_out(CTSAPIInst.getSinkCap(inst));
    node->set_type(TimingNodeType::kSink);
    timing_nodes.emplace_back(node);
  }

  // build timing calculator
  auto* timer = genTimer();

  auto* root = slewAwareBase(timing_nodes, timer);

  root->set_type(TimingNodeType::kBuffer);
  root->get_inst()->set_cell_master(config->get_buffer_types().front());
  root->set_slew_constraint(config->get_max_buf_tran());
  timer->timingPropagate(root);
  timer->updateTiming(root);
  timer->fixTiming(root);
  LOG_INFO << "[Slew Aware] Root Max delay: " << root->get_delay_max();
  LOG_INFO << "[Slew Aware] Root Min delay: " << root->get_delay_min();
  LOG_INFO << "[Slew Aware] Root skew: " << root->get_delay_max() - root->get_delay_min();
  CTSAPIInst.saveToLog("[Slew Aware] Root Max delay: ", root->get_delay_max());
  CTSAPIInst.saveToLog("[Slew Aware] Root Min delay: ", root->get_delay_min());
  CTSAPIInst.saveToLog("[Slew Aware] Root skew: ", root->get_delay_max() - root->get_delay_min());
  // top-down set instance
  topDown(root);
  // build cts topo
  buildClockTopo(root, _net_name);
  LOG_INFO << "[Slew Aware] Insert Buffer num: " << _clock_topos.size();
  CTSAPIInst.saveToLog("[Slew Aware] Insert Buffer num: ", _clock_topos.size());
}

void SlewAware::slewAwareByKmeans()
{
  LOG_INFO << "############### [Slew Aware By Kmeans] ###############";
  LOG_INFO << "[Slew Aware By Kmeans] build net: " << _net_name;
  LOG_INFO << "[Slew Aware By Kmeans] Flip-Flops num: " << _instances.size();
  CTSAPIInst.saveToLog("############### [Slew Aware By Kmeans] ###############");
  CTSAPIInst.saveToLog("[Slew Aware By Kmeans] build net: ", _net_name);
  CTSAPIInst.saveToLog("[Slew Aware By Kmeans] Flip-Flops num: ", _instances.size());
  auto* config = CTSAPIInst.get_config();
  // init timing nodes
  auto max_sink_tran = config->get_max_sink_tran();
  auto clusters = clustering(_instances, _instances.size() / 50);
  std::vector<TimingNode*> root_nodes;
  // level 1: cluster sinks build
  for (size_t i = 0; i < clusters.size(); ++i) {
    if (clusters[i].empty()) {
      continue;
    }
    std::vector<TimingNode*> timing_nodes;
    auto sinks = clusters[i];
    for (auto* inst : sinks) {
      auto* node = new TimingNode(inst);
      node->set_slew_constraint(max_sink_tran);
      node->set_cap_out(CTSAPIInst.getSinkCap(inst));
      node->set_type(TimingNodeType::kSink);
      timing_nodes.emplace_back(node);
    }

    LOG_INFO_IF((i + 1) % 10 == 0) << "Sink Level(" << i + 1 << "): Flip-Flops num: " << timing_nodes.size();

    // build timing calculator
    auto* timer = genTimer();
    timer->set_skew_bound(config->get_skew_bound() * 0.5);

    auto* root = slewAwareBase(timing_nodes, timer);
    root_nodes.emplace_back(root);
  }
  // level 2: buffer level build
  std::vector<TimingNode*> level_buffer_nodes;
  for (auto* root : root_nodes) {
    level_buffer_nodes.emplace_back(root);
  }
  LOG_INFO << "Slew Aware Buffer Level: node num: " << level_buffer_nodes.size();

  // build timing calculator
  auto* timer = genTimer();
  timer->set_skew_bound(config->get_skew_bound());

  auto* root = slewAwareBase(level_buffer_nodes, timer);

  root->set_type(TimingNodeType::kBuffer);
  root->get_inst()->set_cell_master(config->get_buffer_types().front());
  root->set_slew_constraint(config->get_max_buf_tran());
  timer->timingPropagate(root);
  timer->updateTiming(root);
  timer->fixTiming(root);
  LOG_INFO << "[Slew Aware By Kmeans] Root Max delay: " << root->get_delay_max();
  LOG_INFO << "[Slew Aware By Kmeans] Root Min delay: " << root->get_delay_min();
  LOG_INFO << "[Slew Aware By Kmeans] Root skew: " << root->get_delay_max() - root->get_delay_min();
  CTSAPIInst.saveToLog("[Slew Aware By Kmeans] Root Max delay: ", root->get_delay_max());
  CTSAPIInst.saveToLog("[Slew Aware By Kmeans] Root Min delay: ", root->get_delay_min());
  CTSAPIInst.saveToLog("[Slew Aware By Kmeans] Root skew: ", root->get_delay_max() - root->get_delay_min());
  // top-down set instance
  topDown(root);
  // build cts topo
  buildClockTopo(root, _net_name);
  LOG_INFO << "[Slew Aware By Kmeans] Insert Buffer num: " << _clock_topos.size();
  CTSAPIInst.saveToLog("[Slew Aware By Kmeans] Insert Buffer num: ", _clock_topos.size());
}

void SlewAware::slewAwareByBiCluster()
{
  LOG_INFO << "############# [Slew Aware By Bi-Cluster] #############";
  LOG_INFO << "[Slew Aware By Bi-Cluster] build net: " << _net_name;
  LOG_INFO << "[Slew Aware By Bi-Cluster] Flip-Flops num: " << _instances.size();
  CTSAPIInst.saveToLog("############# [Slew Aware By Bi-Cluster] #############");
  CTSAPIInst.saveToLog("[Slew Aware By Bi-Cluster] build net: ", _net_name);
  CTSAPIInst.saveToLog("[Slew Aware By Bi-Cluster] Flip-Flops num: ", _instances.size());

  auto* config = CTSAPIInst.get_config();
  auto* timer = genTimer();
  timer->set_skew_bound(config->get_skew_bound());
  auto* root = biCluster(_instances);
  recursiveMerge(root, timer);
  if (!root->is_buffer()) {
    root->set_type(TimingNodeType::kBuffer);
    root->get_inst()->set_cell_master(config->get_buffer_types().front());
    root->set_slew_constraint(config->get_max_buf_tran());
    timer->timingPropagate(root);
    timer->updateTiming(root);
    timer->fixTiming(root);
  }
  // top-down set instance
  topDown(root);
  // build cts topo
  buildClockTopo(root, _net_name);
  LOG_INFO << "[Slew Aware By Bi-Cluster] Insert Buffer num: " << _clock_topos.size();
  CTSAPIInst.saveToLog("[Slew Aware By Bi-Cluster] Insert Buffer num: ", _clock_topos.size());
}

void SlewAware::recursiveMerge(TimingNode* root, TimingCalculator* timer)
{
  auto* left = root->get_left();
  auto* right = root->get_right();
  if (left) {
    recursiveMerge(root->get_left(), timer);
  }
  if (right) {
    recursiveMerge(root->get_right(), timer);
  }
  if (left && right) {
    timer->mergeNode(root);
  }
}

void SlewAware::costBuild(const std::vector<TimingNode*>& timing_nodes, CostCalculator* cost_cal, TimingCalculator* timer)
{
  // cost init
  auto exist_id = cost_cal->get_id_set();
  for (auto itr_i = exist_id.begin(); itr_i != exist_id.end(); ++itr_i) {
    for (auto itr_j = std::next(itr_i); itr_j != exist_id.end(); ++itr_j) {
      auto* node_i = timing_nodes[*itr_i];
      auto* node_j = timing_nodes[*itr_j];
      double slew_wire = timer->calcMergeWireSlew(node_i, node_j);
      if (slew_wire < std::min(node_i->get_slew_constraint(), node_j->get_slew_constraint())) {
        double cost = timer->calcMergeCost(node_i, node_j);
        cost_cal->insertCost(*itr_i, *itr_j, cost);
      }
    }
  }
}

void SlewAware::topDown(TimingNode* root)
{
  if (root == nullptr) {
    return;
  }
  if (root->get_parent()) {
    auto closest_point = pgl::closest_point(root->get_parent()->get_location(), root->get_join_segment());
    root->set_location(closest_point);
  } else {
    auto center = pgl::center(root->get_merge_region());
    root->set_location(center);
  }
  auto loc = root->get_location();
  root->set_join_segment(Segment(loc, loc));
  root->set_merge_region({Polygon({loc})});
  if (root->is_buffer()) {
    CTSAPIInst.placeInstance(root->get_inst());
  }
  auto* left = root->get_left();
  auto* right = root->get_right();
  topDown(left);
  topDown(right);
}

void SlewAware::buildClockTopo(TimingNode* root, const std::string& net_name)
{
  // step1: find all drivers
  std::vector<TimingNode*> roots = findDrivers(root);
  // step2: build ClockTopo for a root,
  // then set net name and driver name
  for (int i = roots.size() - 1; i >= 0; --i) {
    auto* root = roots[i];
    auto topo = makeTopo(root, net_name + "_" + std::to_string(i));
    _clock_topos.emplace_back(topo);
  }
}

ClockTopo SlewAware::makeTopo(TimingNode* root, const std::string& net_name)
{
  ClockTopo topo(net_name);
  root->set_name(net_name + "_buf");
  _timing_node_map[root->get_name()] = root;
  CTSAPIInst.addTimingNode(root);
  topo.add_driver(root->get_inst());
  std::queue<TimingNode*> q;
  if (root->get_left()) {
    q.push(root->get_left());
  }
  if (root->get_right()) {
    q.push(root->get_right());
  }
  int steiner_id = 0;
  while (!q.empty()) {
    auto* current = q.front();
    q.pop();
    if (!current) {
      continue;
    }

    if (current->is_steiner()) {
      current->set_name("steiner_" + std::to_string(steiner_id++));
    } else {
      topo.add_load(current->get_inst());
    }
    // add wire from parent
    auto current_loc = current->get_location();
    auto* parent = current->get_parent();
    auto parent_loc = parent->get_location();
    auto* config = CTSAPIInst.get_config();
    if (current->get_need_snake() > 0) {
      auto require_snake = std::ceil(current->get_need_snake() * config->get_micron_dbu());
      auto dist = pgl::manhattan_distance(parent_loc, current_loc);
      auto snake_dist = require_snake - dist;
      auto delta_x = current_loc.x() - parent_loc.x();
      auto direction = delta_x > 0 ? 1 : -1;
      auto snake_p1 = Point(parent_loc.x() + direction * snake_dist / 2, parent_loc.y());
      auto snake_p2 = Point(parent_loc.x() + direction * snake_dist / 2, current_loc.y());
      if (!(CTSAPIInst.isInDie(snake_p1) && CTSAPIInst.isInDie(snake_p2))) {  // is in die
        snake_p1 = Point(current_loc.x() - direction * snake_dist / 2, parent_loc.y());
        snake_p2 = Point(current_loc.x() - direction * snake_dist / 2, current_loc.y());
      }
      std::vector<std::string> name_vec
          = {parent->get_name(), "steiner_" + std::to_string(steiner_id++), "steiner_" + std::to_string(steiner_id++), current->get_name()};
      std::vector<Point> point_vec = {parent_loc, snake_p1, snake_p2, current_loc};
      for (size_t i = 0; i < name_vec.size() - 1; ++i) {
        topo.add_signal_wire(CtsSignalWire(Endpoint(name_vec[i], point_vec[i]), Endpoint(name_vec[i + 1], point_vec[i + 1])));
      }
    } else {
      if (pgl::rectilinear(parent_loc, current_loc)) {
        topo.add_signal_wire(CtsSignalWire(Endpoint(parent->get_name(), parent_loc), Endpoint(current->get_name(), current_loc)));
      } else {
        auto trunk_loc = Point(parent_loc.x(), current_loc.y());
        auto trunk_name = "steiner_" + std::to_string(steiner_id++);
        topo.add_signal_wire(CtsSignalWire(Endpoint(parent->get_name(), parent_loc), Endpoint(trunk_name, trunk_loc)));
        topo.add_signal_wire(CtsSignalWire(Endpoint(trunk_name, trunk_loc), Endpoint(current->get_name(), current_loc)));
      }
    }
    if (current->is_steiner()) {
      q.push(current->get_left());
      q.push(current->get_right());
    }
  }
  return topo;
}

std::vector<TimingNode*> SlewAware::findDrivers(TimingNode* root)
{
  std::vector<TimingNode*> roots;
  std::queue<TimingNode*> q;
  q.push(root);
  while (!q.empty()) {
    auto* current = q.front();
    q.pop();
    if (!current) {
      continue;
    }
    if (current->is_buffer() && (current->get_left() || current->get_right())) {
      roots.emplace_back(current);
    }
    q.push(current->get_left());
    q.push(current->get_right());
  }
  return roots;
}

std::vector<std::vector<CtsInstance*>> SlewAware::clustering(const std::vector<CtsInstance*>& insts, const int& cluster_size)
{
  std::vector<std::vector<CtsInstance*>> clusters;
  auto cluster_num = std::ceil(insts.size() / cluster_size);
  auto temp_clusters = manhattanKmeans(insts, cluster_num);
  for (auto itr = temp_clusters.begin(); itr != temp_clusters.end(); ++itr) {
    auto cluster = *itr;
    if (cluster.size() <= 1.2 * cluster_size) {
      clusters.emplace_back(cluster);
    } else {
      auto sub_cluster_num = std::ceil(cluster.size() / cluster_size);
      auto sub_clusters = manhattanKmeans(cluster, sub_cluster_num);
      for (auto sub_cluster : sub_clusters) {
        clusters.emplace_back(sub_cluster);
      }
    }
  }
  return clusters;
}

TimingNode* SlewAware::biCluster(const std::vector<CtsInstance*>& insts)
{
  if (insts.size() == 1) {
    auto* node = new TimingNode(insts[0]);
    auto* config = CTSAPIInst.get_config();
    auto max_sink_tran = config->get_max_sink_tran();
    auto sink_cap = CTSAPIInst.getSinkCap(insts[0]);
    node->set_type(TimingNodeType::kSink);
    node->set_slew_constraint(max_sink_tran);
    node->set_cap_out(sink_cap);
    return node;
  }
  if (insts.size() == 2) {
    auto* left = biCluster(std::vector<CtsInstance*>{insts[0]});
    auto* right = biCluster(std::vector<CtsInstance*>{insts[1]});
    return new TimingNode(left, right);
  }
  auto clusters = manhattanKmeans(insts, 2);
  auto* left = biCluster(clusters[0]);
  auto* right = biCluster(clusters[1]);
  return new TimingNode(left, right);
}

vector<vector<CtsInstance*>> SlewAware::manhattanKmeans(const vector<CtsInstance*>& instances, const int& k, const int& max_iterations)
{
  std::vector<CtsPoint<int64_t>> centers;
  int num_instances = instances.size();
  std::vector<int> assignments(num_instances);

  // Randomly choose first center from instances
  // std::random_device rd;
  // std::mt19937 gen(rd());
  std::mt19937::result_type seed = 0;
  std::mt19937 gen(seed);
  std::uniform_int_distribution<> dis(0, num_instances - 1);
  centers.push_back(instances[dis(gen)]->get_location());
  // Choose k-1 remaining centers using kmeans++ algorithm
  while (static_cast<int>(centers.size()) < k) {
    std::vector<double> distances(num_instances, std::numeric_limits<double>::max());
    for (int i = 0; i < num_instances; i++) {
      CtsPoint<int64_t> instance_location = instances[i]->get_location();
      double min_distance = std::numeric_limits<double>::max();
      for (size_t j = 0; j < centers.size(); j++) {
        double distance = pgl::manhattan_distance(instance_location, centers[j]);
        min_distance = std::min(min_distance, distance);
      }
      distances[i] = min_distance * min_distance;  // square distance
    }

    std::discrete_distribution<> distribution(distances.begin(), distances.end());
    int selected_index = distribution(gen);
    centers.emplace_back(instances[selected_index]->get_location());
  }

  int num_iterations = 0;
  double prev_cap_variance = std::numeric_limits<double>::max();
  while (num_iterations++ < max_iterations) {
    // Assignment step
    for (int i = 0; i < num_instances; i++) {
      CtsPoint<int64_t> instance_location = instances[i]->get_location();
      double min_distance = std::numeric_limits<double>::max();
      int min_center_index = -1;
      for (size_t j = 0; j < centers.size(); j++) {
        double distance = pgl::manhattan_distance(instance_location, centers[j]);
        if (distance < min_distance) {
          min_distance = distance;
          min_center_index = j;
        }
      }
      assignments[i] = min_center_index;
    }
    // Check cap variance
    std::vector<double> cluster_cap(k, 0);
    for (int i = 0; i < num_instances; i++) {
      int center_index = assignments[i];
      cluster_cap[center_index] += CTSAPIInst.getSinkCap(instances[i]);
    }
    double sum = std::accumulate(std::begin(cluster_cap), std::end(cluster_cap), 0.0);
    double mean = sum / cluster_cap.size();
    double cap_variance = 0.0;
    std::for_each(std::begin(cluster_cap), std::end(cluster_cap), [&](const double d) { cap_variance += pow(d - mean, 2); });
    cap_variance /= cluster_cap.size();
    // Check for convergence
    if (cap_variance < prev_cap_variance) {
      break;
    }
    prev_cap_variance = std::min(prev_cap_variance, cap_variance);
    // Update step
    std::vector<CtsPoint<int64_t>> new_centers(k, CtsPoint<int64_t>(0, 0));
    std::vector<int> center_counts(k, 0);
    for (int i = 0; i < num_instances; i++) {
      Point instance_location = instances[i]->get_location();
      int center_index = assignments[i];
      new_centers[center_index] += instance_location;
      center_counts[center_index]++;
    }
    for (int i = 0; i < k; i++) {
      if (center_counts[i] > 0) {
        new_centers[i] /= center_counts[i];
      }
    }
    centers = new_centers;
  }
  // Collect results
  std::vector<std::vector<CtsInstance*>> clusters(k);
  for (int i = 0; i < num_instances; i++) {
    int center_index = assignments[i];
    clusters[center_index].push_back(instances[i]);
  }

  return clusters;
}

TimingNode* SlewAware::findTimingNode(const std::string& buffer_name)
{
  if (_timing_node_map.find(buffer_name) == _timing_node_map.end()) {
    LOG_WARNING << "can't find node name: " << buffer_name;
    return nullptr;
  }
  return _timing_node_map[buffer_name];
}

TimingCalculator* SlewAware::genTimer() const
{
  auto* timer = new TimingCalculator();
#if (defined PY_MODEL) && (defined USE_EXTERNAL_MODEL)
  timer->set_external_model(CTSAPIInst.findExternalModel(_net_name));
#endif
  return timer;
}

std::string SlewAware::getInsertTypeStr(TimingNode* node) const
{
  std::string insert_type_str;
  switch (node->get_insert_type()) {
    case InsertType::kMake:
      insert_type_str = "Make";
      break;
    case InsertType::kMultiple:
      insert_type_str = "Multiple";
      break;
    case InsertType::kTopInsert:
      insert_type_str = "TopInsert";
      break;
    case InsertType::kBreakWire:
      insert_type_str = "BreakWire";
      break;
    case InsertType::kNone:
      insert_type_str = "None";
      break;
    default:
      insert_type_str = "Unknown";
      break;
  }
  return insert_type_str;
}

std::string SlewAware::getInsertTypeEncodeStr(TimingNode* node) const
{
  std::string insert_type_encode;
  switch (node->get_insert_type()) {
    case InsertType::kMake:
      insert_type_encode = "1,0,0,0";
      break;
    case InsertType::kMultiple:
      insert_type_encode = "0,1,0,0";
      break;
    case InsertType::kTopInsert:
      insert_type_encode = "0,0,1,0";
      break;
    case InsertType::kBreakWire:
      insert_type_encode = "0,0,0,1";
      break;
    default:
      insert_type_encode = "0,0,0,0";
      break;
  }
  return insert_type_encode;
}

void SlewAware::timingLog() const
{
  auto timing_rpt = CtsReportTable::createReportTable("Timing Log", CtsReportType::kTIMING_NODE_LOG);
  for (auto [_, node] : _timing_node_map) {
    auto timer = TimingCalculator();
    double cap_out = timer.calcCapLoad(node);
    auto loc_str = CTSAPIInst.toString("(", node->get_location().x(), ",", node->get_location().y(), ")");
    (*timing_rpt) << node->get_id() << node->get_name() << node->get_need_snake() << node->get_net_length() << loc_str
                  << node->get_delay_min() << node->get_delay_max() << getInsertTypeStr(node) << node->get_slew_in() << cap_out
                  << node->get_insertion_delay() << TABLE_ENDLINE;
  }
  auto dir = CTSAPIInst.get_config()->get_sta_workspace() + "/timing_log";
  auto file_name = _net_name + "_timing_log.rpt";
  auto save_path = dir + "/" + file_name;
  if (!std::filesystem::exists(dir)) {
    std::filesystem::create_directories(dir);
  }
  std::ofstream outfile(save_path);
  outfile << "Generate the report at " << Time::getNowWallTime() << std::endl;
  outfile << timing_rpt->c_str();
  outfile.close();
}

void SlewAware::saveTrainingData() const
{
  auto dir = CTSAPIInst.get_config()->get_sta_workspace() + "/training_data";
  auto file_name = _net_name + "_timing_data.csv";
  auto save_path = dir + "/" + file_name;
  if (!std::filesystem::exists(dir)) {
    std::filesystem::create_directories(dir);
  }

  std::ofstream outfile(save_path, std::ios::app);
  if (outfile.tellp() == 0) {
    outfile << "type_make,type_multiple,type_top_insert,type_break_wire,"
               "level,cap_out,fanout,min_delay,max_delay,slew_in,insertion_delay"
            << std::endl;
  }
  auto timer = TimingCalculator();
  for (auto [_, node] : _timing_node_map) {
    if (node->get_insertion_delay() == 0) {
      continue;
    }
    auto insert_type_encode = getInsertTypeEncodeStr(node);
    auto level = node->get_level();
    double cap_out = timer.calcCapLoad(node);
    auto fanout = node->getFanout();
    auto min_delay = node->get_delay_min() - node->get_insertion_delay();
    auto max_delay = node->get_delay_max() - node->get_insertion_delay();
    auto slew_in = node->get_slew_in();
    auto insertion_delay = node->get_insertion_delay();
    outfile << insert_type_encode << "," << level << "," << cap_out << "," << fanout << "," << min_delay << "," << max_delay << ","
            << slew_in << "," << insertion_delay << std::endl;
  }
  outfile.close();
}
}  // namespace icts